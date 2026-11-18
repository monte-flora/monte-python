import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

try:  # compiled core (monte_python/_rust); _OverlapStepper below is the numpy fallback of the same step
    from . import _rust
    if not hasattr(_rust, "OverlapTracker"):  # an older build without the tracker
        _rust = None
except ImportError:
    _rust = None


def calc_dist(xy1, xy2):
    """Euclidean distance between xy1 = (x1, y1) and xy2 = (x2, y2); 0 if either is NaN (missing)."""
    dist = np.hypot(xy1[0] - xy2[0], xy1[-1] - xy2[-1])
    if np.isnan(dist):
        return 0.0
    return float(dist)


def _pair_overlaps(data_a, data_b):
    """(labels_a, labels_b, n_overlap) for every pair of overlapping objects, sorted by (label_a, label_b), and
    the per-label areas of both grids, from one pass over the pixels."""
    a = np.asarray(data_a).ravel().astype(np.int64)
    b = np.asarray(data_b).ravel().astype(np.int64)
    both = (a > 0) & (b > 0)
    m = int(b.max()) + 1 if b.size else 1
    keys, counts = np.unique(a[both] * m + b[both], return_counts=True)
    return keys // m, keys % m, counts


def _areas(data):
    lab = np.asarray(data).ravel().astype(np.int64)
    lab = lab[lab > 0]
    if lab.size == 0:
        return {}
    cnt = np.bincount(lab)
    nz = np.flatnonzero(cnt)
    return dict(zip(nz.tolist(), cnt[nz].tolist()))


class _OverlapStepper:
    """Streaming form of ObjectTracker (numpy; the Rust ``_rust.OverlapTracker`` is the same algorithm).

    Per frame t (> 0): overlap pairs with frame t - 1 scored (1 if cap else 2) n / (|A| + |B|), kept if
    > percent_overlap, ranked by score (ties: previous track id, then label, ascending), greedy one-to-one; an
    unmatched object starts a track (pre-mend ids in label order, as ObjectTracker.relabel numbers them). With
    mending, unmended tracks that ended at frame t - g (g = 1 .. mend_gap), in (g, id) order, take the nearest
    unclaimed new track of frame t whose start centroid lies within mend_dist of their end centroid + g x mean
    motion (truncated centroids, as mend_broken_tracks); output ids follow the mend chains to their root.
    """

    def __init__(self, percent_overlap, cap_overlap, mend_tracks, mend_dist, mend_gap=1):
        self.po, self.k = float(percent_overlap), 1.0 if cap_overlap else 2.0
        self.mend, self.mend_dist, self.mend_gap = bool(mend_tracks), float(mend_dist), int(mend_gap)
        self.dormant = {}      # pre-mend id -> end frame: ended, unmended tracks still within mend_gap
        self.frame = -1
        self.prev_idx = None   # previous frame: per pixel 0 or 1 + object index (int64)
        self.prev_pid = None   # previous frame: pre-mend track id per object
        self.prev_area = None
        self.next_id = 1
        self.root = {}         # pre-mend id -> output id
        self.trk = {}          # pre-mend id -> [start, x0, y0, x1, y1, n] (x = row, y = column, truncated)
        self._records, self._events = [], []

    def records(self):
        return self._records

    def events(self):
        return self._events

    def step(self, labels):
        self.frame += 1
        t = self.frame
        flat = labels.ravel()
        nz = np.flatnonzero(flat > 0)
        v = flat[nz].astype(np.int64)
        if v.size and v.max() <= 4 * flat.size + 65536:  # dense ranks (no sort of the grid)
            lab = np.flatnonzero(np.bincount(v))
            rank = np.zeros(int(lab[-1]) + 1, np.int64)
            rank[lab] = np.arange(1, lab.size + 1)
            iv = rank[v]
        else:
            lab, iv = np.unique(v, return_inverse=True)
            iv = iv.astype(np.int64).ravel() + 1
        idx = np.zeros(flat.size, np.int64)
        idx[nz] = iv
        nobj = lab.size
        area = np.bincount(iv, minlength=nobj + 1)[1:]
        rows, cols = np.divmod(nz, labels.shape[1])
        sr = np.bincount(iv, rows, minlength=nobj + 1)[1:]
        sc = np.bincount(iv, cols, minlength=nobj + 1)[1:]
        cy, cx = sr / np.maximum(area, 1), sc / np.maximum(area, 1)
        pid = np.zeros(nobj, np.int64)
        matched = np.zeros(nobj, bool)
        pa_l = pb_l = None
        if self.prev_idx is not None and self.prev_idx.shape == idx.shape and nobj and self.prev_pid.size:
            a, b = self.prev_idx, idx
            both = (a > 0) & (b > 0)
            keys, n = np.unique((a[both] - 1) * nobj + (b[both] - 1), return_counts=True)
            pa, pb = keys // nobj, keys % nobj
            pa_l, pb_l = pa, pb
            score = self.k * n / (self.prev_area[pa] + area[pb])
            keep = score > self.po
            pa, pb, score = pa[keep], pb[keep], score[keep]
            order = np.lexsort((pb, self.prev_pid[pa], -score))
            used_a = set()
            for i in order.tolist():
                ia, ib = int(pa[i]), int(pb[i])
                if ia in used_a or matched[ib]:
                    continue
                used_a.add(ia); matched[ib] = True
                pid[ib] = self.prev_pid[ia]
        elif self.prev_idx is not None and self.prev_idx.shape != idx.shape:
            raise ValueError(f"frame {t}: shape {labels.shape} differs from the previous frame")
        new = np.flatnonzero(~matched)
        for ib in new.tolist():
            pid[ib] = self.next_id
            self.root[self.next_id] = self.next_id
            self.next_id += 1
        xs, ys = cy.astype(np.int64), cx.astype(np.int64)  # int() of the centroid, as get_track_path
        for ib in new.tolist():
            self.trk[int(pid[ib])] = [t, int(xs[ib]), int(ys[ib]), int(xs[ib]), int(ys[ib]), 0]
        if self.prev_pid is not None:
            cont = set(pid[matched].tolist())
            ended = sorted(p for p in self.prev_pid.tolist() if p not in cont)
            if self.mend:
                for L in ended:
                    self.dormant[L] = t - 1
            mended = set()
            if self.mend and len(new) and self.dormant:
                cand = [int(pid[ib]) for ib in new.tolist()]
                new_lab = dict(zip(cand, new.tolist()))
                taken = set()
                for L in sorted(self.dormant, key=lambda q: (-self.dormant[q], q)):
                    g = t - self.dormant[L]
                    s0, x0, y0, x1, y1, nfr = self.trk[L]
                    dx = (x1 - x0) / (nfr - 1) if nfr > 1 else 0.0
                    dy = (y1 - y0) / (nfr - 1) if nfr > 1 else 0.0
                    xp, yp = x1 + g * dx, y1 + g * dy
                    best, bd = None, None
                    for S in cand:
                        if S in taken:
                            continue
                        o = self.trk[S]
                        d = float(np.hypot(xp - o[1], yp - o[2]))
                        if d <= self.mend_dist and (bd is None or d < bd):
                            best, bd = S, d
                    if best is not None:
                        taken.add(best)
                        mended.add(L)
                        self.root[best] = self.root[L]
                        self._events.append((t, "mend", self.root[L], 0, int(lab[new_lab[best]])))
            if pa_l is not None:  # split / merge events (overlapping pairs that did not match), final ids
                pidl = pid.tolist()
                for ia, ib in zip(pa_l.tolist(), pb_l.tolist()):
                    ap, bp = int(self.prev_pid[ia]), pidl[ib]
                    if ap == bp:
                        continue
                    if not matched[ib]:
                        self._events.append((t, "split", self.root[bp], self.root[ap], int(lab[ib])))
                    elif ap not in cont:
                        self._events.append((t, "merge", self.root[ap], self.root[bp], int(lab[ib])))
            if not self.mend:
                for L in ended:  # ended tracks are never seen again
                    self.trk.pop(L, None)
            for L in [q for q, e in self.dormant.items() if q in mended or t - e >= self.mend_gap]:
                del self.dormant[L]  # mended, or too old to be mended at t + 1
                self.trk.pop(L, None)
        out = np.array([self.root[p] for p in pid.tolist()], np.int32)
        for ib, p in enumerate(pid.tolist()):
            r = self.trk[p]
            r[3], r[4], r[5] = int(xs[ib]), int(ys[ib]), r[5] + 1
            self._records.append((t, int(out[ib]), int(lab[ib]), int(area[ib]), float(cy[ib]), float(cx[ib])))
        self.prev_idx, self.prev_pid, self.prev_area = idx, pid, area
        lut = np.zeros(nobj + 1, np.int32)
        lut[1:] = out
        return lut[idx].reshape(labels.shape)


class ObjectTracker:
    """
    ObjectTracker links labelled objects from time step to time step by overlap (designed for 5-10 min steps).

    Each pair of consecutive frames is matched greedily one-to-one by overlap score (largest first; ties: lower
    track id, then lower label). On a merge (several old objects overlap one new object) the best-overlapping old
    track continues and the others end; on a split the best-overlapping new object continues the track and the
    others start new tracks. Unmatched objects start new tracks.

    With mend_tracks=True (default since v1.3) a track that ended at frame t is continued by the nearest new
    track starting at t + 1 whose start centroid is within ``mend_dist`` grid points of the ended track's end
    centroid + mean motion (centroids truncated to grid points; ties: lowest id). With ``mend_gap`` = G > 1 a
    track missing for up to G - 1 frames can be mended too (start at t + g, projected g x motion; at each frame
    the most recently ended tracks choose first). Mending a track twice keeps the first id. There is no motion advection in the matching itself (see StreamingTracker for that).

    Tracking streams (``step``, ``track_iter``): memory is two frames plus the active tracks, and the ids
    returned for a frame are final. ``track`` stacks the streamed frames.

    Parameters
    -----------
    percent_overlap : float, default = 0.0
        Minimum overlap score for a match (score > percent_overlap). The score is the Dice coefficient
        2 |A n B| / (|A| + |B|) (1 for identical objects).
    mend_tracks : bool, default True
        Mend broken tracks (above). v1.2 and earlier defaulted to False (and its mending had no effect).
    mend_dist : float, default 3
        Maximum mending distance (grid points, Euclidean). The default 3 was set for 3-km WoFS at 5-min steps;
        scale it with the grid spacing and the step (storm motion per step).
    mend_gap : int, default 1
        Frames from a track's last object to the mending object (1 = the next frame, as mend_broken_tracks).
        ``track`` / ``step`` only: the batch reference (``_track_batch``) mends with gap 1.
    cap_overlap : bool, default False
        True: the original score |A n B| / (|A| + |B|) (at most 0.5), so the same percent_overlap is twice as
        strict as with Dice. Same ranking; with percent_overlap=0 the tracks are identical.
    use_rust : bool or None, default None
        None uses the compiled core (monte_python._rust) when it is installed, else the numpy implementation of
        the same algorithm (~3x slower per frame on CONUS); True requires it; False forces numpy.

    Labels: 2-D integer grids, values <= 0 are background (e.g. EnhancedWatershed's -1 / -3), all frames the
    same shape.

    Attributes
    ----------
    tracks : int32 array (NT, NY, NX), after ``track``.
    trackprops : pandas.DataFrame of track duration and length, after ``track``.
    table() / events() : per-object records and split / merge / mend events of the run (also streaming).

    Author: Montgomery Flora (git: monte-flora)
    Email : monte.flora@noaa.gov
    """
    def __init__(self, percent_overlap=0.0, mend_tracks=True, mend_dist=3, cap_overlap=False, use_rust=None,
                 mend_gap=1):
        if use_rust and _rust is None:
            raise ImportError("use_rust=True but monte_python._rust is not built (pip install with a Rust toolchain)")
        self._percent_overlap = float(percent_overlap)
        self._mend_tracks = bool(mend_tracks)
        self._mend_dist = float(mend_dist)
        if int(mend_gap) < 1:
            raise ValueError("mend_gap must be >= 1")
        self._mend_gap = int(mend_gap)
        self._cap_overlap = bool(cap_overlap)
        self.use_rust = (_rust is not None) if use_rust is None else bool(use_rust)
        self.reset()

    def reset(self):
        """Forget the streaming state (``step``): the next frame starts frame 0 with fresh track ids."""
        impl = _rust.OverlapTracker if self.use_rust else _OverlapStepper
        self._stepper = impl(self._percent_overlap, self._cap_overlap, self._mend_tracks, self._mend_dist,
                             self._mend_gap)

    def step(self, labels):
        """
        Track one frame (streaming): returns the track-id grid (int32, same shape) of this 2-D label grid. The
        ids are final when returned (mending of a track that ended at the previous frame is decided here), and
        equal what ``track`` gives for the stacked frames. Memory: two frames plus per-track summaries.
        """
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError('labels has to be 2 dimensional!')
        if labels.dtype != np.int32:
            if labels.size and (labels.max() > np.iinfo(np.int32).max or labels.min() < np.iinfo(np.int32).min):
                raise ValueError('labels exceed the int32 range')
            labels = labels.astype(np.int32)
        return self._stepper.step(np.ascontiguousarray(labels))

    def track_iter(self, frames):
        """Generator (streaming): yields the track-id grid for each 2-D label grid of ``frames`` (any iterable)."""
        self.reset()
        for labels in frames:
            yield self.step(labels)

    def table(self):
        """Per-object records of the streaming run (frame, track, label, area, cy, cx) as a DataFrame."""
        return pd.DataFrame(self._stepper.records(), columns=['frame', 'track', 'label', 'area', 'cy', 'cx'])

    def events(self):
        """Split / merge / mend events of the streaming run (frame, kind, track, other_track, label) as a
        DataFrame, with final ids; ``label`` is the input label of the object at ``frame``. split: ``track``
        started at ``frame`` and its object overlapped ``other_track``'s at frame - 1; merge: ``track`` ended at
        frame - 1 and its object overlapped object ``label``, which continued ``other_track``; mend: ``track``
        ended at frame - 1 and continues as new object ``label`` (other_track 0)."""
        return pd.DataFrame(self._stepper.events(), columns=['frame', 'kind', 'track', 'other_track', 'label'])

    def track(self, data):
        """ Given a 3D labelled dataset (the first dimension being time)
            track the labelled data in time.

        Parameters
        ----------
        data : array-like of shape (NT, NY, NX) or any iterable of 2D arrays
            Labelled data (integer values) from different times.

        Returns
        ----------
        tracks : int32 array of shape (NT, NY, NX)
            Labelled data where interger values for different time steps
            are meaningful. I.e., the label 1 at different time indices
            is referencing the same object at different times.
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 3:
                raise ValueError('data has to be 3 dimensional!')
            tracks = np.empty(data.shape, np.int32)  # filled frame by frame (no list of frames + stack copy)
            for k, grid in enumerate(self.track_iter(data)):
                tracks[k] = grid
        else:
            tracks = np.stack(list(self.track_iter(data)))
        self.tracks = tracks
        return tracks

    def _track_batch(self, data):
        """The pre-v1.3 batch implementation (whole (NT, NY, NX) array in memory), kept as the reference that
        ``track`` equals (tests). The helpers it calls (match, check_for_mergers, check_for_splits, relabel,
        mend_broken_tracks, ...) stay public for compatibility; the merge and split checks are no-ops because
        match() is one-to-one."""
        """ Given a 3D labelled dataset (the first dimension being time) 
            track the labelled data in time. 
        
        Parameters
        ----------
        data : array-like of shape (NT, NY, NX) or list of 2D arrays 
            Labelled data (integer values) from different times.
           

        Returns
        ----------
        tracks : array-like of shape (NT, NY, NX) or list of 2D arrays 
            Labelled data where interger values for different time steps
            are meaningful. I.e., the label 1 at different time indices 
            is referencing the same object at different times. 
        """
        # If neccesary, convert to np.array. 
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Check that the data is 3-dimensional. 
        if np.ndim(data) != 3:
            raise ValueError('data has to be 3 dimensional!') 
        
        # Re-label the data such that each object has a unique label. 
        tracks = self.get_unique_labels(data)
        
        for t in np.arange(tracks.shape[0]-1):
            # Get these labels and the labels from the next time step
            # and match them based on percent overlap.
            current_data, future_data = tracks[t,:,:], tracks[t+1,:,:]
            labels_before, labels_after  = self.match(current_data, future_data)

            # Compute the area for the before and after objects. Used for determing
            # the retaining label in cases of merging/splitting.
            areas_before, areas_after = self._get_area(current_data), self._get_area(future_data)

            # Check for mergers.
            labels_before, labels_after = self.check_for_mergers(labels_before, labels_after, areas_before)

            # Check for splits.
            labels_before, labels_after = self.check_for_splits(labels_before, labels_after, areas_after)

            # This is where the tracking is done. We re-label the objects
            # at a future time if it is matched to a current object (one lookup-table pass).
            if labels_after:
                lut = np.arange(int(future_data.max()) + 1, dtype=tracks.dtype)
                lut[np.asarray(labels_after, dtype=np.int64)] = labels_before
                tracks[t+1] = lut[future_data]

        # Do a final re-label so that np.max(relabel_data) == number of tracked data.
        tracks = self.relabel(tracks)

        # Apply a mend to broken tracks (optional).
        if self._mend_tracks:
            tracks = self.mend_broken_tracks(tracks, mend_dist=self._mend_dist)

        return tracks
    
    @property
    def trackprops(self,):
        """
        Similar to skimage.measure.regionprops_table, this attribute returns 
        a dataframe with track properties. 
        
        Properties:
        -label : Object integer label 
        -duration : Number of timesteps a given object exists for. 
        -length: Total distance traversed per object. 
        """
        # TODO: Add geopandas and then add the x-,y- tracks to the trackprops
        
        if not hasattr(self, 'tracks'):
            raise AttributeError('Must create the tracked data using .track_data()!') 
       
        x_cent, y_cent = self.get_track_path(self.tracks)
        labels = list(x_cent.keys())
        duration = [np.count_nonzero(~np.isnan(x_cent[label])) for label in labels]
        
        xy_pairs = [list(zip(x_cent[label],y_cent[label])) for label in labels]
        length = [ np.sum([calc_dist(xy[i], xy[i+1]) for i in range(len(xy)-1)]) for xy in xy_pairs]
        
        data = { 
                 'labels' : labels, 
                 'duration' : duration,
                 'length' : length
        }
    
        return pd.DataFrame(data) 
    

    def _get_area(self, arr):
        """
        Get the area of each object and return a dict.
        """
        return _areas(arr)
        

    def check_for_mergers(self, labels_before, labels_after, areas_before):
        """
        Mergers are cases where there are non-unique labels in the after step
        (i.e., two or more labels become one). For longer tracks, 
        the merged object label is inherited from the largest object in 
        the merger. 
    
        E.g., 
    
        labels_before = [1,2,3,4,4] - > [2,3,4,4] 
        labels_after  = [5,5,6,7,8] - > [5,6,7,8]
    
        Parameters
        --------------------
        labels_before : list of ints 
        labels_after : list of ints 
        areas_before : dict 
            object label, area pairs for the prior objects 
    
        Returns
        --------------------
        labels_before, labels_after 
        
        """
        # Determine if there is a merged based on non-unqique labels. 
        unique_labels_after, counts_after = np.unique(labels_after, return_counts=True)
        if any(counts_after>1):
            # Get the labels that non-unique. 
            merged_labels = unique_labels_after[counts_after>1]

            for label in merged_labels:
                # This should be 2 or more labels (which are being merged together).
                potential_label_for_merged_obj = [l for i, l in enumerate(labels_before) if labels_after[i] == label]
                # Sort the potential merged object labels by area. Keep the largest object and remove the 
                # others. 
                inds = np.argsort([areas_before[label] for label in potential_label_for_merged_obj])[::-1]
                labels_sorted = np.array(potential_label_for_merged_obj)[inds]
                for label in labels_sorted[1:]:
                    index = labels_before.index(label)
                    del labels_before[index]
                    del labels_after[index]
    
        return labels_before, labels_after

    def check_for_splits(self, labels_before, labels_after, areas_after):
        """
        Splits are cases where there are non-unique labels in the before step
        (i.e., one labels becomes two or more). For longer tracks, 
        of the split labels, the largest one inherits the before step label. 
    
    
        labels_before = [1,2,3,4,4] - > [1,2,3,4] 
        labels_after  = [5,5,6,7,8] - > [5,5,6,7]
    
    
        Parameters
        --------------------
        labels_before : list of ints 
        labels_after : list of ints 
        areas_after : dict 
            object label, area pairs for the future objects 
    
        Returns
        --------------------
        labels_before, labels_after 
        """
        unique_labels_before, counts_before = np.unique(labels_before, return_counts=True)
        if any(counts_before>1): 
            split_labels = unique_labels_before[counts_before>1]
    
            for label in split_labels: 
                # This should be 2 or more labels (which are being merged together).
                potential_label_for_split_obj = [l for i, l in enumerate(labels_after) if labels_before[i] == label]  
                # Sort the potential split object labels by area. Keep the largest object and remove the 
                # others. 
                inds = np.argsort([areas_after[label] for label in potential_label_for_split_obj])[::-1]
                labels_sorted = np.array(potential_label_for_split_obj)[inds]
        
                for label in labels_sorted[1:]:
                    index = labels_after.index(label)
                    del labels_before[index]
                    del labels_after[index]
    
        return labels_before, labels_after 
    
    
    def get_unique_labels(self, data):
        """Ensure that initially, each object for the different times have a unique label"""
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        unique_track_set = np.zeros(data.shape, dtype=np.int32)

        num = 1
        for i in range(len(data)):
            current_obj = np.asarray(data[i])
            labels = np.unique(current_obj)
            labels = labels[labels > 0]
            if labels.size == 0:
                continue
            # objects numbered num, num+1, ... in label order
            idx = np.searchsorted(labels, current_obj)
            obj = current_obj > 0
            unique_track_set[i][obj] = (idx[obj] + num).astype(np.int32)
            num += labels.size

        return unique_track_set

    def relabel(self, data):
        """Re-label data so that np.max(data) == number of objects."""
        data = np.asarray(data)
        # Ignore the zero label
        unique_labels, inverse = np.unique(data, return_inverse=True)
        offset = 1 if unique_labels.size and unique_labels[0] <= 0 else 0
        relabelled_data = (inverse.reshape(data.shape) + (1 - offset)).astype(data.dtype)
        relabelled_data[data <= 0] = 0
        return relabelled_data

    def match(self, data_a, data_b):
        """ Match two set of data valid at a single or multiple times.
        Parameters
        -------------
            object_set_a, 2D array or list of 2D arrays, object labels at a single or multiple times
            object_set_b, 2D array or list of 2D arrays, object labels at a single or multiple times
        Returns
        -------------
            Lists of matched labels in set a, matched labels in set b,
            and tuples of y- and x- components of centroid displacement of matched pairs
        """
        matched_object_set_a_labels  = [ ]
        matched_object_set_b_labels  = [ ]
        
        possible_matched_pairs = self.find_possible_matches(data_a, data_b) 

        # Reverse means large values first! 
        sorted_possible_matched_pairs  = sorted(possible_matched_pairs, key=possible_matched_pairs.get, reverse=True) 
        for label_a, label_b in sorted_possible_matched_pairs:
            # One to one matching is deprecated! 
            #if self.one_to_one:
            if label_a not in matched_object_set_a_labels and label_b not in matched_object_set_b_labels: 
                #otherwise pair[0] has already been matched!
                matched_object_set_a_labels.append(label_a)
                matched_object_set_b_labels.append(label_b)
            #else:
            #    if label_a not in matched_object_set_a_labels: 
            #        #otherwise pair[0] has already been matched!
            #        matched_object_set_a_labels.append(label_a)
            #        matched_object_set_b_labels.append(label_b)
       
        return matched_object_set_a_labels, matched_object_set_b_labels
    
    def percent_intersection(self, region_a, region_b):
        """
        Compute percent overlap with the region coordinates0
        """
        # Dice coefficient 2 |A n B| / (|A| + |B|); cap_overlap=True: |A n B| / (|A| + |B|) (capped at 0.5)
        region_a_coords = set(map(tuple, region_a.coords))
        region_b_coords = set(map(tuple, region_b.coords))
        denom = len(region_a_coords) + len(region_b_coords)
        return (1.0 if self._cap_overlap else 2.0) * len(region_a_coords & region_b_coords) / denom
    
    def find_possible_matches(self, data_a, data_b): 
        """ Finds matches based on amount of intersection between data at time = t and time = t+1.
        Parameters 
        -----------
            regionprops_set_a, skimage.measure.regionprops for object_set_a
            regionprops_set_b, skimage.measure.regionprops for object_set_b
            
        Returns 
        ----------
            Dictionary of tuples of possible matched object pairs associated with their total interest score 
            Dictionary of y- and x-component of centroid displacement of possible matched object pairs             
        """
        
        # Only overlapping pairs can score > 0: count them in one pass (was all pairs of regionprops).
        la, lb, inter = _pair_overlaps(data_a, data_b)
        area_a, area_b = _areas(data_a), _areas(data_b)
        possible_matched_pairs = {}
        for a, b, n in zip(la.tolist(), lb.tolist(), inter.tolist()):
            percent_overlap = (1.0 if self._cap_overlap else 2.0) * n / (area_a[a] + area_b[b])
            if percent_overlap > self._percent_overlap:
                possible_matched_pairs[(a, b)] = percent_overlap
        return possible_matched_pairs
        
    def get_centroid(self, df, label):
        try:
            df=df.loc[df['label'] == label]
            x_cent, y_cent = df['centroid-0'], df['centroid-1']
            x_cent=int(x_cent.iloc[0])
            y_cent=int(y_cent.iloc[0])
        except Exception:
            return np.nan, np.nan

        return x_cent, y_cent

    def get_track_path(self, tracks):
        """ Create track path: per label, the (truncated) centroid row ('x') and column ('y') per time, NaN
        where the label is absent. """
        unique_labels = np.unique(tracks)
        unique_labels = unique_labels[unique_labels > 0]
        centroid_x = {l: [np.nan] * len(tracks) for l in unique_labels}
        centroid_y = {l: [np.nan] * len(tracks) for l in unique_labels}
        for t, frame in enumerate(tracks):
            props = regionprops_table(np.asarray(frame), properties=['label', 'centroid'])
            for l, cx, cy in zip(props['label'], props['centroid-0'], props['centroid-1']):
                centroid_x[l][t] = int(cx)
                centroid_y[l][t] = int(cy)
        return centroid_x, centroid_y

    def find_track_start_and_end(self, data):
        """
        Based on the x-centriod or y-centroid values for a track, 
        determine when the time index when the track starts and stops. 
        """
        time_indices = [i for i, v in enumerate(data) if not np.isnan([v])]
        return time_indices[0], time_indices[-1]
 
    def mend_broken_tracks(self, tracks, mend_dist=3):
        """
        Mend broken tracks by project track ends forward
        in time based on estimated storm motion and
        search for tracks that start in that projected area.
        If close enough, assume that those two tracks
        should be combined. Re-label that new tracks with
        the projected tracks label.

        A track ending at time t is projected one step with its mean centroid motion (0 for a one-time track);
        the closest track starting at t + 1 within ``mend_dist`` grid points (Euclidean) continues it. Chains
        are followed, so a track mended twice keeps the first label.
        """
        tracks = np.asarray(tracks)
        x_cent, y_cent = self.get_track_path(tracks)
        labels = list(x_cent.keys())
        if not labels:
            return np.copy(tracks)
        info = {}
        for label in labels:
            x = np.asarray(x_cent[label], dtype=float)
            y = np.asarray(y_cent[label], dtype=float)
            t_idx = np.flatnonzero(~np.isnan(x))
            if t_idx.size == 0:
                continue
            start, end = int(t_idx[0]), int(t_idx[-1])
            if t_idx.size > 1:
                steps = np.diff(t_idx)
                dx = float(np.mean(np.diff(x[t_idx]) / steps))
                dy = float(np.mean(np.diff(y[t_idx]) / steps))
            else:
                dx = dy = 0.0
            info[label] = dict(start=start, end=end, x0=x[start], y0=y[start], x1=x[end], y1=y[end], dx=dx, dy=dy)

        successor, has_pred = {}, set()
        for label in sorted(info, key=lambda l: (info[l]['end'], l)):
            d = info[label]
            x_proj, y_proj = d['x1'] + d['dx'], d['y1'] + d['dy']
            best, best_dist = None, None
            for other, o in info.items():
                if other == label or other in has_pred or o['start'] != d['end'] + 1:
                    continue
                dist = float(np.hypot(x_proj - o['x0'], y_proj - o['y0']))
                if dist <= mend_dist and (best_dist is None or dist < best_dist):
                    best, best_dist = other, dist
            if best is not None:
                successor[label] = best
                has_pred.add(best)

        lut = np.arange(int(tracks.max()) + 1, dtype=tracks.dtype)
        for label in sorted(info):
            if label in has_pred:
                continue
            nxt = successor.get(label)
            while nxt is not None:
                lut[nxt] = label
                nxt = successor.get(nxt)
        new_tracks = lut[np.clip(tracks, 0, None)]
        return new_tracks


class StreamingTracker:
    """
    Frame-by-frame object tracker with motion-advected overlap matching, for long runs of large grids
    (e.g. CONUS MRMS at 10 min): memory is one frame plus the pixels of the active tracks, never (NT, NY, NX).

    Per frame (``step``):

    1. Each active track's last pixels are advected by its velocity x (missed frames + 1) (pixels leaving the
       grid are dropped) and overlap counts with the new objects are taken in one pass.
    2. Pairs are ranked by overlap (pixels), greedy one-to-one: on a split or merge the larger overlap keeps the
       track id (recorded in ``events``).
    3. Unmatched objects take the free track whose projected centroid is nearest, within ``fallback_px``
       (greedy by distance).
    4. Unmatched objects start new tracks, with the median velocity of established tracks (>= ``min_frames``
       frames) within ``motion_radius_px`` (0 if fewer than 3 such tracks).
    5. Velocity of a matched track: v <- (1 - alpha) v + alpha d (d = centroid displacement / frames) when the
       area ratio is within ``area_ratio``; |v| capped at ``vmax_px``. Tracks not seen coast up to ``max_gap``
       frames, then end.

    Parameters
    ----------
    max_gap : int, default 2. Frames a track may be missing and still be continued.
    fallback_px : float, default 8. Centroid-distance fallback radius (grid points); 0 disables it.
    vmax_px : float, default 10. Speed cap (grid points per frame).
    motion_radius_px : float, default 67. Radius for the new-track motion (67 px ~ 150 km at 0.02 deg).
    min_frames : int, default 3. Frames before a track counts as established (its motion is used for new tracks).
    alpha : float, default 0.5. Velocity smoothing weight of the newest displacement.
    area_ratio : (float, float), default (0.5, 2.0). New/old area ratio range for a velocity update.
    x_scale : float, default 1.0. Column-distance factor for the motion radius (e.g. cos(lat) on a lat-lon grid).
    min_overlap : int, default 1. Minimum overlap (pixels) for an overlap match.

    Attributes
    ----------
    records : list of tuples (frame, track, label, area, cy, cx, vy, vx); ``table()`` gives a DataFrame.
    events : list of tuples (frame, kind, track, other_track, label): kind "split" (label is a new track that
       overlapped ``other_track``) or "merge" (``track`` overlapped the object that kept ``other_track``).

    Example
    -------
    >>> tracker = StreamingTracker()
    >>> for labels in frames:            # 2-D int label grids, e.g. EnhancedWatershed output
    ...     track_ids = tracker.step(labels)
    >>> df = tracker.table()
    """

    def __init__(self, max_gap=2, fallback_px=8.0, vmax_px=10.0, motion_radius_px=67.0, min_frames=3,
                 alpha=0.5, area_ratio=(0.5, 2.0), x_scale=1.0, min_overlap=1):
        self.max_gap = int(max_gap)
        self.fallback_px = float(fallback_px)
        self.vmax_px = float(vmax_px)
        self.motion_radius_px = float(motion_radius_px)
        self.min_frames = int(min_frames)
        self.alpha = float(alpha)
        self.area_ratio = area_ratio
        self.x_scale = float(x_scale)
        self.min_overlap = int(min_overlap)
        self.reset()

    def reset(self):
        self.frame = -1
        self.shape = None
        self.next_id = 1
        self.pix = {}     # track -> last flat pixel indices (int64)
        self.vel = {}     # track -> (vy, vx) px / frame
        self.cen = {}     # track -> (cy, cx) last seen
        self.area = {}
        self.miss = {}    # track -> frames missed since last seen
        self.nfr = {}     # track -> frames seen
        self.records = []
        self.events = []

    @property
    def active_tracks(self):
        return sorted(self.pix)

    def table(self):
        """Per-object records as a DataFrame (frame, track, label, area, cy, cx, vy, vx)."""
        return pd.DataFrame(self.records, columns=['frame', 'track', 'label', 'area', 'cy', 'cx', 'vy', 'vx'])

    def track(self, frames):
        """Generator: yields the track-id grid for each label grid of ``frames`` (any iterable)."""
        for labels in frames:
            yield self.step(labels)

    def step(self, labels, return_grid=True):
        """
        Track one frame.

        Parameters
        ----------
        labels : 2-D int array, object labels (0 = background); labels need not be consecutive.
        return_grid : bool, if True return the track-id grid (int32, same shape), else the label -> track lookup
            table (``lut[labels]`` is the grid).
        """
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError('labels must be 2-D')
        if self.shape is None:
            self.shape = labels.shape
        elif labels.shape != self.shape:
            raise ValueError(f'frame shape {labels.shape} != {self.shape}')
        self.frame += 1
        f = self.frame
        ny, nx = self.shape
        flat = labels.ravel()
        idx = np.flatnonzero(flat > 0)
        lab_raw = flat[idx].astype(np.int64)
        # compact labels 1..n (consecutive) for the bookkeeping; keep the caller's label values for the records
        ulab, lab = np.unique(lab_raw, return_inverse=True)
        lab = lab.astype(np.int64) + 1
        nobj = ulab.size
        compact = np.zeros(ny * nx, dtype=np.int32)  # only needed for advected lookups
        compact[idx] = lab
        y, x = np.divmod(idx, nx)
        ar = np.bincount(lab, minlength=nobj + 1)[1:].astype(float)
        cy = np.bincount(lab, y, minlength=nobj + 1)[1:] / np.maximum(ar, 1)
        cx = np.bincount(lab, x, minlength=nobj + 1)[1:] / np.maximum(ar, 1)
        assign = np.zeros(nobj + 1, dtype=np.int64)
        used = set()
        ov_pairs = []  # (track, compact label) overlaps, largest first
        new_tracks = set()
        trks = np.array(sorted(self.pix), dtype=np.int64)
        vy = vx = None
        if len(trks) and nobj:
            g = np.array([self.miss[t] + 1 for t in trks], float)
            vy = np.array([self.vel[t][0] for t in trks]) * g
            vx = np.array([self.vel[t][1] for t in trks]) * g
            sizes = [len(self.pix[t]) for t in trks]
            P = np.concatenate([self.pix[t] for t in trks])
            K = np.repeat(np.arange(len(trks)), sizes)
            py, px = np.divmod(P, nx)
            qy = np.rint(py + vy[K]).astype(np.int64)
            qx = np.rint(px + vx[K]).astype(np.int64)
            inside = (qy >= 0) & (qy < ny) & (qx >= 0) & (qx < nx)
            hit = np.zeros(P.size, dtype=np.int64)
            hit[inside] = compact[qy[inside] * nx + qx[inside]]
            s = hit > 0
            if s.any():
                key = K[s] * (nobj + 1) + hit[s]
                u, c = np.unique(key, return_counts=True)
                keep = c >= self.min_overlap
                u, c = u[keep], c[keep]
                order = np.argsort(-c, kind='stable')
                for kk in order:
                    k, l = divmod(int(u[kk]), nobj + 1)
                    t = int(trks[k])
                    ov_pairs.append((t, l))
                    if assign[l] == 0 and t not in used:
                        assign[l] = t
                        used.add(t)
            free = [i for i, t in enumerate(trks) if int(t) not in used]
            un = np.flatnonzero(assign[1:] == 0) + 1
            if self.fallback_px > 0 and free and len(un):
                fy = np.array([self.cen[int(trks[i])][0] for i in free]) + vy[free]
                fx = np.array([self.cen[int(trks[i])][1] for i in free]) + vx[free]
                dd = np.hypot(fy[None, :] - cy[un - 1][:, None], fx[None, :] - cx[un - 1][:, None])
                oi, oj = np.nonzero(dd <= self.fallback_px)
                for q in np.argsort(dd[oi, oj], kind='stable'):
                    l, t = int(un[oi[q]]), int(trks[free[oj[q]]])
                    if assign[l] == 0 and t not in used:
                        assign[l] = t
                        used.add(t)
        # motion for new tracks: median velocity of established tracks matched in this frame, within the radius
        est = [t for t in used if self.nfr.get(t, 0) >= self.min_frames]
        ey = np.array([self.cen[t][0] for t in est]) if est else np.zeros(0)
        ex = np.array([self.cen[t][1] for t in est]) if est else np.zeros(0)
        evy = np.array([self.vel[t][0] for t in est]) if est else np.zeros(0)
        evx = np.array([self.vel[t][1] for t in est]) if est else np.zeros(0)
        upd = {}
        lo, hi = self.area_ratio
        for l in range(1, nobj + 1):
            t = int(assign[l])
            c = (float(cy[l - 1]), float(cx[l - 1]))
            if t:
                pc, pa, gl = self.cen[t], self.area[t], self.miss[t] + 1
                v = self.vel[t]
                if lo <= ar[l - 1] / pa <= hi:
                    d = ((c[0] - pc[0]) / gl, (c[1] - pc[1]) / gl)
                    v = ((1 - self.alpha) * v[0] + self.alpha * d[0], (1 - self.alpha) * v[1] + self.alpha * d[1])
                    sp = float(np.hypot(*v))
                    if sp > self.vmax_px:
                        v = (v[0] * self.vmax_px / sp, v[1] * self.vmax_px / sp)
                self.nfr[t] += 1
            else:
                t = self.next_id
                self.next_id += 1
                new_tracks.add(t)
                assign[l] = t
                self.nfr[t] = 1
                v = (0.0, 0.0)
                if len(ey):
                    dd = np.hypot(ey - c[0], (ex - c[1]) * self.x_scale)
                    near = dd < self.motion_radius_px
                    if near.sum() >= 3:
                        v = (float(np.median(evy[near])), float(np.median(evx[near])))
            upd[t] = (v, c, ar[l - 1])
            self.records.append((f, t, int(ulab[l - 1]), int(ar[l - 1]), c[0], c[1], float(v[0]), float(v[1])))
        merged, split = set(), set()
        for t, l in ov_pairs:
            tl = int(assign[l])
            if tl == t:
                continue
            if t not in upd and t not in merged:
                merged.add(t)
                self.events.append((f, 'merge', t, tl, int(ulab[l - 1])))
            elif t in upd and tl in new_tracks and l not in split:
                split.add(l)
                self.events.append((f, 'split', tl, t, int(ulab[l - 1])))
        # coast or end the tracks not seen in this frame
        for t in list(self.pix):
            if t in upd:
                continue
            self.miss[t] += 1
            if self.miss[t] > self.max_gap:
                for dct in (self.pix, self.vel, self.cen, self.area, self.miss):
                    dct.pop(t, None)
        trk = assign[lab] if nobj else np.zeros(0, np.int64)
        if nobj:
            o = np.argsort(trk, kind='stable')
            ts, ps = trk[o], idx[o]
            st = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
            en = np.r_[st[1:], len(ts)]
            for a0, a1 in zip(st, en):
                t = int(ts[a0])
                v, c, a = upd[t]
                self.pix[t], self.vel[t], self.cen[t], self.area[t], self.miss[t] = ps[a0:a1].copy(), v, c, a, 0
        lut_compact = assign.astype(np.int32)
        if not return_grid:
            lut = np.zeros(int(ulab.max()) + 1 if nobj else 1, dtype=np.int32)
            lut[ulab] = lut_compact[1:]
            return lut
        out = np.zeros(ny * nx, dtype=np.int32)
        out[idx] = lut_compact[lab]
        return out.reshape(ny, nx)
