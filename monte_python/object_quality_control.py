import numpy as np
from collections import OrderedDict
from scipy.spatial import cKDTree
from skimage.measure import regionprops


def whereeq(arr, arrind, val, setval):
    """arr[arrind == val] = setval (in place; returns arr). Kept for callers of the old numba helper."""
    arr[arrind == val] = setval
    return arr


def _relabel(object_labels, kept_labels):
    """int32 grid with kept_labels (in order) renumbered 1..n and everything else 0, in one O(N) pass."""
    object_labels = np.asarray(object_labels)
    kept_labels = np.asarray(kept_labels, dtype=np.int64)
    top = int(object_labels.max()) if object_labels.size else 0
    lut = np.zeros(max(top, int(kept_labels.max()) if kept_labels.size else 0) + 1, dtype=np.int32)
    lut[kept_labels] = np.arange(1, len(kept_labels) + 1, dtype=np.int32)
    return lut[np.clip(object_labels, 0, None).astype(np.int64)]


def _boundary_pixels(labels):
    """(rows, cols, labels) of labelled pixels with a 4-neighbour of another value. The closest pixel pair of two
    disjoint objects is always a boundary pair, so merging distances need only these."""
    lab = np.asarray(labels)
    pad = np.pad(lab, 1, mode='constant', constant_values=0)
    c = pad[1:-1, 1:-1]
    edge = (c != pad[:-2, 1:-1]) | (c != pad[2:, 1:-1]) | (c != pad[1:-1, :-2]) | (c != pad[1:-1, 2:])
    rows, cols = np.nonzero(edge & (c > 0))
    return rows, cols, lab[rows, cols]


def merge_labels(object_labels, merge_thresh):
    """Merge objects closer than merge_thresh (grid points; distances rounded to 2 decimals, as before).

    Same result as the old sequential loop: labels are visited in increasing order; a still-existing label L
    absorbs every other still-existing object (with what it absorbed earlier) that has a member within
    merge_thresh of L's own pixels. Labels are then renumbered 1..n in order of the surviving label values.
    """
    labels = np.asarray(object_labels)
    originals = [int(v) for v in np.unique(labels) if v > 0]
    if not originals:
        return np.zeros(labels.shape, dtype=np.int32)
    rows, cols, labs = _boundary_pixels(labels)
    near = {k: set() for k in originals}
    if len(rows) > 1:
        tree = cKDTree(np.c_[rows, cols])
        pairs = tree.query_pairs(r=float(merge_thresh) + 0.01, output_type='ndarray')
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            diff = labs[a] != labs[b]
            a, b = a[diff], b[diff]
            d = np.hypot(rows[a] - rows[b], cols[a] - cols[b])
            ok = np.round(d, 2) <= merge_thresh
            for la, lb in zip(labs[a][ok].tolist(), labs[b][ok].tolist()):
                near[la].add(lb)
                near[lb].add(la)
    owner = {k: k for k in originals}          # original label -> current label value
    members = {k: [k] for k in originals}      # current label value -> original labels
    for L in originals:
        if owner[L] != L:                      # L was absorbed earlier
            continue
        for O in sorted({owner[m] for m in near[L]} - {L}):
            for m in members[O]:
                owner[m] = L
            members[L].extend(members.pop(O))
    survivors = sorted(members)
    lut = np.zeros(max(originals) + 1, dtype=np.int32)
    rank = {rep: i + 1 for i, rep in enumerate(survivors)}
    for k in originals:
        lut[k] = rank[owner[k]]
    return lut[np.clip(labels, 0, None).astype(np.int64)]


class QualityControler:
    '''
    QualityControl implements multiple quality control measures to post-process identified objects.
    These include: 
        - Minimum Area
        - Merging
        - Maximum Length
        - Duration
        - Maximum Value Within
        - Matched to Local Storm Reports
        - Split objects in half along minor axis based on maximum area
        - Trimming 
    '''
    qc_to_module_dict = {'min_area' : '_remove_small_objects', 
                         'merge_thresh' : '_merge', 
                         'max_length' : '_remove_long_objects',
                         'min_time' : '_remove_short_duration_objects',
                         'max_thresh' : '_remove_low_intensity_objects',
                         'match_to_lsr' : '_remove_objects_unmatched_to_lsrs',
                         'max_area_before_split' : '_split_region_in_half',
                         'trim' : '_trim',
                        }
    
    def __call__(self, input_data, object_labels, object_properties, qc_params):
        return self.quality_control(input_data, object_labels, object_properties, qc_params)
        
        
    def quality_control(self, input_data, object_labels, object_properties, qc_params): 
        """
        Applies quality control to identified objects. 
        Parameters:
        -------------------------
            object_labels, 2D numpy array of labeled objects
            
            object_properties, object properties calculated by skimage.measure.regionprops for object labels
            
            input_data, original 2D numpy array of data from which object_labels was generated from
            
            qc_params,  List of tuples or ordered dictionary of the quality control option to apply 
                          and the corresponding criteria. E.g., to apply a minimum area threshold 
                          qc_params = [('min_area', 10)]. 
                Potential quality control
                 - 'min_area', removing objects below the minimum area (in number of pixels)
                 - 'merge_thresh', merging objects together closer than a minimum distance (in gridpoints)
                 - 'max_length', removing objects with a major axis length greater the given criterion (in gridpoints)
                 - 'min_time', removing objects with duration less than the given criterion (in time steps) 
                 - 'max_thresh', removing objects if the Pth percentile value is less than the given value. 
                                 For max, P=100 and min, P=0. For this quality control, use a nested tuple
                                 E.g., qc_params = [('max_thresh', (value, P)]
                 - 'match_to_lsr', remove objects not matched to an local storm report
                 - 'max_area_before_split', split objects in half along minor axis based on the given maximum area
                For additional details, read the functions below. 
 
        Returns:
        -------------------------
            2-tuple of (object_labels, object_props) 
            object_labels, quality-controlled version of the original object_labels
            object_props, object properties of the quality-controlled objects
        """
        self.input_data = input_data
        self.object_labels = object_labels
        self.object_properties = object_properties
        
        if isinstance(qc_params, list):
            qc_params = OrderedDict(qc_params)
        self.qc_params = qc_params
        
        for qc_method in self.qc_params.keys():
            getattr(self, self.qc_to_module_dict[qc_method])()
        return (self.object_labels, self.object_properties) 

    def _remove_small_objects(self):
        ''' 
        Removes objects with area less than min_area. Area measured by the number of
        grid cells.
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : min_area, int, minimum area (in number of grid points) of an object
        '''
        kept = [region.label for region in self.object_properties if region.area >= self.qc_params['min_area']]
        qc_object_labels = _relabel(self.object_labels, kept)
        qc_object_properties = regionprops( qc_object_labels, self.input_data,  ) 
        
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_long_objects(self):
        ''' 
        Removes objects with a major axis length greater than max_length 
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : max_length, int, maximum object length (in grid point distances) 
        '''
        # If the object is shorter than the length, keep it.
        kept = [region.label for region in self.object_properties
                if round(region.major_axis_length, 2) <= self.qc_params['max_length']]
        qc_object_labels = _relabel(self.object_labels, kept)
        qc_object_properties = regionprops( qc_object_labels, self.input_data,  )

        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_low_intensity_objects(self):
        ''' 
        Removes objects where the 75th percentile intensity is below the max_thresh
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : max_thresh, float, intensity threshold for the 75th percentile
        '''
        val, P = self.qc_params['max_thresh']
        data = np.asarray(self.input_data)
        kept = [region.label for region in self.object_properties
                if round(np.percentile(data[region.coords[:, 0], region.coords[:, 1]], P), 10) >= round(val, 10)]
        qc_object_labels = _relabel(self.object_labels, kept)
        qc_object_properties = regionprops( qc_object_labels, self.input_data, )

        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties 

    def _remove_short_duration_objects(self): 
        ''' 
        When identify objects over a time duration, removes objects existing for a duration less than min_time
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : min_time, int, minimum duration (in terms of number of time steps) 
            : time_argmax_idxs, When applying np.argmax over time dimensions of an array
        '''
        time_argmax_idxs = np.asarray(self.qc_params['min_time'][1])
        kept = [region.label for region in self.object_properties
                if len(np.unique(time_argmax_idxs[region.coords[:, 0], region.coords[:, 1]]))
                >= self.qc_params['min_time'][0]]
        qc_object_labels = _relabel(self.object_labels, kept)
        qc_object_properties = regionprops( qc_object_labels, self.input_data, )        
        
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _trim(self):
        """Trim labelled regions. Useful for the ensemble storm tracks identified 
        with the iterative watershed method. If the maximum intensity of a region exceeds
        a threshold than trim the object based on a minimum intensity threshold
        
        : trim , 2-tuple of maximum intensity and minimum intensity thresholds
       
        """
        qc_object_labels = np.array(self.object_labels, dtype=np.int32)
        data = np.asarray(self.input_data)

        max_thresh = self.qc_params['trim'][0]
        bdry_thresh = self.qc_params['trim'][1]

        for region in self.object_properties:
            if region.intensity_max >= max_thresh:
                r, c = region.coords[:, 0], region.coords[:, 1]
                low = data[r, c] <= bdry_thresh
                qc_object_labels[r[low], c[low]] = 0
    
        qc_object_properties = regionprops(qc_object_labels, self.input_data)
    
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    
    def _split_region_in_half(self):
        """Split labelled regions along the minor axis slope if the 
        maximum area threshold is exceeded"""
        
        max_area = self.qc_params['max_area_before_split']
        qc_object_labels = np.zeros(self.object_labels.shape, dtype=np.int32)

        c = 1
        for region in self.object_properties:
            if region.area >= max_area:
                # Get the region's center
                y_cent, x_cent = np.mean(region.coords, axis=0).astype(int)
            
                # Define the slope and intercept of the splitting line
                slope = -region.orientation
                intercept = y_cent - (slope * x_cent)
    
                # Split the region along the line
                for j,i in region.coords:
                    if j < (slope * i) + intercept:
                        qc_object_labels[j,i] = c
                    else:
                        qc_object_labels[j,i] = c+1

                # Check that the area has been divided with no loss area!
                vals = qc_object_labels[region.coords[:, 0], region.coords[:, 1]]
                assert np.count_nonzero((vals == c) | (vals == c + 1)) == region.area
                c+=2
            else:
                qc_object_labels[region.coords[:, 0], region.coords[:, 1]] = c
                c+=1

        qc_object_properties = regionprops(qc_object_labels, self.input_data) 
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties
            
        
    def _merge(self):
        ''' 
        Merges near-by objects to limit discontinuous objects from counting as multiple objects.
        Applicable when using the single threshold object identification method. 
        Args,
            : input_data, 2D numpy array, original data from which the object labels were generated from.
                                          Used to recalculate the regionprops after qualtiy control is applied. 
            : object_labels_and_props, 2-tuple of 2D np array of object labels and the associated regionprops object 
            : merge_thresh, float, merging distance (objects closer than the merging distance 
                                   are combined together; in grid point distance)
        '''
        qc_object_labels = merge_labels(self.object_labels, self.qc_params['merge_thresh'])

        qc_object_properties = regionprops( qc_object_labels, self.input_data, )
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties

    def _remove_objects_unmatched_to_lsrs(self):
        '''
        Removes objects unmatched to Local Storm Reports (LSRs).
        '''
        lsr_points = np.atleast_2d(np.asarray(self.qc_params['match_to_lsr']['lsr_points'], dtype=float))
        kept = []
        for region in self.object_properties:
            if len(lsr_points) == 0:
                break
            dist_btw_region_and_lsr, _ = cKDTree(region.coords).query(lsr_points)
            if round(np.amin(dist_btw_region_and_lsr), 10) < self.qc_params['match_to_lsr']['dist_to_lsr']:
                kept.append(region.label)
        qc_object_labels = _relabel(self.object_labels, kept)

        qc_object_properties = regionprops( qc_object_labels, self.input_data, )
        self.object_labels = qc_object_labels
        self.object_properties = qc_object_properties
