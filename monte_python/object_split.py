"""Split threshold objects into storm cells with the EnhancedWatershed (MRMS-style object ID).

``split_objects`` labels ``field >= threshold`` (8-connected); objects larger than ``split_px`` with two or more
EnhancedWatershed cores are divided by growing the cores back over the object (each pixel goes to the
geodesically nearest core, 8-connected, inside the object), so the object's footprint is kept exactly.
"""
from collections import deque

import numpy as np
from scipy import ndimage

from .EnhancedWatershedSegmenter import EnhancedWatershed, _rust

_EIGHT = np.ones((3, 3), dtype=bool)


def grow_labels(labels, mask, use_rust=None):
    """Grow labels (> 0) into ``mask`` pixels by 8-connected breadth-first search: every mask pixel connected to a
    labelled pixel takes the label of the geodesically nearest one (ties: row-major source order, FIFO).
    Pixels outside ``mask`` keep their label if labelled, else 0. Returns int32."""
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    mask = np.ascontiguousarray(mask, dtype=bool)
    if labels.shape != mask.shape or labels.ndim != 2:
        raise ValueError('labels and mask must be 2-D arrays of the same shape')
    if use_rust is None:
        use_rust = _rust is not None
    if use_rust:
        if _rust is None:
            raise ImportError('use_rust=True but monte_python._rust is not built')
        return _rust.grow_labels(labels, mask)
    ny, nx = labels.shape
    out = np.where(labels > 0, labels, 0).astype(np.int32).ravel()
    m = mask.ravel()
    queue = deque(np.flatnonzero(out > 0).tolist())
    while queue:
        i = queue.popleft()
        r, c = divmod(i, nx)
        lab = out[i]
        for rr in range(max(r - 1, 0), min(r + 2, ny)):
            base = rr * nx
            for cc in range(max(c - 1, 0), min(c + 2, nx)):
                j = base + cc
                if out[j] == 0 and m[j]:
                    out[j] = lab
                    queue.append(j)
    return out.reshape(ny, nx)


def split_objects(field, threshold=40.0, min_px=8, split_px=300, min_thresh=None, max_thresh=65.0,
                  data_increment=1.0, area_threshold=100, dist_btw_objects=15, min_size=6, use_rust=None,
                  return_cores=False):
    """
    Threshold objects split into cells by the EnhancedWatershed.

    Parameters
    ----------
    field : 2-D float array (e.g. reflectivity, dBZ); NaN is below threshold.
    threshold : object threshold (``field >= threshold``, 8-connected).
    min_px : objects with fewer pixels are dropped.
    split_px : objects with more pixels than this are split when they hold >= 2 watershed cores.
    min_thresh, max_thresh, data_increment, area_threshold, dist_btw_objects, min_size :
        EnhancedWatershed parameters (``min_thresh`` defaults to ``threshold``).
    use_rust : None (compiled core if built), True or False.
    return_cores : also return the watershed core labels (int32).

    Returns
    -------
    labels : int32 2-D, objects 1..n in row-major order of first pixel; every pixel of a kept threshold object is
        labelled (the split pieces tile the object).
    """
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError('field must be 2-D')
    if min_thresh is None:
        min_thresh = threshold
    with np.errstate(invalid='ignore'):
        mask = np.isfinite(field) & (field >= threshold)
    comp, n = ndimage.label(mask, structure=_EIGHT)
    out = np.zeros(field.shape, dtype=np.int32)
    cores = np.zeros(field.shape, dtype=np.int32)
    if n == 0:
        return (out, cores) if return_cores else out
    # sparse from here on: only threshold pixels and the bounding boxes of big objects are touched
    nz = np.flatnonzero(mask)
    cnz = comp.ravel()[nz]
    sizes = np.bincount(cnz, minlength=n + 1)
    keep = sizes >= min_px
    keep[0] = False
    big = keep & (sizes > split_px)
    lut = np.zeros(n + 1, dtype=np.int32)          # whole objects: one id each (renumbered below)
    lut[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    out.ravel()[nz] = lut[cnz]
    next_id = int(keep.sum()) + 1
    if big.any():
        ew = EnhancedWatershed(min_thresh=min_thresh, max_thresh=max_thresh, data_increment=data_increment,
                               area_threshold=area_threshold, dist_btw_objects=dist_btw_objects, min_size=min_size,
                               use_rust=use_rust)
        cores = ew.label(field)
        slices = ndimage.find_objects(comp)
        for k in np.flatnonzero(big):
            sl = slices[k - 1]
            obj = comp[sl] == k
            c = np.where(obj, cores[sl], 0)
            ids = np.unique(c[obj])
            ids = ids[ids > 0]
            if ids.size < 2:
                continue
            seeds = np.zeros(c.shape, dtype=np.int32)
            hit = c > 0
            seeds[hit] = np.searchsorted(ids, c[hit]).astype(np.int32) + 1
            grown = grow_labels(seeds, obj, use_rust=use_rust)
            # grown covers obj (every object pixel connects to a core); pieces get fresh ids
            out[sl][obj] = grown[obj] + (next_id - 1)
            next_id += ids.size
    # renumber 1..n in row-major order of first pixel
    vals = out.ravel()[nz]
    used, first = np.unique(vals, return_index=True)
    if used.size and used[0] == 0:
        used, first = used[1:], first[1:]
    order = np.argsort(first, kind='stable')
    remap = np.zeros(next_id, dtype=np.int32)
    remap[used[order]] = np.arange(1, used.size + 1, dtype=np.int32)
    out.ravel()[nz] = remap[vals]
    return (out, cores) if return_cores else out
