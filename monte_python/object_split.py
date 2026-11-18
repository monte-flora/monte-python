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
    if min_thresh is None:
        min_thresh = threshold
    with np.errstate(invalid='ignore'):
        mask = np.isfinite(field) & (field >= threshold)
    comp, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        out = np.zeros(field.shape, dtype=np.int32)
        return (out, out.copy()) if return_cores else out
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    keep = sizes >= min_px
    keep[0] = False
    big = keep & (sizes > split_px)
    cores = np.zeros(field.shape, dtype=np.int32)
    if big.any():
        ew = EnhancedWatershed(min_thresh=min_thresh, max_thresh=max_thresh, data_increment=data_increment,
                               area_threshold=area_threshold, dist_btw_objects=dist_btw_objects, min_size=min_size,
                               use_rust=use_rust)
        cores = ew.label(field)
    # cores inside big objects, unique per (object, core) pair
    in_big = big[comp] & (cores > 0)
    key = comp[in_big].astype(np.int64) * (int(cores.max()) + 1) + cores[in_big]
    ukey, inv = np.unique(key, return_inverse=True)
    core_comp = ukey // (int(cores.max()) + 1)
    ncores = np.bincount(core_comp, minlength=n + 1)
    split = big & (ncores >= 2)
    seeds = np.zeros(field.shape, dtype=np.int32)
    sel = split[comp[in_big]]
    rows, cols = np.nonzero(in_big)
    seeds[rows[sel], cols[sel]] = inv.ravel()[sel].astype(np.int32) + 1
    grown = grow_labels(seeds, split[comp], use_rust=use_rust)
    # whole objects (kept, not split) get their own ids after the split pieces
    whole = keep & ~split
    offset = int(grown.max())
    lut = np.zeros(n + 1, dtype=np.int64)
    lut[whole] = offset + np.arange(1, int(whole.sum()) + 1)
    out = np.where(grown > 0, grown, lut[comp]).astype(np.int64)
    # renumber 1..n in row-major order of first pixel
    flat = out.ravel()
    nz = np.flatnonzero(flat)
    if nz.size:
        vals, first = np.unique(flat[nz], return_index=True)
        order = np.argsort(first, kind='stable')
        remap = np.zeros(int(vals.max()) + 1, dtype=np.int32)
        remap[vals[order]] = np.arange(1, len(vals) + 1, dtype=np.int32)
        out = remap[out]
    out = out.astype(np.int32)
    return (out, cores) if return_cores else out
