"""Regression tests for the review's bug list (TD-1074)."""
import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree

import monte_python
from monte_python.object_identification import IterativeWatershed, MCIT_Identifier, modal_filter
from monte_python.object_quality_control import _relabel, merge_labels


def test_modal_filter_matches_skimage_uint16():
    from skimage.filters.rank import modal
    from skimage.morphology import disk
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 600, size=(40, 50)).astype(np.uint16)
    lab[rng.random(lab.shape) < 0.5] = 7
    np.testing.assert_array_equal(modal_filter(lab, 2), modal(lab, disk(2)))


def test_iterative_watershed_keeps_labels_above_255():
    """_fix_bad_pixels used img_as_ubyte, which wrapped labels > 255."""
    lab = np.zeros((100, 100), dtype=np.int64)
    k = 0
    for i in range(0, 100, 5):
        for j in range(0, 100, 5):
            k += 1
            lab[i:i + 4, j:j + 4] = k
    assert k == 400
    out = IterativeWatershed(params=[])._fix_bad_pixels(lab)
    assert set(np.unique(out)) == set(np.unique(lab))


def test_mcit_through_label():
    """label(method='mcit') used MCIT_Identifier(**params) -> TypeError; `== np.nan` never fired."""
    yy, xx = np.mgrid[:60, :60]
    f = 10 * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / 30) + 8 * np.exp(-((yy - 40) ** 2 + (xx - 42) ** 2) / 30)
    labels = monte_python.label(f, method="mcit", params={"min_value": 1.0, "valley_depth": 0.5},
                                return_object_properties=False)
    assert labels.max() >= 1
    m = MCIT_Identifier()
    assert np.isnan(m.min_value)
    m.label(f)
    assert np.isfinite(m.min_value)
    assert MCIT_Identifier(min_value=1.0, valley_depth=0.5).min_value == 1.0


def _legacy_merge(labels, thresh):
    """The pre-1.2 _merge loop (KD-tree per label, sequential relabel), with scipy's cKDTree."""
    q = labels.copy()
    original = np.unique(q)[1:]
    remaining = np.unique(q)[1:]
    for label in original:
        if label in remaining:
            y, x = np.where(q == label)
            tree = cKDTree(np.c_[x, y])
            for other in original:
                if other != label and other in remaining:
                    oy, ox = np.where(q == other)
                    d, _ = tree.query(np.c_[ox, oy])
                    if round(np.amin(d), 2) <= thresh:
                        q[q == other] = label
                        remaining = np.unique(q)[1:]
    remaining = np.unique(q)[1:]
    for i, label in enumerate(remaining):
        q[q == label] = i + 1
    return q


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("thresh", [1.0, 2.5, 4.0, 7.0])
def test_merge_matches_legacy(seed, thresh):
    rng = np.random.default_rng(seed)
    mask = ndimage.gaussian_filter(rng.standard_normal((60, 70)), 1.5) > 0.35
    labels, n = ndimage.label(mask)
    assert n > 5
    np.testing.assert_array_equal(merge_labels(labels, thresh), _legacy_merge(labels, thresh))


def test_qc_int32_labels():
    """QC wrote int16 labels (overflow past 32,767 objects)."""
    lab = np.arange(1, 40001, dtype=np.int64).reshape(200, 200)
    out = _relabel(lab, np.arange(1, 40001))
    assert out.dtype == np.int32 and out.max() == 40000
    np.testing.assert_array_equal(out, lab)


def test_qc_match_to_lsr(fake_storms):
    """_remove_objects_unmatched_to_lsrs raised NameError (spatial / kdtree undefined)."""
    labels, props = monte_python.label(fake_storms, method="single_threshold", params={"bdry_thresh": 20})
    r = props[0]
    pt = r.coords[0]
    qc = monte_python.QualityControler()
    new, new_props = qc.quality_control(fake_storms, labels, props,
                                        [("match_to_lsr", {"lsr_points": [pt], "dist_to_lsr": 1.0})])
    assert len(new_props) == 1 and new.max() == 1
    assert new[pt[0], pt[1]] == 1


def test_qc_chain_unchanged(fake_storms):
    labels, props = monte_python.label(fake_storms, method="single_threshold", params={"bdry_thresh": 20})
    qc = monte_python.QualityControler()
    new, props2 = qc.quality_control(fake_storms, labels, props,
                                     [("min_area", 150), ("merge_thresh", 3), ("max_thresh", (40, 90))])
    assert new.dtype == np.int32
    assert len(props2) == new.max()


def test_mcit_all_valid_keeps_label_one():
    """MCIT used np.unique(labels)[1:] as the ids: with no pixel below min_value it dropped label 1 (KeyError)."""
    yy, xx = np.mgrid[:40, :40]
    f = 5 + 10 * np.exp(-((yy - 20) ** 2 + (xx - 20) ** 2) / 30)
    labels = MCIT_Identifier(min_value=0.0, valley_depth=0.5).label(f)
    assert labels.min() >= 1 and labels.dtype == np.int32


def test_enhanced_watershed_rejects_bad_increment():
    from monte_python.EnhancedWatershedSegmenter import EnhancedWatershed
    import pytest
    with pytest.raises(ValueError):
        EnhancedWatershed(min_thresh=40, max_thresh=65, area_threshold=10, dist_btw_objects=5, data_increment=0)
