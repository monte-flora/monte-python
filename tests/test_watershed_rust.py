"""EnhancedWatershed: Rust core == bug-fixed Python reference == the original code away from edges / NaN."""
import importlib.util
import os
import warnings

import numpy as np
import pytest

from conftest import smooth_field
from monte_python.EnhancedWatershedSegmenter import HAS_RUST, EnhancedWatershed

needs_rust = pytest.mark.skipif(not HAS_RUST, reason="monte_python._rust not built")

PARAMS = [
    dict(min_thresh=40, max_thresh=65, data_increment=1, area_threshold=100, dist_btw_objects=15),
    dict(min_thresh=35, max_thresh=70, data_increment=5, area_threshold=40, dist_btw_objects=50),
    dict(min_thresh=0, max_thresh=60, data_increment=2, area_threshold=60, dist_btw_objects=10, min_size=3),
    dict(min_thresh=25, max_thresh=80, data_increment=20, area_threshold=150, dist_btw_objects=50),
    dict(min_thresh=30.5, max_thresh=62.0, data_increment=0.5, area_threshold=30, dist_btw_objects=8),
]


def _legacy():
    path = os.path.join(os.path.dirname(__file__), "legacy", "ews_cc70583.py")
    spec = importlib.util.spec_from_file_location("ews_cc70583", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EnhancedWatershed


def ew(p, use_rust):
    return EnhancedWatershed(use_rust=use_rust, **p)


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_matches_original_away_from_edges(p, seed):
    """Fixes only change edge / NaN behaviour: with no candidate pixels near the edges, labels are unchanged."""
    if "min_size" in p:
        pytest.skip("min_size is new")
    f = smooth_field((90, 120), seed, border=16)
    old = _legacy()(**p).label(f)
    new = ew(p, False).label(f)
    np.testing.assert_array_equal(new, old)
    full_old = _legacy()(**p).label(f, only_objects=False)
    np.testing.assert_array_equal(ew(p, False).label(f, only_objects=False), full_old)


@needs_rust
@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("seed", range(6))
def test_rust_matches_reference(p, seed):
    f = smooth_field((70, 110), seed, sigma=3.0 + seed % 3)
    ref = ew(p, False).label(f, only_objects=False)
    rs = ew(p, True).label(f, only_objects=False)
    assert rs.dtype == np.int32
    np.testing.assert_array_equal(rs, ref)
    assert (ref > 0).any()


@needs_rust
@pytest.mark.parametrize("seed", range(3))
def test_rust_matches_reference_nan_and_edges(seed):
    p = PARAMS[0]
    f = smooth_field((80, 100), seed, sigma=3.0)
    f[:3, :] = 60.0          # strong band on the top edge
    f[:, -2:] = 55.0         # and the right edge
    f[30:40, 30:45] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = ew(p, False).label(f, only_objects=False)
        rs = ew(p, True).label(f, only_objects=False)
    np.testing.assert_array_equal(rs, ref)
    assert (ref[30:40, 30:45] <= 0).all()


@needs_rust
def test_rust_float32_and_masked(fake_storms):
    p = PARAMS[3]
    ref = ew(p, False).label(fake_storms)
    np.testing.assert_array_equal(ew(p, True).label(fake_storms), ref)
    f = smooth_field((60, 60), 5)
    np.testing.assert_array_equal(ew(PARAMS[0], True).label(f.astype(np.float32)),
                                  ew(PARAMS[0], False).label(f.astype(np.float32)))


@needs_rust
def test_rust_int_input():
    f = np.rint(smooth_field((60, 80), 7)).astype(np.int16)
    np.testing.assert_array_equal(ew(PARAMS[0], True).label(f), ew(PARAMS[0], False).label(f))


def test_edge_object_found():
    """Bug fix: a band on row 0 / column 0 is an object (edge pixels used to lose their neighbours)."""
    f = np.zeros((40, 40))
    f[0, :] = 60.0
    f[:, 0] = 60.0
    for use_rust in ([False, True] if HAS_RUST else [False]):
        lab = EnhancedWatershed(40, 65, 10, 15, min_size=2, use_rust=use_rust).label(f)
        assert (lab[0, :] > 0).all() and (lab[:, 0] > 0).all()


def test_centre_near_edge():
    """A lone peak 2 px from the top-left corner becomes a centre (its window used to wrap to an empty slice)."""
    yy, xx = np.mgrid[:50, :50]
    f = 60.0 * np.exp(-((yy - 2) ** 2 + (xx - 2) ** 2) / 20.0)
    lab = EnhancedWatershed(30, 65, 10, 15, use_rust=False).label(f)
    assert lab[2, 2] > 0


def test_nan_no_warning():
    f = np.full((30, 30), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lab = EnhancedWatershed(40, 65, 10, 15, use_rust=False).label(f)
    assert (lab == 0).all()


def test_min_size_param():
    f = np.zeros((30, 30))
    f[10:12, 10:14] = 60.0   # 8 px
    assert (EnhancedWatershed(40, 65, 100, 15, min_size=6, use_rust=False).label(f) > 0).sum() == 8
    assert (EnhancedWatershed(40, 65, 100, 15, min_size=8, use_rust=False).label(f) > 0).sum() == 0


def test_use_rust_true_without_extension(monkeypatch):
    import monte_python.EnhancedWatershedSegmenter as m
    monkeypatch.setattr(m, "HAS_RUST", False)
    with pytest.raises(ImportError):
        m.EnhancedWatershed(40, 65, 10, 15, use_rust=True)


def test_label_api_watershed(fake_storms):
    import monte_python
    labels, props = monte_python.label(fake_storms, method="watershed", return_object_properties=True,
                                       params={"min_thresh": 25, "max_thresh": 80, "data_increment": 20,
                                               "area_threshold": 150, "dist_btw_objects": 50})
    np.testing.assert_allclose([r.area for r in props], [149, 145, 151, 147, 71], atol=5)
