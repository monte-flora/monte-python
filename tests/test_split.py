"""split_objects / grow_labels (TD-1071 blob splitting)."""
import numpy as np
import pytest
from scipy import ndimage

from conftest import smooth_field
from monte_python.EnhancedWatershedSegmenter import HAS_RUST
from monte_python.object_split import grow_labels, split_objects

needs_rust = pytest.mark.skipif(not HAS_RUST, reason="monte_python._rust not built")


def two_cells(shape=(80, 120)):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return 62 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 300.) + 60 * np.exp(-((yy - 40) ** 2 + (xx - 72) ** 2) / 300.)


@pytest.mark.parametrize("use_rust", [False, True] if HAS_RUST else [False])
def test_split_two_cells_tiles_the_object(use_rust):
    f = two_cells()
    lab = split_objects(f, threshold=40, split_px=300, area_threshold=50, dist_btw_objects=15, use_rust=use_rust)
    mask = f >= 40
    assert ndimage.label(mask, structure=np.ones((3, 3)))[1] == 1
    assert lab.max() == 2
    np.testing.assert_array_equal(lab > 0, mask)
    assert lab[40, 40] != lab[40, 72]


def test_small_objects_whole_and_dropped():
    f = np.zeros((40, 40))
    f[5:8, 5:8] = 50     # 9 px: kept whole (below split_px)
    f[20, 20] = 50       # 1 px: dropped (min_px)
    lab = split_objects(f, threshold=40, min_px=8, split_px=300)
    assert lab.max() == 1 and (lab > 0).sum() == 9


@needs_rust
@pytest.mark.parametrize("seed", range(4))
def test_split_rust_equals_python(seed):
    f = smooth_field((120, 160), seed, sigma=5)
    a = split_objects(f, threshold=40, split_px=100, area_threshold=30, use_rust=True)
    b = split_objects(f, threshold=40, split_px=100, area_threshold=30, use_rust=False)
    np.testing.assert_array_equal(a, b)


@needs_rust
@pytest.mark.parametrize("seed", range(4))
def test_grow_labels_rust_equals_python(seed):
    rng = np.random.default_rng(seed)
    mask = ndimage.gaussian_filter(rng.standard_normal((60, 80)), 2) > 0
    seeds = np.zeros(mask.shape, np.int32)
    pts = rng.integers(0, 60, 15), rng.integers(0, 80, 15)
    seeds[pts] = np.arange(1, 16)
    np.testing.assert_array_equal(grow_labels(seeds, mask, use_rust=True), grow_labels(seeds, mask, use_rust=False))


def test_core_field_smoothed():
    """Cores from a smoothed copy: the footprint is still field >= threshold, and split pieces still tile it."""
    f = two_cells()
    lab = split_objects(f, threshold=40, split_px=300, area_threshold=50, dist_btw_objects=15,
                        core_field=ndimage.gaussian_filter(f, 2))
    np.testing.assert_array_equal(lab > 0, f >= 40)
    assert lab.max() == 2
    with pytest.raises(ValueError):
        split_objects(f, core_field=f[:10])
