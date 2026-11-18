"""ObjectTracker streaming (Rust and numpy) == the batch reference; mending off + cap == the original (cc70583)."""
import numpy as np
import pytest

from monte_python import ObjectTracker
from monte_python.object_tracking import _rust

from legacy.tracker_cc70583 import ObjectTracker as LegacyTracker

BACKENDS = [False] + ([True] if _rust is not None else [])  # use_rust


def storms(seed, nt=12, shape=(60, 80), n=9):
    """Moving rectangles and blobs with splits, merges, births, deaths and frame-to-frame jitter; labels shuffled
    and non-consecutive so label order and track order differ."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    y = rng.uniform(0, ny, n); x = rng.uniform(0, nx, n)
    vy = rng.uniform(-2, 2, n); vx = rng.uniform(-3, 3, n)
    h = rng.integers(2, 9, n); w = rng.integers(2, 11, n)
    born = rng.integers(0, nt // 2, n); dies = born + rng.integers(2, nt, n)
    out = []
    for t in range(nt):
        f = np.zeros(shape, np.int32)
        perm = rng.permutation(np.arange(1, 4 * n + 1))
        for i in range(n):
            if not born[i] <= t < dies[i]:
                continue
            yy = int(y[i] + vy[i] * t + rng.normal(0, 0.7)); xx = int(x[i] + vx[i] * t + rng.normal(0, 0.7))
            hh = max(1, h[i] + int(rng.integers(-1, 2))); ww = max(1, w[i] + int(rng.integers(-1, 2)))
            f[max(yy, 0):max(yy + hh, 0), max(xx, 0):max(xx + ww, 0)] = perm[i]
            if (t + i) % 5 == 0 and ww > 3:  # a split: right part gets its own label
                f[max(yy, 0):max(yy + hh, 0), max(xx + ww // 2, 0):max(xx + ww, 0)] = perm[n + i]
        out.append(f)
    return np.stack(out)


CASES = [(seed, po, cap, mend, md) for seed in range(6) for po in (0.0, 0.15, 0.35) for cap in (False, True)
         for mend, md in ((False, 3), (True, 3), (True, 8))]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed,po,cap,mend,md", CASES)
def test_stream_equals_batch_reference(backend, seed, po, cap, mend, md):
    data = storms(seed)
    kw = dict(percent_overlap=po, cap_overlap=cap, mend_tracks=mend, mend_dist=md)
    ref = ObjectTracker(**kw)._track_batch(data)
    got = ObjectTracker(use_rust=backend, **kw).track(data)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seed,po", [(s, po) for s in range(4) for po in (0.0, 0.1, 0.3)])
def test_mend_off_cap_equals_original(backend, seed, po):
    data = storms(seed, nt=8, shape=(40, 50), n=6)
    legacy = LegacyTracker(percent_overlap=po).track(data.copy())
    got = ObjectTracker(use_rust=backend, percent_overlap=po, cap_overlap=True, mend_tracks=False).track(data)
    np.testing.assert_array_equal(got, legacy.astype(np.int64))


@pytest.mark.skipif(_rust is None, reason="Rust extension not built")
@pytest.mark.parametrize("seed", range(4))
def test_rust_equals_python_records_and_events(seed):
    data = storms(seed, nt=15)
    r, p = ObjectTracker(use_rust=True, mend_dist=6), ObjectTracker(use_rust=False, mend_dist=6)
    for f in data:
        np.testing.assert_array_equal(r.step(f), p.step(f))
    tr, tp = r.table(), p.table()
    assert tr[["frame", "track", "label", "area"]].equals(tp[["frame", "track", "label", "area"]])
    np.testing.assert_allclose(tr[["cy", "cx"]].values, tp[["cy", "cx"]].values, rtol=0, atol=1e-9)
    assert r.events().equals(p.events())


def test_merge_split_checks_are_noops():
    """match() is one-to-one, so check_for_mergers / check_for_splits never remove a pair (dropped in Rust)."""
    data = storms(3)
    t = ObjectTracker()
    for a, b in zip(data[:-1], data[1:]):
        la, lb = t.match(a, b)
        assert len(set(la)) == len(la) and len(set(lb)) == len(lb)


def test_streaming_matches_track_and_resets():
    data = storms(1)
    t = ObjectTracker()
    full = t.track(data)
    again = np.stack(list(t.track_iter(iter(list(data)))))
    np.testing.assert_array_equal(full, again)
    assert set(t.table()["track"]) == set(np.unique(full[full > 0]))


def test_mend_default_on_and_events():
    box = lambda x: np.pad(np.ones((3, 3), np.int32), ((10, 27), (x, 57 - x)))  # noqa: E731
    frames = [box(5 + 4 * k) for k in range(6)]  # moves 4 px / frame: never overlaps
    t = ObjectTracker(mend_dist=5)
    tracks = t.track(frames)
    assert len(np.unique(tracks)) == 2
    ev = t.events()
    assert list(ev["kind"]) == ["mend"] * 5 and set(ev["track"]) == {1}
    assert len(np.unique(ObjectTracker(mend_tracks=False).track(frames))) == 7


def test_empty_frames_and_shape_check():
    t = ObjectTracker()
    z = np.zeros((5, 6), np.int32)
    a = z.copy(); a[1:3, 1:3] = 4
    out = t.track([z, a, z, a])
    assert out[1].max() == 1 and out[3].max() == 2 and out[0].max() == 0
    with pytest.raises(ValueError):
        t.step(np.zeros((4, 4), np.int32))


def _both(frames, **kw):
    """(reference batch, each streaming backend) results; asserts they agree and returns the reference."""
    ref = ObjectTracker(**kw)._track_batch(np.stack(frames))
    for b in BACKENDS:
        np.testing.assert_array_equal(ObjectTracker(use_rust=b, **kw).track(frames), ref)
    return ref


def test_ties_lower_label_and_lower_track_win():
    z = np.zeros((10, 20), np.int32)
    old = z.copy(); old[2:6, 4:12] = 1
    new = z.copy(); new[2:6, 4:8] = 9; new[2:6, 8:12] = 2  # split into equal halves: label 2 (right) continues
    out = _both([old, new])
    assert out[1][3, 9] == 1 and out[1][3, 5] == 2
    a = z.copy(); a[2:6, 2:6] = 3; a[2:6, 10:14] = 1   # two tracks (ids by label: 1 right, 2 left)
    m = z.copy(); m[2:6, 4:12] = 5                      # merge with equal overlaps: the lower track id (1) wins
    out = _both([a, m])
    assert out[0][3, 11] == 1 and out[1][3, 6] == 1


@pytest.mark.parametrize("md,joined", [(4.0, True), (3.999, False)])
def test_mend_distance_boundary_inclusive(md, joined):
    z = np.zeros((10, 30), np.int32)
    a = z.copy(); a[3:6, 2:5] = 1      # one-frame track (no motion), centroid col 3
    b = z.copy(); b[3:6, 6:9] = 1      # next frame, no overlap, centroid col 7: distance 4
    out = _both([a, b], mend_dist=md)
    assert (out[1].max() == 1) == joined


def test_mend_uses_truncated_centroids():
    z = np.zeros((10, 40), np.int32)
    a = z.copy(); a[3:5, 1:5] = 1      # centroid col 2.5 -> 2
    d = z.copy(); d[3:5, 5:8] = 1      # no overlap; centroid col 6.0: distance 4 truncated, 3.5 untruncated
    assert _both([a, d], mend_dist=3.75)[1].max() == 2   # not mended (untruncated centroids would mend)
    assert _both([a, d], mend_dist=4)[1].max() == 1


@pytest.mark.skipif(_rust is None, reason="Rust extension not built")
@pytest.mark.parametrize("seed,gap,md", [(s, g, md) for s in range(4) for g in (2, 3) for md in (6, 15)])
def test_rust_equals_python_mend_gap(seed, gap, md):
    data = storms(seed, nt=15)
    data[4] = 0  # an empty frame: every track breaks for a frame
    data[9][:, ::2] = 0  # thinned frame: fewer overlaps, more mends
    r = ObjectTracker(use_rust=True, mend_dist=md, mend_gap=gap)
    p = ObjectTracker(use_rust=False, mend_dist=md, mend_gap=gap)
    for f in data:
        np.testing.assert_array_equal(r.step(f), p.step(f))
    assert r.events().equals(p.events())
    assert r.table()[["frame", "track", "label", "area"]].equals(p.table()[["frame", "track", "label", "area"]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_mend_gap_bridges_missing_frames(backend):
    box = lambda x: np.pad(np.ones((3, 6), np.int32), ((10, 27), (x, 54 - x)))  # noqa: E731
    z = np.zeros_like(box(0))
    # 6 px wide, moves 4 px / frame (overlaps: motion 4 px); frames 3 and 4 missing: resumes 3 frames later
    frames = [box(5 + 4 * k) if k not in (3, 4) else z for k in range(7)]
    ids = lambda **kw: set(np.unique(ObjectTracker(use_rust=backend, **kw).track(frames))) - {0}  # noqa: E731
    assert ids(mend_dist=1, mend_gap=1) == {1, 2}  # gap 1: the frame-2 track cannot reach frame 5
    assert ids(mend_dist=1, mend_gap=2) == {1, 2}
    assert ids(mend_dist=1, mend_gap=3) == {1}     # projected 3 x 4 px: exact
    with pytest.raises(ValueError):
        ObjectTracker(mend_gap=0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mend_gap_one_equals_default(backend):
    data = storms(5, nt=15)
    a = ObjectTracker(use_rust=backend, mend_dist=8).track(data)
    b = ObjectTracker(use_rust=backend, mend_dist=8, mend_gap=1).track(data)
    np.testing.assert_array_equal(a, b)
