"""ObjectTracker fixes and StreamingTracker."""
import numpy as np

from monte_python import ObjectTracker, StreamingTracker
from monte_python.object_tracking import calc_dist


def box(shape, y, x, h, w, val=1):
    a = np.zeros(shape, dtype=np.int32)
    a[max(y, 0):y + h, max(x, 0):x + w] = val
    return a


def test_overlap_score_reaches_one():
    """percent_intersection capped at 0.5; Dice reaches 1 for identical objects."""
    a = box((20, 20), 5, 5, 4, 4)
    t = ObjectTracker(percent_overlap=0.6)
    pairs = t.find_possible_matches(a, a)
    assert pairs == {(1, 1): 1.0}
    tracks = t.track([a, a, a])
    assert len(np.unique(tracks)) == 2  # one track + background


def test_objecttracker_same_pairs_as_all_pairs():
    rng = np.random.default_rng(0)
    a = (rng.random((30, 30)) > 0.7).astype(int) * rng.integers(1, 6, (30, 30))
    b = (rng.random((30, 30)) > 0.7).astype(int) * rng.integers(1, 6, (30, 30))
    from skimage.measure import regionprops
    t = ObjectTracker()
    brute = {}
    for ra in regionprops(a):
        for rb in regionprops(b):
            s = t.percent_intersection(ra, rb)
            if s > 0:
                brute[(ra.label, rb.label)] = s
    fast = t.find_possible_matches(a, b)
    assert fast.keys() == brute.keys()
    for k in fast:
        assert abs(fast[k] - brute[k]) < 1e-12


def test_mending_joins_broken_track():
    """mend_broken_tracks was dead (x for y, NaN motion, hard-coded 3, squared distance)."""
    frames = [box((40, 60), 10, 5 + 4 * k, 3, 3) for k in range(6)]  # 3x3 box moving 4 px: never overlaps
    assert len(np.unique(ObjectTracker().track(frames))) == 7
    mended = ObjectTracker(mend_tracks=True, mend_dist=5).track(frames)
    assert len(np.unique(mended)) == 2
    assert len(np.unique(ObjectTracker(mend_tracks=True, mend_dist=3).track(frames))) == 7


def test_calc_dist_euclidean():
    assert calc_dist((0, 0), (3, 4)) == 5.0
    assert calc_dist((np.nan, 0), (3, 4)) == 0.0


def test_trackprops():
    frames = [box((30, 30), 5, 2 + 2 * k, 4, 4) for k in range(4)]
    t = ObjectTracker()
    t.track(frames)
    df = t.trackprops
    assert list(df["duration"]) == [4]
    assert abs(df["length"].iloc[0] - 6.0) < 1e-9


def test_streaming_moving_box_keeps_id():
    st = StreamingTracker()
    ids = [st.step(box((50, 80), 20, 5 + 3 * k, 8, 8, val=k + 1)) for k in range(10)]
    vals = {int(np.unique(g[g > 0])[0]) for g in ids}
    assert vals == {1}
    df = st.table()
    assert len(df) == 10 and (df["track"] == 1).all()
    assert abs(df["vx"].iloc[-1] - 3.0) < 0.3


def test_streaming_gap_and_fast_motion():
    """No overlap between frames (6 px/frame, 4 px box) and one missing frame: the centroid fallback starts the
    motion, then advection keeps the id through the gap."""
    st = StreamingTracker()
    out = []
    for k in range(8):
        if k == 5:
            out.append(st.step(np.zeros((40, 100), np.int32)))
            continue
        out.append(st.step(box((40, 100), 15, 2 + 6 * k, 4, 4)))
    tracks = {int(v) for g in out for v in np.unique(g) if v}
    assert tracks == {1}


def test_streaming_track_ends_after_max_gap():
    st = StreamingTracker(max_gap=1)
    st.step(box((30, 30), 5, 5, 4, 4))
    st.step(np.zeros((30, 30), np.int32))
    st.step(np.zeros((30, 30), np.int32))
    assert st.active_tracks == []
    g = st.step(box((30, 30), 5, 5, 4, 4))
    assert g.max() == 2


def test_streaming_split_larger_keeps_id():
    st = StreamingTracker()
    st.step(box((40, 40), 10, 5, 10, 20))
    two = box((40, 40), 10, 5, 10, 13) + box((40, 40), 10, 20, 10, 5, val=2)
    g = st.step(two)
    assert g[12, 6] == 1 and g[12, 21] == 2
    kinds = [e[1] for e in st.events]
    assert "split" in kinds


def test_streaming_merge_event():
    st = StreamingTracker()
    st.step(box((40, 40), 10, 5, 10, 8) + box((40, 40), 10, 20, 10, 4, val=2))
    g = st.step(box((40, 40), 10, 5, 10, 20))
    assert g[12, 6] == 1
    assert ("merge" in [e[1] for e in st.events])


def test_streaming_lut_and_noncontiguous_labels():
    st = StreamingTracker()
    lab = box((20, 20), 2, 2, 3, 3, val=7) + box((20, 20), 10, 10, 3, 3, val=42)
    lut = st.step(lab, return_grid=False)
    assert lut[7] > 0 and lut[42] > 0 and lut[7] != lut[42]
    g = st.step(lab)
    np.testing.assert_array_equal(g, lut[lab])
