"""CONUS-size benchmark (MRMS 0.02 deg, 1750 x 3500): EnhancedWatershed Rust vs Python, split_objects, and a
streaming-tracker day (144 frames at 10 min) with peak RSS.

    python bench/bench_conus.py ew  [ncells]          # one frame: Rust / Python timing + label equality
    python bench/bench_conus.py day [ncells] [nfr]    # split_objects + StreamingTracker over nfr frames
    python bench/bench_conus.py trk ncells nfr WHICH  # one tracker over a day of split_objects frames; WHICH:
        ot-rust | ot-python (ObjectTracker streaming, mending on) | streaming (StreamingTracker) | none |
        td1071 (twc-severe-wx TD-1071 track.py Tracker; path in $TD1071_TRACK)
"""
import resource
import sys
import time

import numpy as np
import pandas as pd

import monte_python
from monte_python.EnhancedWatershedSegmenter import EnhancedWatershed
from monte_python.object_split import split_objects

NY, NX = 1750, 3500
EW = dict(min_thresh=40, max_thresh=65, data_increment=1, area_threshold=100, dist_btw_objects=15)


class Scene:
    """Gaussian cells + a squall line over noise; `frame(k)` moves everything k * (1, 3) px (deterministic)."""

    def __init__(self, ncells, seed=1):
        r = np.random.default_rng(seed)
        self.noise = np.random.default_rng(0).normal(10, 6, (NY, NX)).astype(np.float32)
        self.cells = [(r.integers(200, 1500), r.integers(200, 3000), r.lognormal(1.2, 0.5), r.uniform(42, 65))
                      for _ in range(ncells)]

    def frame(self, k=0):
        f = np.full((NY, NX), np.nan, np.float32)
        f[100:1650, 100:3400] = self.noise[100:1650, 100:3400]
        dy, dx = k, 3 * k
        for cy, cx, s, a in self.cells:
            cy, cx = cy + dy, cx + dx
            y0, y1, x0, x1 = max(cy - 40, 0), min(cy + 40, NY), max(cx - 40, 0), min(cx + 40, NX)
            if y0 >= y1 or x0 >= x1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            g = a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
            f[y0:y1, x0:x1] = np.fmax(f[y0:y1, x0:x1], g)
        ly = np.arange(600, 900) + dy
        lx = 1800 + dx + (0.3 * (ly - 600 - dy)).astype(int)
        for y, x in zip(ly, lx):
            if 0 <= y < NY and 40 <= x < NX - 40:
                f[y, x - 8:x + 8] = np.fmax(f[y, x - 8:x + 8], 45 + 10 * np.sin(y / 7.0))
                f[y, x + 8:x + 40] = np.fmax(f[y, x + 8:x + 40], 35)
        return f


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench_ew(ncells, python_ref=True):
    f = Scene(ncells).frame()
    print(f"px>=40 {int(np.nansum(f >= 40))}  px>=35 {int(np.nansum(f >= 35))}  rust={monte_python.HAS_RUST}")
    rs = EnhancedWatershed(use_rust=True, **EW)
    rs.label(f)
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        lr = rs.label(f, only_objects=False)
        ts.append(time.perf_counter() - t)
    print(f"rust EW s/frame median {np.median(ts):.3f} min {min(ts):.3f}  objects {int(lr.max())}")
    t = time.perf_counter()
    so = split_objects(f, threshold=40, split_px=300, **{k: v for k, v in EW.items() if k != 'min_thresh'})
    print(f"split_objects s/frame {time.perf_counter() - t:.3f}  objects {int(so.max())}")
    if python_ref:
        t = time.perf_counter()
        lp = EnhancedWatershed(use_rust=False, **EW).label(f, only_objects=False)
        print(f"python EW s/frame {time.perf_counter() - t:.2f}  equal={np.array_equal(lr, lp)}")
    print(f"maxrss MB {rss_mb():.0f}")


def bench_day(ncells, nfr):
    sc = Scene(ncells)
    st = monte_python.StreamingTracker()
    t_split = t_trk = 0.0
    base = rss_mb()
    for k in range(nfr):
        f = sc.frame(k % 48)  # loop the motion so cells stay in the domain
        t = time.perf_counter()
        lab = split_objects(f, threshold=40, split_px=300, **{k2: v for k2, v in EW.items() if k2 != 'min_thresh'})
        t_split += time.perf_counter() - t
        t = time.perf_counter()
        st.step(lab)
        t_trk += time.perf_counter() - t
    df = st.table()
    print(f"frames {nfr}  split s/frame {t_split / nfr:.3f}  track s/frame {t_trk / nfr:.3f}  "
          f"records {len(df)}  tracks {df['track'].nunique()}  events {len(st.events)}")
    print(f"maxrss MB {rss_mb():.0f} (after first frame setup {base:.0f})")


def bench_trk(ncells, nfr, which):
    sc = Scene(ncells)
    if which == "ot-rust":
        tr = monte_python.ObjectTracker(backend="rust")
    elif which == "ot-python":
        tr = monte_python.ObjectTracker(backend="python")
    elif which == "streaming":
        tr = monte_python.StreamingTracker()
    elif which == "td1071":
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location("td1071_track", os.environ["TD1071_TRACK"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tr = mod.Tracker(V=1)
    else:
        tr = None
    ts = []
    base = None
    for k in range(nfr):
        lab = split_objects(sc.frame(k % 48), threshold=40, split_px=300,
                            **{k2: v for k2, v in EW.items() if k2 != 'min_thresh'})
        if base is None:
            base = rss_mb()
        if tr is not None:
            t = time.perf_counter()
            if which == "td1071":
                idx = np.flatnonzero(lab > 0)
                tr.step(k, idx, lab.ravel()[idx].astype(np.int64), lab)
            else:
                tr.step(lab)
            ts.append(time.perf_counter() - t)
    ts = np.array(ts) if ts else np.zeros(1)
    if which == "td1071":
        tab = pd.DataFrame(tr.log, columns=["frame", "track", "area", "cy", "cx"])
    else:
        tab = tr.table() if tr is not None else pd.DataFrame({"track": []})
    n, ntrk = len(tab), tab["track"].nunique()
    print(f"{which}: frames {nfr} track s/frame p50 {np.median(ts):.3f} p90 {np.percentile(ts, 90):.3f} "
          f"p99 {np.percentile(ts, 99):.3f} max {ts.max():.3f} total {ts.sum():.1f} s  records {n} tracks {ntrk}  "
          f"maxrss MB {rss_mb():.0f} (after frame 0 {base:.0f})")


if __name__ == "__main__":
    mode = sys.argv[1]
    nc = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    if mode == "ew":
        bench_ew(nc, python_ref=len(sys.argv) <= 3 or sys.argv[3] != "norefs")
    elif mode == "trk":
        bench_trk(nc, int(sys.argv[3]), sys.argv[4])
    else:
        bench_day(nc, int(sys.argv[3]) if len(sys.argv) > 3 else 144)
