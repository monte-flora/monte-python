# ObjectTracker in Rust: design (TD-1074, 2026-11-17; branch td-1074-objecttracker-rust, after PR #1)

Line cites `:NNN` are `monte_python/object_tracking.py` at b31be5d; `cc70583:NNN` the original. GFS-RTX = the
shared dev box the benchmarks run on (4 cores used).

## Problem

`ObjectTracker` (`monte_python/object_tracking.py:35`) is the user's overlap tracker: designed for 5-10 min steps,
with `mend_tracks` to lengthen tracks. The user wants it ported to Rust (PyO3, same API, same algorithm), mending ON
by default, the original 0.5-capped overlap score back as a flag, and no motion advection. It must run a CONUS MRMS
day (144 frames of 1750 x 3500 at 0.02 deg) in <= 1 GB RSS: today `track()` needs the whole `(NT, NY, NX)` array
in memory plus a copy (`get_unique_labels`, `relabel`): ~3.5 GB int32 per day for each.

## Reference behavior (equivalence contract; b31be5d, original = cc70583)

- `track()` (:88-137): unique labels per frame (`get_unique_labels` :273, ids in label order, cumulative over
  frames); per pair (t, t+1): `match` (:305) -> `check_for_mergers` (:187) -> `check_for_splits` (:231) -> relabel
  frame t+1 through a LUT; then `relabel` (:295, compact ids by sorted value); then optional `mend_broken_tracks`.
- `match` (:305-336): pairs from `find_possible_matches` (:348) sorted by score descending with Python's stable
  sort (ties keep insertion order = (label_a, label_b) ascending, from `np.unique` in `_pair_overlaps` :14); greedy
  one-to-one. Because matching is one-to-one, `labels_before` and `labels_after` have no repeats, so the merger and
  split checks never fire (no-ops; kept in Python, dropped in Rust, asserted by a test).
- Score (:348-369): Dice `2n / (|A| + |B|)` since PR #1; original (cc70583 `percent_intersection`)
  `n / (|A| + |B|)`, capped at 0.5. Same ranking (x2 is exact in floating point); differs only against
  `percent_overlap`. Kept if `score > percent_overlap`.
- Track ids after `relabel` = rank of the first object's unique id = order of track starts by (frame, input label).
- `mend_broken_tracks` (:404-463, bug-fixed in PR #1): per track start, end, truncated centroids (`int` of the
  regionprops centroid, row as "x", column as "y", :382-394) at start and end, mean motion (mean of centroid diffs;
  tracks have no gaps, so = (end - start centroid) / (n - 1), exact on integer centroids; 0 for one-frame tracks).
  Tracks sorted by (end, label) pick the nearest unclaimed track starting at end + 1 within `mend_dist` (Euclidean,
  ties: lowest label); successor chains relabel to the chain root; ids are not compacted afterwards. Original
  cc70583 mending had no effect (constructor flag ignored at cc70583:126, x centroid used for y at cc70583:419,
  NaN motion at cc70583:412, squared distance, and `mend_dist=3` hard-coded in the call).
- `mend_tracks` defaults to False (:81).

## Design

1. One algorithm, streaming. A track ending at t can only be mended to a track starting at t + 1, and tracks
   ending at different frames never compete for the same successor, so mending is decided exactly when frame t + 1
   is processed. Per step: overlap pairs with the previous frame -> greedy one-to-one -> unmatched objects get new
   pre-mend ids in label order (`next_id++`, as `relabel` would number them) -> tracks of frame t not continued
   (ended at t) are mended, in id order, to the new tracks of t + 1 -> output id = chain root. The output frame is
   final when returned: no whole-day array, no second pass. Batch `track(data)` = stream over `data` and stack;
   it equals the bug-fixed Python reference exactly (test).
2. Rust `rust/src/tracking.rs`: `struct OverlapTracker` (params; previous frame's input labels `Vec<i32>`; per
   previous-frame label -> track id; per active track: start frame, start and current truncated centroid, root id;
   `next_id`, frame counter). Overlap counts by one pixel pass into a `HashMap<(i32, i32), u32>` (sparse, never
   objects x grid), areas and centroid sums by a second pass over the current frame. Exposed by PyO3 as
   `_rust.OverlapTracker(percent_overlap, cap_overlap, mend_tracks, mend_dist)` with `.step(labels int32 2-D) ->
   int32 2-D` (GIL released) and `.records()` (frame, track, label, area, cy, cx) and `.events()`
   (frame, kind split|merge|mend, track, other_track).
3. Python `ObjectTracker(percent_overlap=0.0, mend_tracks=True, mend_dist=3, cap_overlap=False, backend="auto")`:
   `track(data)` (3-D array or iterable of 2-D frames, returns 3-D int32 as before), new `step(labels)` and
   `track_iter(frames)` (generator, streaming), `table()` records DataFrame, `trackprops` unchanged. `backend`
   "auto" uses Rust when `HAS_RUST`, else a numpy implementation of the same streaming step; "python-batch"
   keeps today's code path (the reference). The old helper methods stay (public API).
4. Defaults change: `mend_tracks=True` (user). `cap_overlap=True` restores the original score.
5. Memory: two int32 frames + output (~75 MB CONUS) + per-object records (~250 B x ~30k objects/day).

## Tests

- Rust stream == `backend="python-batch"` (bug-fixed reference), mending on and off, Dice and capped, several
  `percent_overlap`, on synthetic moving / splitting / merging storms and the 9 May-2024 MRMS crops (GFS-RTX).
- Rust (mend off, cap on) == original cc70583 `ObjectTracker` (vendored to `tests/legacy/`), small grids.
- Python streaming fallback == Rust (same cases); merge/split checks are no-ops (assert on the reference).
- Memory/speed (GFS-RTX, 4 cores): a synthetic CONUS day, RSS and s/day vs `StreamingTracker` and TD-1071's
  `Tracker`.
- Report: mending on vs off (median and px-weighted track length, share of object-hours with 6/6 frames, id
  switches proxy); animations on the tuned EW split.

## Pass criteria

`step` / `track_iter` over a 144-frame CONUS day (synthetic `bench/bench_conus.py trk`, GFS-RTX, 4 cores): <= 1 GB
RSS and well under the segmentation time (target <= 0.1 s/frame). `track()` returns the 3-D array by contract
(filled in place, no list + stack copy) and is exempt.

## Alternatives

The numpy streaming step alone meets the memory goal; Rust is for speed (CONUS: ~3x per frame). Extending
`StreamingTracker` instead was rejected: it advects and bridges gaps (a different algorithm); both stay.
`ObjectTracker` = the user's overlap method (5-10 min steps, mending); `StreamingTracker` = motion-advected
matching for coarser steps or fast storms.

## Review resolution (Goldfish, 2026-11-17 10:37 pm ET; 0 blockers, 8 major, 6 nit)

- A1 kept, adapted: `use_rust=None|True|False` (repo convention) instead of `backend`; Rust and numpy streaming
  implementations; the batch code stays in the class as private `_track_batch` (the tests' oracle) rather than
  moving to `tests/legacy/`, because its helpers are public API; cc70583 is vendored in `tests/legacy/`.
- A2 kept: pass criteria above; `track()` preallocates.
- A3 kept: mending rule = nearest new-track start centroid to (end centroid + mean motion of the ended pre-mend
  track), grid points, <= `mend_dist`, ties lowest id; "no motion advection" applies to matching only.
- A4 kept: Alternatives above.
- A5 kept: v1.3.0, `mend_tracks` default True noted in the class docstring and README; fallback speed noted.
- A6 kept: 2-D int, `<= 0` background, cast to C-contiguous int32 (ValueError past the int32 range), fixed shape
  (ValueError).
- A7 kept: events `(frame, kind, track, other_track, label)`; split / merge = overlap pairs that lost the greedy
  match (StreamingTracker's rule); mend events too.
- A8 kept: tests for ties (split and merge), the `mend_dist` boundary (inclusive) and centroid truncation; sort
  key (score desc, previous track id asc, label asc) stated in the code docstrings.
- B1-B6 kept as wording (B2: `cap_overlap` documented as "same threshold twice as strict"); B4's CI stand-in is the
  synthetic `storms()` generator in `tests/test_objecttracker_stream.py`; the mending report is informational.
