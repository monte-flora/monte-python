//! Streaming ObjectTracker (overlap matching + mending), the same algorithm as `monte_python.object_tracking.
//! _OverlapStepper` (the numpy fallback) and, stacked over frames, as `ObjectTracker(backend="python-batch")`.
//!
//! Per frame t (> 0): overlap pairs with frame t - 1 are scored k n / (|A| + |B|) (k = 1 with the original 0.5
//! cap, else 2 = Dice), kept if > `percent_overlap`, ranked by score (ties: previous pre-mend track id, then
//! label, ascending) and matched greedily one-to-one. Unmatched objects start tracks (pre-mend ids in label
//! order). With mending, unmended tracks that ended at frame t - g (g = 1 .. `mend_gap`), in (g, id) order, take
//! the nearest unclaimed new track whose start centroid is within `mend_dist` of their end centroid + g x mean
//! motion (truncated centroids, row and column); output ids follow mend chains to their root. Memory: two index frames and the active tracks.
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Trk {
    root: i64,
    x0: i64, // start centroid row (truncated)
    y0: i64, // start centroid column
    x1: i64, // latest centroid row
    y1: i64,
    n: i64,  // frames seen
}

pub type Record = (i64, i64, i32, u64, f64, f64); // frame, track, label, area, cy, cx
pub type Event = (i64, &'static str, i64, i64, i32); // frame, kind, track, other_track, label

pub struct Tracker {
    po: f64,
    k: f64,
    mend: bool,
    mend_dist: f64,
    mend_gap: i64,
    dormant: Vec<(i64, i64)>, // (end frame, pre-mend id): ended, unmended tracks within mend_gap
    frame: i64,
    shape: Option<(usize, usize)>,
    prev_idx: Vec<u32>, // per pixel: 0 or 1 + object index of the previous frame
    prev_pid: Vec<i64>, // pre-mend track id per previous object
    prev_area: Vec<u64>,
    next_id: i64,
    trk: HashMap<i64, Trk>, // active tracks by pre-mend id
    pub records: Vec<Record>,
    pub events: Vec<Event>,
}

/// Sorted positive labels of `labels` and the per-pixel index (0 = background, 1 + rank of the label).
fn index_labels(labels: &[i32]) -> (Vec<i32>, Vec<u32>) {
    let max = labels.iter().copied().max().unwrap_or(0);
    if max <= 0 {
        return (Vec::new(), vec![0; labels.len()]);
    }
    let mut lab: Vec<i32>;
    let mut idx = vec![0u32; labels.len()];
    if (max as usize) <= 4 * labels.len() + 65536 {
        let mut rank = vec![0u32; max as usize + 1];
        for &v in labels {
            if v > 0 {
                rank[v as usize] = 1;
            }
        }
        lab = Vec::new();
        for (v, r) in rank.iter_mut().enumerate() {
            if *r != 0 {
                lab.push(v as i32);
                *r = lab.len() as u32;
            }
        }
        for (o, &v) in idx.iter_mut().zip(labels) {
            if v > 0 {
                *o = rank[v as usize];
            }
        }
    } else {
        lab = labels.iter().copied().filter(|&v| v > 0).collect();
        lab.sort_unstable();
        lab.dedup();
        let rank: HashMap<i32, u32> = lab.iter().enumerate().map(|(i, &v)| (v, i as u32 + 1)).collect();
        for (o, &v) in idx.iter_mut().zip(labels) {
            if v > 0 {
                *o = rank[&v];
            }
        }
    }
    (lab, idx)
}

impl Tracker {
    pub fn new(percent_overlap: f64, cap_overlap: bool, mend_tracks: bool, mend_dist: f64, mend_gap: i64) -> Self {
        Tracker {
            po: percent_overlap,
            k: if cap_overlap { 1.0 } else { 2.0 },
            mend: mend_tracks,
            mend_dist,
            mend_gap: mend_gap.max(1),
            dormant: Vec::new(),
            frame: -1,
            shape: None,
            prev_idx: Vec::new(),
            prev_pid: Vec::new(),
            prev_area: Vec::new(),
            next_id: 1,
            trk: HashMap::new(),
            records: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Track one frame (row-major `ny` x `nx` labels, <= 0 background); returns the track-id grid.
    pub fn step(&mut self, ny: usize, nx: usize, labels: &[i32]) -> Result<Vec<i32>, String> {
        if let Some(s) = self.shape {
            if s != (ny, nx) {
                return Err(format!("frame {}: shape ({}, {}) differs from the previous frame ({}, {})",
                                   self.frame + 1, ny, nx, s.0, s.1));
            }
        }
        self.shape = Some((ny, nx));
        self.frame += 1;
        let t = self.frame;
        let (lab, idx) = index_labels(labels);
        let nobj = lab.len();
        let mut area = vec![0u64; nobj];
        let mut sr = vec![0u64; nobj];
        let mut sc = vec![0u64; nobj];
        for (p, &i) in idx.iter().enumerate() {
            if i > 0 {
                let j = i as usize - 1;
                area[j] += 1;
                sr[j] += (p / nx) as u64;
                sc[j] += (p % nx) as u64;
            }
        }
        let cy: Vec<f64> = (0..nobj).map(|j| sr[j] as f64 / area[j] as f64).collect();
        let cx: Vec<f64> = (0..nobj).map(|j| sc[j] as f64 / area[j] as f64).collect();
        let mut pid = vec![0i64; nobj];
        let mut matched = vec![false; nobj];
        let first = t == 0;
        // overlap pairs (previous object, current object, n) in (a, b) order
        let mut pairs: Vec<(usize, usize, u64)> = Vec::new();
        if !first && nobj > 0 && !self.prev_pid.is_empty() {
            let mut cnt: HashMap<u64, u64> = HashMap::new();
            for (&a, &b) in self.prev_idx.iter().zip(idx.iter()) {
                if a > 0 && b > 0 {
                    *cnt.entry((a as u64 - 1) * nobj as u64 + (b as u64 - 1)).or_insert(0) += 1;
                }
            }
            let mut keys: Vec<(u64, u64)> = cnt.into_iter().collect();
            keys.sort_unstable();
            pairs = keys.iter().map(|&(key, n)| ((key / nobj as u64) as usize, (key % nobj as u64) as usize, n))
                .collect();
            let mut cand: Vec<(f64, i64, usize, usize)> = pairs.iter()
                .map(|&(a, b, n)| (self.k * n as f64 / (self.prev_area[a] + area[b]) as f64, self.prev_pid[a], b, a))
                .filter(|c| c.0 > self.po)
                .collect();
            cand.sort_by(|p, q| q.0.partial_cmp(&p.0).unwrap().then(p.1.cmp(&q.1)).then(p.2.cmp(&q.2)));
            let mut used_a = vec![false; self.prev_pid.len()];
            for &(_, ap, b, a) in &cand {
                if used_a[a] || matched[b] {
                    continue;
                }
                used_a[a] = true;
                matched[b] = true;
                pid[b] = ap;
            }
        }
        let new: Vec<usize> = (0..nobj).filter(|&b| !matched[b]).collect();
        for &b in &new {
            let id = self.next_id;
            self.next_id += 1;
            pid[b] = id;
            let (x, y) = (cy[b] as i64, cx[b] as i64);
            self.trk.insert(id, Trk { root: id, x0: x, y0: y, x1: x, y1: y, n: 0 });
        }
        if !first {
            let mut cont: Vec<i64> = (0..nobj).filter(|&b| matched[b]).map(|b| pid[b]).collect();
            cont.sort_unstable();
            let mut ended: Vec<i64> = self.prev_pid.iter().copied().filter(|p| cont.binary_search(p).is_err())
                .collect();
            ended.sort_unstable();
            if self.mend {
                self.dormant.extend(ended.iter().map(|&l| (t - 1, l)));
                self.dormant.sort_unstable_by(|p, q| q.0.cmp(&p.0).then(p.1.cmp(&q.1)));
            }
            let mut mended: Vec<i64> = Vec::new();
            if self.mend && !new.is_empty() && !self.dormant.is_empty() {
                let mut taken = vec![false; new.len()];
                for &(end, l) in &self.dormant {
                    let g = (t - end) as f64;
                    let d = self.trk[&l];
                    let (dx, dy) = if d.n > 1 {
                        ((d.x1 - d.x0) as f64 / (d.n - 1) as f64, (d.y1 - d.y0) as f64 / (d.n - 1) as f64)
                    } else {
                        (0.0, 0.0)
                    };
                    let (xp, yp) = (d.x1 as f64 + g * dx, d.y1 as f64 + g * dy);
                    let mut best: Option<(usize, f64)> = None;
                    for (i, &b) in new.iter().enumerate() {
                        if taken[i] {
                            continue;
                        }
                        let o = self.trk[&pid[b]];
                        let dist = (xp - o.x0 as f64).hypot(yp - o.y0 as f64);
                        if dist <= self.mend_dist && best.map_or(true, |(_, bd)| dist < bd) {
                            best = Some((i, dist));
                        }
                    }
                    if let Some((i, _)) = best {
                        taken[i] = true;
                        mended.push(l);
                        let s = pid[new[i]];
                        self.trk.get_mut(&s).unwrap().root = d.root;
                        self.events.push((t, "mend", d.root, 0, lab[new[i]]));
                    }
                }
            }
            for &(a, b, _) in &pairs {
                let (ap, bp) = (self.prev_pid[a], pid[b]);
                if ap == bp {
                    continue;
                }
                if !matched[b] {
                    self.events.push((t, "split", self.trk[&bp].root, self.trk[&ap].root, lab[b]));
                } else if cont.binary_search(&ap).is_err() {
                    self.events.push((t, "merge", self.trk[&ap].root, self.trk[&bp].root, lab[b]));
                }
            }
            if !self.mend {
                for l in &ended {
                    self.trk.remove(l);
                }
            }
            let (gap, trk) = (self.mend_gap, &mut self.trk);
            self.dormant.retain(|&(end, l)| {
                let keep = !mended.contains(&l) && t - end < gap; // else mended or too old for t + 1
                if !keep {
                    trk.remove(&l);
                }
                keep
            });
        }
        let mut out = vec![0i32; nobj + 1];
        for b in 0..nobj {
            let r = self.trk.get_mut(&pid[b]).unwrap();
            r.x1 = cy[b] as i64;
            r.y1 = cx[b] as i64;
            r.n += 1;
            out[b + 1] = r.root as i32;
            self.records.push((t, r.root, lab[b], area[b], cy[b], cx[b]));
        }
        let grid: Vec<i32> = idx.iter().map(|&i| out[i as usize]).collect();
        self.prev_idx = idx;
        self.prev_pid = pid;
        self.prev_area = area;
        Ok(grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxf(ny: usize, nx: usize, y: usize, x: usize, v: i32) -> Vec<i32> {
        let mut f = vec![0; ny * nx];
        for r in y..y + 3 {
            for c in x..x + 3 {
                f[r * nx + c] = v;
            }
        }
        f
    }

    #[test]
    fn overlap_continues_and_mend_joins() {
        let mut t = Tracker::new(0.0, false, true, 5.0, 1);
        let a = t.step(20, 40, &boxf(20, 40, 5, 5, 7)).unwrap();
        let b = t.step(20, 40, &boxf(20, 40, 5, 6, 3)).unwrap(); // overlaps: same track
        assert_eq!(a.iter().max(), Some(&1));
        assert_eq!(b.iter().max(), Some(&1));
        let c = t.step(20, 40, &boxf(20, 40, 5, 10, 9)).unwrap(); // jump 4 px: mended (motion 1 px -> dist 3)
        assert_eq!(c.iter().max(), Some(&1));
        assert_eq!(t.events.len(), 1);
        assert!(t.step(10, 10, &vec![0; 100]).is_err());
    }
}
