//! Enhanced watershed (Lakshmanan et al. 2009), a line-for-line port of the bug-fixed Python reference in
//! `monte_python/EnhancedWatershedSegmenter.py`: same visit order (row-major bins, LIFO stacks, row-major 3x3
//! neighbours), so the labels are identical. Grids are row-major, flat index = row * nx + col.

pub const UNMARKED: i32 = -1;
pub const GLOBBED: i32 = -3;

pub struct Params {
    pub min_thresh: f64,
    pub data_increment: f64,
    pub max_bin: i32,
    pub max_size: usize,
    pub min_size: usize,
    pub min_infl: i64,
    pub max_infl: i64,
}

/// Python's float floor division `a // b` (CPython `float_floor_div`), so the bins match numpy's `floor_divide`.
fn py_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b; // fmod
    let mut div = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let fl = div.floor();
        if div - fl > 0.5 {
            fl + 1.0
        } else {
            fl
        }
    } else {
        0.0f64.copysign(a / b)
    }
}

/// Quantized bins: floor((trunc(x) - min) / inc) clipped to max_bin; -1 below threshold (bin <= 0 when min == 0)
/// and for non-finite x.
pub fn quantize<T: Copy + Into<f64>>(data: &[T], p: &Params) -> Vec<i32> {
    data.iter()
        .map(|&v| {
            let x: f64 = v.into();
            if !x.is_finite() {
                return -1;
            }
            let d = py_floordiv(x.trunc() - p.min_thresh, p.data_increment);
            let bad = if p.min_thresh == 0.0 { d <= 0.0 } else { d < 0.0 };
            if bad {
                -1
            } else if d > p.max_bin as f64 {
                p.max_bin
            } else {
                d as i32
            }
        })
        .collect()
}

/// Bucket grid of centres for `is_closest`: centres are added bin by bin (high to low) so the set "bins >= b / 2"
/// is always the inserted prefix.
struct CentreGrid {
    cell: i64,
    gy: i64,
    gx: i64,
    cells: Vec<Vec<(i64, i64)>>,
}

impl CentreGrid {
    fn new(ny: usize, nx: usize, cell: i64) -> Self {
        let gy = (ny as i64 + cell - 1) / cell;
        let gx = (nx as i64 + cell - 1) / cell;
        CentreGrid { cell, gy: gy.max(1), gx: gx.max(1), cells: vec![Vec::new(); (gy.max(1) * gx.max(1)) as usize] }
    }
    fn add(&mut self, y: i64, x: i64) {
        let k = (y / self.cell) * self.gx + x / self.cell;
        self.cells[k as usize].push((y, x));
    }
    /// True if any inserted centre is strictly closer (squared distance) to (py, px) than `my_d2`.
    fn any_closer(&self, py: i64, px: i64, my_d2: i64) -> bool {
        if my_d2 <= 0 {
            return false;
        }
        let cy = py / self.cell;
        let cx = px / self.cell;
        let r = (my_d2 as f64).sqrt();
        let kmax = (r / self.cell as f64).ceil() as i64 + 1;
        for k in 0..=kmax {
            // ring k around (cy, cx); skip rings whose nearest edge is already >= r
            if k >= 2 {
                let edge = ((k - 1) * self.cell) as f64;
                if edge >= r {
                    break;
                }
            }
            let y0 = cy - k;
            let y1 = cy + k;
            for gy in y0..=y1 {
                if gy < 0 || gy >= self.gy {
                    continue;
                }
                let full_row = gy == y0 || gy == y1;
                let mut gx = cx - k;
                while gx <= cx + k {
                    if gx >= 0 && gx < self.gx {
                        for &(y, x) in &self.cells[(gy * self.gx + gx) as usize] {
                            let d2 = (py - y) * (py - y) + (px - x) * (px - x);
                            if d2 < my_d2 {
                                return true;
                            }
                        }
                    }
                    if full_row || k == 0 {
                        gx += 1;
                    } else {
                        gx += 2 * k; // only the two side cells on inner rows
                    }
                }
            }
        }
        false
    }
}

pub struct Watershed<'a> {
    ny: usize,
    nx: usize,
    q: &'a [i32],
    p: &'a Params,
    marked: Vec<i32>,
}

#[inline]
fn neighbours(ny: usize, nx: usize, idx: usize, out: &mut [usize; 9]) -> usize {
    let r = idx / nx;
    let c = idx % nx;
    let r0 = r.saturating_sub(1);
    let r1 = (r + 1).min(ny - 1);
    let c0 = c.saturating_sub(1);
    let c1 = (c + 1).min(nx - 1);
    let mut n = 0;
    for i in r0..=r1 {
        for j in c0..=c1 {
            out[n] = i * nx + j;
            n += 1;
        }
    }
    n
}

/// Centres per bin (flat indices, visit order), as `find_local_maxima`.
pub fn find_local_maxima(ny: usize, nx: usize, q: &[i32], p: &Params) -> Vec<Vec<usize>> {
    let nbins = (p.max_bin + 1) as usize;
    // pixels per bin in row-major order (counting sort)
    let mut counts = vec![0usize; nbins + 1];
    for &v in q {
        if v >= 0 {
            counts[v as usize + 1] += 1;
        }
    }
    for b in 0..nbins {
        counts[b + 1] += counts[b];
    }
    let mut fill = counts.clone();
    let mut order = vec![0usize; counts[nbins]];
    for (i, &v) in q.iter().enumerate() {
        if v >= 0 {
            order[fill[v as usize]] = i;
            fill[v as usize] += 1;
        }
    }
    let mut marked = vec![UNMARKED; ny * nx];
    let mut centers: Vec<Vec<usize>> = vec![Vec::new(); nbins];
    for b in (0..nbins).rev() {
        let frac = if p.max_bin > 0 { b as f64 / p.max_bin as f64 } else { 0.0 };
        let infl = p.min_infl + (frac * (p.max_infl - p.min_infl) as f64).round_ties_even() as i64;
        for &idx in &order[counts[b]..counts[b + 1]] {
            if marked[idx] != UNMARKED {
                continue;
            }
            let r = (idx / nx) as i64;
            let c = (idx % nx) as i64;
            let r0 = (r - infl).max(0) as usize;
            let r1 = ((r + infl + 1).min(ny as i64)) as usize;
            let c0 = (c - infl).max(0) as usize;
            let c1 = ((c + infl + 1).min(nx as i64)) as usize;
            let mut free = true;
            'scan: for i in r0..r1 {
                for &m in &marked[i * nx + c0..i * nx + c1] {
                    if m != UNMARKED {
                        free = false;
                        break 'scan;
                    }
                }
            }
            if free {
                for i in r0..r1 {
                    for m in &mut marked[i * nx + c0..i * nx + c1] {
                        *m = b as i32;
                    }
                }
                centers[b].push(idx);
            }
        }
    }
    centers
}

impl<'a> Watershed<'a> {
    pub fn new(ny: usize, nx: usize, q: &'a [i32], p: &'a Params) -> Self {
        Watershed { ny, nx, q, p, marked: vec![UNMARKED; ny * nx] }
    }

    fn set_maximum(&mut self, center: usize, bin_lower: i32, foothills: &mut Vec<(usize, Vec<usize>)>,
                   capture_index: i32) -> bool {
        let mut as_bin: Vec<usize> = vec![center];
        let mut as_glob: Vec<usize> = Vec::new();
        let mut marked_so_far: Vec<usize> = Vec::new();
        let mut again = false;
        let center_data = self.q[center];
        let mut nb = [0usize; 9];
        while let Some(px) = as_bin.pop() {
            if self.marked[px] != UNMARKED {
                continue;
            }
            self.marked[px] = capture_index;
            marked_so_far.push(px);
            let n = neighbours(self.ny, self.nx, px, &mut nb);
            for &pix in &nb[..n] {
                if self.marked[pix] == UNMARKED {
                    let pd = self.q[pix];
                    if !again && pd >= 0 && pd < center_data {
                        again = true;
                    }
                    if pd >= bin_lower {
                        as_bin.push(pix);
                    } else if pd >= 0 {
                        as_glob.push(pix);
                    }
                }
            }
        }
        if bin_lower == 0 {
            again = false;
        }
        let big_enough = marked_so_far.len() >= self.p.max_size;
        if big_enough {
            foothills.push((center, as_glob));
        } else if marked_so_far.len() <= self.p.min_size || again {
            for &m in &marked_so_far {
                self.marked[m] = UNMARKED;
            }
        }
        big_enough || !again
    }

    fn remove_foothills(&mut self, bin_lower: i32, grid: &CentreGrid, foothills: &mut Vec<(usize, Vec<usize>)>) {
        let nx = self.nx as i64;
        let mut nb = [0usize; 9];
        for (center, glob) in foothills.drain(..) {
            let cy = center as i64 / nx;
            let cx = center as i64 % nx;
            let mut hills = glob;
            while let Some(pt) = hills.pop() {
                self.marked[pt] = GLOBBED;
                let qpt = self.q[pt];
                let n = neighbours(self.ny, self.nx, pt, &mut nb);
                for &idx in &nb[..n] {
                    if self.marked[idx] != UNMARKED {
                        continue;
                    }
                    let qi = self.q[idx];
                    if qi >= 0 && qi < bin_lower {
                        let ok = qi <= qpt || {
                            let py = idx as i64 / nx;
                            let px = idx as i64 % nx;
                            let my_d2 = (py - cy) * (py - cy) + (px - cx) * (px - cx);
                            !grid.any_closer(py, px, my_d2)
                        };
                        if ok {
                            hills.push(idx);
                        }
                    }
                }
            }
        }
    }

    /// `grow_centers`: the full marked grid (labels > 0, UNMARKED, GLOBBED).
    pub fn grow_centers(mut self, centers: &[Vec<usize>]) -> Vec<i32> {
        let nbins = centers.len();
        let mut grid = CentreGrid::new(self.ny, self.nx, 32);
        let mut next_to_add = nbins as i64 - 1; // bins > next_to_add are in the grid
        let mut deferred_to_next: Vec<usize> = Vec::new();
        let mut foothills: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut capture_index: i32 = 1;
        let nx = self.nx;
        for b in (0..nbins).rev() {
            let mut bin_lower = b as i32 - 1;
            let deferred_from_last = std::mem::take(&mut deferred_to_next);
            for &center in deferred_from_last.iter().chain(centers[b].iter()) {
                if bin_lower < 0 {
                    bin_lower = 0;
                }
                if self.marked[center] == UNMARKED {
                    if self.set_maximum(center, bin_lower, &mut foothills, capture_index) {
                        capture_index += 1;
                    } else {
                        deferred_to_next.push(center);
                    }
                }
            }
            // is_closest looks at centres in bins >= int(b / 2)
            let bin_thresh = (b / 2) as i64;
            while next_to_add >= bin_thresh {
                for &c in &centers[next_to_add as usize] {
                    grid.add((c / nx) as i64, (c % nx) as i64);
                }
                next_to_add -= 1;
            }
            self.remove_foothills(bin_lower, &grid, &mut foothills);
        }
        self.marked
    }
}

pub fn enhanced_watershed<T: Copy + Into<f64>>(ny: usize, nx: usize, data: &[T], p: &Params) -> Vec<i32> {
    if ny == 0 || nx == 0 {
        return Vec::new();
    }
    let q = quantize(data, p);
    let centers = find_local_maxima(ny, nx, &q, p);
    Watershed::new(ny, nx, &q, p).grow_centers(&centers)
}

/// Multi-source 8-connected BFS of labels (> 0) into `mask` pixels with label 0; sources in row-major order,
/// FIFO, neighbours row-major: each mask pixel connected to a labelled one takes the label of the geodesically
/// nearest source (ties by queue order).
pub fn grow_labels(ny: usize, nx: usize, labels: &[i32], mask: &[bool]) -> Vec<i32> {
    let mut out: Vec<i32> = labels.iter().map(|&l| if l > 0 { l } else { 0 }).collect();
    let mut queue: std::collections::VecDeque<usize> =
        out.iter().enumerate().filter(|(_, &l)| l > 0).map(|(i, _)| i).collect();
    let mut nb = [0usize; 9];
    while let Some(i) = queue.pop_front() {
        let l = out[i];
        let n = neighbours(ny, nx, i, &mut nb);
        for &j in &nb[..n] {
            if out[j] == 0 && mask[j] {
                out[j] = l;
                queue.push_back(j);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floordiv_matches_python() {
        assert_eq!(py_floordiv(1.0, 0.1), 9.0);
        assert_eq!(py_floordiv(7.0, 2.0), 3.0);
        assert_eq!(py_floordiv(-7.0, 2.0), -4.0);
        assert_eq!(py_floordiv(-1.0, 5.0), -1.0);
    }

    #[test]
    fn grid_matches_linear_scan() {
        let mut g = CentreGrid::new(200, 300, 32);
        let pts: Vec<(i64, i64)> = (0..50).map(|k| ((k * 37) % 200, (k * 91) % 300)).collect();
        for &(y, x) in &pts {
            g.add(y, x);
        }
        for py in (0..200).step_by(7) {
            for px in (0..300).step_by(11) {
                for my_d2 in [0i64, 1, 50, 400, 5000, 100000] {
                    let lin = pts.iter().any(|&(y, x)| (py - y).pow(2) + (px - x).pow(2) < my_d2);
                    assert_eq!(g.any_closer(py, px, my_d2), lin, "{py} {px} {my_d2}");
                }
            }
        }
    }

    #[test]
    fn edge_object_is_labelled() {
        let (ny, nx) = (20usize, 20usize);
        let mut d = vec![0.0f64; ny * nx];
        for j in 0..nx {
            d[j] = 50.0; // a one-row object on row 0
        }
        let p = Params { min_thresh: 10.0, data_increment: 1.0, max_bin: 50, max_size: 5, min_size: 2,
                         min_infl: 3, max_infl: 6 };
        let m = enhanced_watershed(ny, nx, &d, &p);
        assert!(m[..nx].iter().all(|&v| v > 0));
    }
}
