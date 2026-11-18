#
# The Enhanced Watershed code is based on the algorithm from Lakshmanan et al. (2009) and was adapted
# from Java code found at https://github.com/lakshmanok/asgbook/blob/master/src/edu/ou/asgbook/segmentation/EnhancedWatershedSegmenter.java.
#
# References
# Valliappa Lakshmanan, Kurt Hondl, and Robert Rabin, 2009: An Efficient, General-Purpose 
#  Technique for Identifying Storm Cells in Geospatial Images. J. Atmos. Oceanic Technol., 26, 523-537.
# https://journals.ametsoc.org/doi/full/10.1175/2008JTECHA1153.1
"""
@author: David John Gagne (djgagne@ou.edu) and Montgomery Flora (monte.flora@noaa.gov)
"""

import numpy as np
from scipy.ndimage import find_objects
from collections import OrderedDict

try:  # compiled core (monte_python/_rust, built by maturin); the pure-Python code below is the fallback
    from . import _rust
except ImportError:  # pragma: no cover - depends on the build
    _rust = None

HAS_RUST = _rust is not None


class EnhancedWatershed(object):
    """
    The enhanced watershed performs image segmentation using a modified version of the traditional watershed technique.
    It includes a size criteria and creates foothills around each object to keep them distinct. The object is used to
    store the quantization and size parameters. It can be used to watershed multiple grids.

    Attributes:
        min_thresh (int): minimum pixel value for pixel to be part of a region
        data_increment (int): quantization interval. Use 1 if you don't want to quantize
        max_thresh (int): values greater than maxThresh are treated as the maximum threshold
        area_threshold (int): clusters smaller than this threshold are ignored.
        dist_btw_objects (int): clearance between local maxima (sets the centre search window).
        min_size (int): regions of <= min_size pixels are dropped (default 6).
        use_rust (bool or None): None uses the compiled core when it is installed, True requires it,
            False forces the pure-Python reference. Both give identical labels.

    Non-finite input (NaN, inf) is treated as below ``min_thresh``.
    """

    def __init__(self, min_thresh, max_thresh, area_threshold, dist_btw_objects, data_increment=1,
                 min_size=6, use_rust=None):
        self.min_thresh = min_thresh
        self.data_increment = data_increment
        self.max_thresh = max_thresh
        self.max_size = area_threshold
        self.min_size = min_size
        self.dist_btw_objects = dist_btw_objects
        self.max_bin = int((self.max_thresh - self.min_thresh) / self.data_increment)
        self.UNMARKED = -1
        self.GLOBBED = -3
        self.TOOSMALL = -4
        if use_rust and not HAS_RUST:
            raise ImportError("use_rust=True but monte_python._rust is not built (pip install with a Rust toolchain)")
        self.use_rust = HAS_RUST if use_rust is None else bool(use_rust)

    def label(self, input_grid, only_objects=True):
        """
        Labels input grid using enhanced watershed algorithm.

        Args:
            input_grid (numpy.ndarray): Grid to be labeled.
            only_objects (bool): Only return object pixel values on final grid

        Returns:
            Array of labeled pixels (int32)
        """
        if self.use_rust:
            marked = self._label_rust(input_grid)
        else:
            pixels, q_data = self.quantize(input_grid)
            centers = self.find_local_maxima(pixels, q_data)
            marked = self.grow_centers(centers, q_data)
        if only_objects:
            marked = np.where(marked > 0, marked, 0).astype(np.int32)
        return marked

    def _label_rust(self, input_grid):
        grid = np.asarray(input_grid)
        if grid.dtype not in (np.float32, np.float64):
            grid = grid.astype(np.float64)
        grid = np.ascontiguousarray(grid)
        if grid.ndim != 2:
            raise ValueError("input_grid must be 2-D")
        min_infl, max_infl = self._infl_dists()
        return _rust.enhanced_watershed(grid, float(self.min_thresh), int(self.max_bin), float(self.data_increment),
                                        int(self.max_size), int(self.min_size), int(min_infl), int(max_infl))

    @staticmethod
    def size_filter(labeled_grid, min_size):
        """
        Removes labeled objects that are smaller than min_size, and relabels the remaining objects.

        Args:
            labeled_grid: Grid that has been labeled
            min_size: Minimium object size.
        Returns:
            Labeled array with re-numbered objects to account for those that have been removed
        """
        out_grid = np.zeros(labeled_grid.shape, dtype=int)
        slices = find_objects(labeled_grid)
        j = 1
        for i, s in enumerate(slices):
            if s is None:
                continue
            box = labeled_grid[s]
            size = np.count_nonzero(box == i + 1)
            if size >= min_size and box.shape[0] > 1 and box.shape[1] > 1:
                out_grid[s][box == i + 1] = j
                j += 1
        return out_grid

    def _infl_dists(self):
        min_infl = int(np.round(1 + 0.5 * np.sqrt(self.dist_btw_objects)))
        max_infl = 2 * min_infl
        return min_infl, max_infl

    def find_local_maxima(self, pixels, q_data):
        """
        Finds the local maxima in the inputGrid and perform region growing to identify objects.

        Args:
            pixels: dictionary of quantized pixel values
            q_data: 2D array representation of quantized input data
        Returns:
            dict of bin -> list of centre pixels (row, col), in visit order.
        """
        centers = OrderedDict()
        for p in pixels.keys():
            centers[p] = []
        marked = np.full(q_data.shape, self.UNMARKED, dtype=np.int32)
        ny, nx = q_data.shape
        MIN_INFL, MAX_INFL = self._infl_dists()
        # Find the maxima. These are high-values with enough clearance
        # around them.
        # Work from high to low bins. The pixels in the highest bin mark their
        # neighborhoods first. If you did it from low to high the lowest maxima
        # would mark their neighborhoods first and interfere with the identification of higher maxima.
        for b in sorted(pixels.keys(), reverse=True):
            # Square starts large with high intensity bins and gets smaller with low intensity bins.
            infl_dist = MIN_INFL + int(np.round(float(b) / self.max_bin * (MAX_INFL - MIN_INFL)))
            for p in pixels[b]:
                if marked[p] == self.UNMARKED:
                    # A new centre if no pixel in the square around it (clipped to the grid) was marked already
                    # by a higher-intensity centre; the square is then marked with this bin. (The original loop
                    # marked pixels one by one and reverted on the first marked one: same result.)
                    win = marked[max(p[0] - infl_dist, 0):min(p[0] + infl_dist + 1, ny),
                                 max(p[1] - infl_dist, 0):min(p[1] + infl_dist + 1, nx)]
                    if (win == self.UNMARKED).all():
                        win[...] = b
                        centers[b].append(p)
        return centers

    def grow_centers(self, centers, q_data):
        """
        Grow regions from the centres, bin by bin from high to low.

        Args:
            centers: dict of bin -> list of centre pixels (from find_local_maxima)
            q_data: quantized data

        Returns:
            int32 grid: object labels (> 0), UNMARKED (-1) and GLOBBED (-3) pixels
        """
        marked = np.full(q_data.shape, self.UNMARKED, dtype=np.int32)
        deferred_from_last = []
        deferred_to_next = []
        center_keys = np.array(list(centers.keys()))[::-1]
        capture_index = 1
        foothills = []
        # Work from high to low bins.
        for b in center_keys:
            bin_lower = b - 1
            deferred_from_last[:] = deferred_to_next[:]
            del deferred_to_next[:]
            new_centers = len(centers[b])
            old_centers = len(deferred_from_last)
            tot_centers = new_centers + old_centers
            for i in range(tot_centers):
                # done this way to minimize memory overhead of maintaining two lists
                if i < old_centers:
                    center = deferred_from_last[i]
                else:
                    center = centers[b][i - old_centers]
                if bin_lower < 0:
                    bin_lower = 0
                if marked[center] == self.UNMARKED:
                    captured = self.set_maximum(q_data, marked, center, bin_lower, foothills, capture_index)
                    if not captured:
                        # decrement to lower value to see if it'll get big enough
                        deferred_to_next.append(center)
                    else:
                        capture_index += 1
                # this is the last one for this bin
            self.remove_foothills(q_data, marked, b, bin_lower, centers, foothills)
        del deferred_from_last[:]
        del deferred_to_next[:]
        return marked

    @staticmethod
    def _neighbours(p, shape):
        """3x3 neighbourhood of p (row-major, p included), clipped to the grid."""
        ny, nx = shape
        for i in range(max(p[0] - 1, 0), min(p[0] + 2, ny)):
            for j in range(max(p[1] - 1, 0), min(p[1] + 2, nx)):
                yield (i, j)

    def set_maximum(self, q_data, marked, center, bin_lower, foothills, capture_index):
        """
        Grow a region at a certain bin level and check if the region has reached the maximum size.

        Args:
            q_data: Quantized data array
            marked: Array marking points that are objects
            center: Coordinates of the center pixel of the region being grown
            bin_lower: Intensity level of lower bin being evaluated
            foothills: List of points that are associated with a center but fall outside the the size or
                intensity criteria
            capture_index:
        Returns:
            True if the object is finished growing and False if the object should be grown again at the next
            threshold level.
        """
        as_bin = []  # pixels to be included in peak
        as_glob = []  # pixels to be globbed up as part of foothills
        marked_so_far = []  # pixels that have already been marked
        will_be_considered_again = False
        as_bin.append(center)
        center_data = q_data[center]
        shape = q_data.shape
        while len(as_bin) > 0:
            p = as_bin.pop(-1)  # remove and return last pixel in as_bin
            if marked[p] != self.UNMARKED:  # already processed
                continue
            marked[p] = capture_index
            marked_so_far.append(p)

            # check neighbors
            for pixel in self._neighbours(p, shape):
                # is neighbor part of peak or part of mountain?
                if marked[pixel] == self.UNMARKED:
                    p_data = q_data[pixel]
                    if (not will_be_considered_again) and (p_data >= 0) and (p_data < center_data):
                        will_be_considered_again = True
                    if p_data >= bin_lower:
                        as_bin.append(pixel)
                    # Do not check that this is the closest: this way, a narrow channel of globbed pixels form
                    elif p_data >= 0:
                        as_glob.append(pixel)
        if bin_lower == 0:
            will_be_considered_again = False
        big_enough = len(marked_so_far) >= self.max_size
        if big_enough:
            # remove lower values within region of influence
            foothills.append((center, as_glob))

        elif len(marked_so_far) <= self.min_size:
            for m in marked_so_far:
                marked[m] = self.UNMARKED

        elif will_be_considered_again:  # remove the check if you want to ignore regions smaller than max_size
            for m in marked_so_far:
                marked[m] = self.UNMARKED
            del as_bin[:]
            del as_glob[:]
            del marked_so_far[:]
        return big_enough or (not will_be_considered_again)

    def remove_foothills(self, q_data, marked, bin_num, bin_lower, centers, foothills):
        """
        Mark points determined to be foothills as globbed, so that they are not included in
        future searches. Also searches neighboring points to foothill points to determine
        if they should also be considered foothills.

        Args:
            q_data: Quantized data
            marked: Marked
            bin_num: Current bin being searched
            bin_lower: Next bin being searched
            centers: dictionary of local maxima considered to be object centers
            foothills: List of foothill points being removed.
        """
        hills = []
        shape = q_data.shape
        for foot in foothills:
            center = foot[0]
            hills[:] = foot[1][:]
            # remove all foothills
            while len(hills) > 0:
                # mark this point
                pt = hills.pop(-1)
                marked[pt] = self.GLOBBED
                for index in self._neighbours(pt, shape):
                    # is neighbor part of peak or part of mountain?
                    if marked[index] == self.UNMARKED:
                        # will let in even minor peaks
                        if (q_data[index] >= 0) and \
                                (q_data[index] < bin_lower) and \
                                ((q_data[index] <= q_data[pt]) or self.is_closest(index, center, centers, bin_num)):
                            hills.append(index)
        del foothills[:]

    @staticmethod
    def is_closest(point, center, centers, bin_num):
        bin_thresh = int(bin_num / 2)
        py, px = int(point[0]), int(point[1])
        my_dist = (py - int(center[0])) ** 2 + (px - int(center[1])) ** 2
        for o_bin in range(bin_thresh, len(centers.keys())):
            for c in centers[o_bin]:
                if (py - int(c[0])) ** 2 + (px - int(c[1])) ** 2 < my_dist:
                    return False
        return True

    def quantize(self, input_grid):
        """
        Quantize a grid into discrete steps based on input parameters.

        Bins: ``floor((trunc(x) - min_thresh) / data_increment)`` clipped to ``max_bin`` (the original int32-cast
        rule); bins < 0 (and bin 0 when ``min_thresh == 0``) and non-finite values are -1 (not a candidate).

        Args:
            input_grid: 2-d array of values

        Returns:
            Dictionary of value pointing to pixel locations, and quantized 2-d array of data
        """
        pixels = dict()
        for i in range(self.max_bin + 1):
            pixels[i] = []

        q_data = self.quantize_grid(input_grid)
        good_points = np.nonzero(q_data >= 0)
        for i, j in zip(*good_points):
            pixels[int(q_data[i, j])].append((int(i), int(j)))
        return pixels, q_data

    def quantize_grid(self, input_grid):
        """The quantized int32 grid (see `quantize`)."""
        grid = np.asarray(input_grid, dtype=np.float64)
        finite = np.isfinite(grid)
        with np.errstate(invalid='ignore'):
            data = np.floor_divide(np.trunc(np.where(finite, grid, 0.0)) - self.min_thresh, self.data_increment)
        if self.min_thresh == 0:
            bad = data <= 0
        else:
            bad = data < 0
        data = np.minimum(data, self.max_bin)
        data[bad | ~finite] = -1
        return data.astype(np.int32)

    @staticmethod
    def is_valid(point, shape):
        return np.all((np.array(point) >= 0) & (np.array(shape) - np.array(point) > 0))


def rescale_data(data, data_min, data_max, out_min=0.0, out_max=100.0):
    """
    Rescale your input data so that is ranges over integer values, which will perform better in the watershed.

    Args:
        data: 2D or 3D ndarray being rescaled
        data_min: minimum value of input data for scaling purposes
        data_max: maximum value of input data for scaling purposes
        out_min: minimum value of scaled data
        out_max: maximum value of scaled data

    Returns:
        Linearly scaled ndarray
    """
    return (out_max - out_min) / (data_max - data_min) * (data - data_min) + out_min
