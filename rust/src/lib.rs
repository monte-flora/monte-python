//! PyO3 bindings: `monte_python._rust`.
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

pub mod watershed;

fn to_pyarray<'py>(py: Python<'py>, ny: usize, nx: usize, v: Vec<i32>) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let a = Array2::from_shape_vec((ny, nx), v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(a.into_pyarray(py))
}

/// enhanced_watershed(data, min_thresh, max_bin, data_increment, area_threshold, min_size, min_infl, max_infl)
///
/// The full marked grid of `EnhancedWatershed` (int32: labels > 0, -1 unmarked, -3 globbed) for a C-contiguous
/// float32 or float64 2-D array. `max_bin`, `min_infl`, `max_infl` are computed by the Python wrapper.
#[pyfunction]
#[pyo3(signature = (data, min_thresh, max_bin, data_increment, area_threshold, min_size, min_infl, max_infl))]
#[allow(clippy::too_many_arguments)]
fn enhanced_watershed<'py>(py: Python<'py>, data: &Bound<'py, PyAny>, min_thresh: f64, max_bin: i32,
                           data_increment: f64, area_threshold: usize, min_size: usize, min_infl: i64,
                           max_infl: i64) -> PyResult<Bound<'py, PyArray2<i32>>> {
    if max_bin < 0 {
        return Err(PyValueError::new_err("max_bin < 0 (max_thresh below min_thresh)"));
    }
    let p = watershed::Params { min_thresh, data_increment, max_bin, max_size: area_threshold, min_size, min_infl,
                                max_infl };
    if let Ok(a) = data.extract::<PyReadonlyArray2<f64>>() {
        let (ny, nx) = (a.shape()[0], a.shape()[1]);
        let s = a.as_slice().map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
        let out = py.detach(|| watershed::enhanced_watershed(ny, nx, s, &p));
        return to_pyarray(py, ny, nx, out);
    }
    if let Ok(a) = data.extract::<PyReadonlyArray2<f32>>() {
        let (ny, nx) = (a.shape()[0], a.shape()[1]);
        let s = a.as_slice().map_err(|_| PyValueError::new_err("data must be C-contiguous"))?;
        let out = py.detach(|| watershed::enhanced_watershed(ny, nx, s, &p));
        return to_pyarray(py, ny, nx, out);
    }
    Err(PyTypeError::new_err("data must be a 2-D float32 or float64 numpy array"))
}

/// grow_labels(labels int32 2-D, mask bool 2-D) -> int32 2-D: labels grown into `mask` by 8-connected BFS
/// (each mask pixel takes the label of the geodesically nearest labelled pixel; unreachable pixels stay 0).
#[pyfunction]
fn grow_labels<'py>(py: Python<'py>, labels: PyReadonlyArray2<'py, i32>, mask: PyReadonlyArray2<'py, bool>)
                    -> PyResult<Bound<'py, PyArray2<i32>>> {
    let (ny, nx) = (labels.shape()[0], labels.shape()[1]);
    if mask.shape() != labels.shape() {
        return Err(PyValueError::new_err("labels and mask shapes differ"));
    }
    let l = labels.as_slice().map_err(|_| PyValueError::new_err("labels must be C-contiguous"))?;
    let m = mask.as_slice().map_err(|_| PyValueError::new_err("mask must be C-contiguous"))?;
    let out = py.detach(|| watershed::grow_labels(ny, nx, l, m));
    to_pyarray(py, ny, nx, out)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enhanced_watershed, m)?)?;
    m.add_function(wrap_pyfunction!(grow_labels, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
