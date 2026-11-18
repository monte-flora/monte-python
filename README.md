![Unit Tests](https://github.com/WarnOnForecast/MontePython/actions/workflows/continuous_integration.yml/badge.svg)

# MontePython


Methods for Object-based and Neighborhood Threat Evaluation in Python (MontePython) is a Python module for meteorological-based image segementation, matching, quality control and tracking. This package was originally built for [NOAA's Warn-on-Forecast System](https://www.nssl.noaa.gov/projects/wof/) (WoFS), but is applicable to any gridded data. MontePython also includes a 7-scheme storm identification and classification scheme from [Potvin et al. (2022)](https://journals.ametsoc.org/view/journals/atot/39/7/JTECH-D-21-0141.1.xml?rskey=dHTUIB&result=5).


## Object Identification 

There are three different object identification (ID) methods in MontePython: single threhsold, enhanced watershed, and iterative watershed. To see examples of how to use these different methods, see the [object ID tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_id.ipynb)

## Object Quality Control 

Object identification methods are imperfect and as such a post-processing step is often required to improve the identification. MontePython's ObjectQualityControler has multiple options for qc'ing the output of object identification. To see examples of differnt quality control methods, see the [object quality control tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_quality_control.ipynb)


## Object Matching 

MontePython can perform matching between two sets of objects (e.g., forecasts and observations). The default matching is based on a total interest score in [Skinner et al. (2018)](https://journals.ametsoc.org/view/journals/wefo/33/5/waf-d-18-0020_1.xml), which includes minimum displacement, centroid displacement, and time displacement. The user can also provide a custom scoring function. Additionally, the code can also handle non-one-to-one matching. This is applicable to situations where we might want to associate multiple observed objects with a single forecast object. To see examples of how to perform object matching, see the [object matching tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_matching.ipynb)

## Object Tracking

Once identified, objects over time can be tracked using MontePython's ObjectTracker. The algorithm is designed for high temporal resolution data such that matches in time can be largely determined by the amount of overlap. The tracking algorithm can handle both splits and mergers. To see examples of how to perform object tracking, see the [object tracking tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_tracking.ipynb)

Since v1.3 `ObjectTracker` streams (`step(labels)` / `track_iter(frames)`: two frames in memory, ids final per
frame; `track()` stacks them), runs on the Rust core when built (`use_rust=None`; numpy otherwise), mends broken
tracks by default (`mend_tracks=True`, `mend_dist=3` grid points; v1.2 defaulted to False and its mending had no
effect, so track ids differ from v1.2), and keeps the original 0.5-bounded overlap score as `cap_overlap=True`.

## Storm Identification and Classification 
[Potvin et al. (2022)](https://journals.ametsoc.org/view/journals/atot/39/7/JTECH-D-21-0141.1.xml?rskey=dHTUIB&result=5) developed a storm identification and classification method, which is applicable to both convection-allowing model output and observed radar. To see examples of how to use this method , see the [storm mode tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/storm_mode_classification.ipynb)

## Fast EnhancedWatershed (Rust core) and streaming tracking

`EnhancedWatershed` has a compiled Rust core (`monte_python._rust`, built by maturin) that gives the same labels as
the pure-Python code, about 20x faster (~0.04 s for a CONUS MRMS 0.02 deg frame, 1750 x 3500). It is used
automatically when built; `use_rust=False` forces the Python code, `use_rust=True` raises if the core is missing.

- `monte_python.split_objects(field, threshold=40, split_px=300, ...)`: threshold objects, with those larger than
  `split_px` split into EnhancedWatershed cells grown back over the object (the pieces tile the object).
- `monte_python.StreamingTracker`: frame-by-frame tracking with motion-advected overlap matching; memory is one
  frame plus the active tracks (a CONUS day at 10 min stays well under 1 GB).

### Install

    pip install "monte_python @ git+https://github.com/monte-flora/monte-python@<sha>"   # needs a Rust toolchain
    pip install "monte_python[all] @ git+..."    # + matplotlib, opencv (MCIT), numba/xarray (storm mode)

Without a Rust toolchain, `PYTHONPATH=<checkout>` runs the pure-Python fallback.

### Requirements
- numpy, scipy, scikit-image, pandas (core)
- optional: matplotlib (`plot`), opencv-python-headless (`mcit`), numba + xarray (`storm-mode`)
- build: maturin and Rust (stable)
