"""monte_python: object identification, quality control, matching and tracking.

Heavy or optional dependencies load on first use (PEP 562): ``import monte_python.EnhancedWatershedSegmenter``
needs only numpy + scipy; plotting needs matplotlib (``[plot]``), MCIT needs opencv (``[mcit]``), the storm-mode
classifier needs numba + xarray (``[storm-mode]``).
"""
import importlib
import os

__key__ = 'PACKAGE_VERSION'
__version__ = os.environ[__key__] if __key__ in os.environ else '1.3.0'

_LAZY = {
    'label': '.object_identification',
    'quantize_probabilities': '.object_identification',
    'label_with_qc': '.object_identification',
    'label_per_member': '.object_identification',
    'QualityControler': '.object_quality_control',
    'ObjectMatcher': '.object_matching',
    'ObjectTracker': '.object_tracking',
    'StreamingTracker': '.object_tracking',
    'EnhancedWatershed': '.EnhancedWatershedSegmenter',
    'HAS_RUST': '.EnhancedWatershedSegmenter',
    'split_objects': '.object_split',
    'grow_labels': '.object_split',
    'StormModeClassifier': '.storm_mode_classifier',
    'ObjectVerifier': '.object_verifier',
    # plotting (matplotlib)
    'create_fake_storms': '._synthetic',
    'plot_fake_storms': '._plot.plotter',
    'plot_storm_labels': '._plot.plotter',
    'get_centroid_coords': '._plot.plotter',
    'matching_path': '._plot.plotter',
    'plot_displacement': '._plot.plotter',
    'plot_storm_modes': '._plot.plotter',
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
