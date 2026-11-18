import numpy as np
import pytest
from scipy import ndimage


def smooth_field(shape, seed, sigma=4.0, vmax=70.0, border=0):
    """Smooth random 'reflectivity' in [0, vmax] with several peaks; zero within `border` of the edges."""
    rng = np.random.default_rng(seed)
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    f = (f - f.min()) / (f.max() - f.min() + 1e-12) * vmax
    if border:
        f[:border] = 0
        f[-border:] = 0
        f[:, :border] = 0
        f[:, -border:] = 0
    return f


@pytest.fixture
def fake_storms():
    import monte_python
    centers = [(40, 45), (40, 58), (65, 90), (90, 55), (40, 20)]
    storms, _, _ = monte_python.create_fake_storms(centers)
    return storms
