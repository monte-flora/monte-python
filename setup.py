# Always prefer setuptools over distutils

import setuptools  # this is the "magic" import

# Do NOT import monte_python here — that triggers monte_python/__init__.py
# which imports cv2, scikit-image, xarray, etc. Those don't exist yet at
# setup time on a clean env, so pip install fails with ModuleNotFoundError.
# Version is canonical at monte_python/__init__.py (fallback '1.1.0' when
# PACKAGE_VERSION env var is unset); mirror that literal here.
_VERSION = '1.1.0'

from setuptools import setup

#from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='monte_python', 
    version=_VERSION,
    description='Methods for Object-based and Neighborhood Threat Evaluation in Python', 
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/WarnOnForecast/MontePython', 
    author='NOAA National Severe Storms Laboratory', 
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists',
        'Programming Language :: Python :: 3'
    ],
    install_requires = [
        'scikit-learn',
        'scikit-image>=0.18.1',
        'matplotlib<=3.4.3',
        'xarray >=0.21.1',
        'numba',
        'numba-kdtree',
        'scipy',
        # cv2 is imported from monte_python.object_identification.
        # Use the headless variant for server envs (no Qt5/X11).
        'opencv-python-headless',
    ],
    packages=['monte_python', 'monte_python._plot'],  # Required
    python_requires='>=3.8, <4',
    package_dir={'monte_python': 'monte_python'},
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/WarnOnForecast/MontePython/issues',
        'Source': 'https://github.com/WarnOnForecast/MontePython',
    },
)
