"""
ANDESITE Package Setup Script
============================

This setup script builds the andesite package for geostatistical analysis and mining applications.
The package contains modules for estimation, variography, compositing, and data analysis.

Build Process:
1. Imports version information from utils.version module using git describe
2. Reads package metadata from __version__.py
3. Configures package structure to map current directory structure to andesite namespace
4. Includes binary executables and data files from utils/ directory
5. Creates wheel distribution with all submodules properly namespaced

Usage:
    python setup.py bdist_wheel    # Build wheel package
    python setup.py sdist          # Build source distribution
"""

from setuptools import setup
from utils import version

# Load version metadata from __version__.py
meta = {}
with open('__version__.py') as f:
    exec(f.read(), meta)


# Read long description from README for PyPI display
with open('README.md', 'r') as f:
    long_description = f.read()

# Package configuration
setup(
    # Basic package information
    name='andesite',
    version=version.get_git_version(),  # Dynamic version from git tags
    author='ANDESITE SpA',
    author_email='dev@andesite.cl',
    description='Geostatistical analysis and mining estimation software',
    keywords="geostatistics mining estimation kriging variography compositing resource-modeling",
    url="http://www.andesite.cl/",

    # Long description for PyPI
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Package structure configuration
    # Maps current directory (.) to andesite namespace in the wheel
    package_dir={'andesite': '.'},

    # Explicitly list all subpackages to include in distribution
    packages=[
        'andesite',                    # Main package
        'andesite.analisis',          # Cross validation, histograms, slicer views
        'andesite.clasification',     # Pass classification algorithms
        'andesite.composite',         # Drill hole compositing operations
        'andesite.datafiles',         # Data file handling utilities
        'andesite.estimations',       # Kriging and estimation algorithms
        'andesite.tests',             # Unit tests
        'andesite.utils',             # Common utilities and helper functions
        'andesite.variography'        # Variogram modeling and analysis
    ],

    # Include additional data files in the package
    include_package_data=True,
    package_data={
        'andesite': [
            "utils/RELEASE-VERSION",   # Version file for git-less installations
            "utils/bin/*"              # Binary executables (kt3d, gamv, etc.)
        ]
    },

    # Python version requirement
    python_requires='>=3.9',

    # Package dependencies
    install_requires=[
        "Cython"  # Required for numerical computations
    ],
)
