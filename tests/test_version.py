"""The installed version (pyproject.toml) is the one in the package."""
from importlib.metadata import version

import spectroscopy_analysis as sa


def test_version_single_source():
    assert version('spectroscopy-analysis') == sa.__version__
