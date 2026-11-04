"""Shared test setup: non-interactive matplotlib, path of the example data."""
import os

import matplotlib
import pytest

matplotlib.use('Agg')

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'data')


@pytest.fixture
def examples():
    """Absolute path of examples/data."""
    return os.path.abspath(EXAMPLES)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close('all')
