"""Data: float arrays, spike removal, normalisation, integration window."""
import os

import numpy as np
import pandas as pd
import pytest

import spectroscopy_analysis as sa


def _peak(x, center=520.0, height=1000.0, sigma=2.0):
    return height * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def test_x_y_float_arrays_and_caller_untouched():
    x = pd.Series(np.linspace(400, 600, 201), index=[str(i) for i in range(201)])
    y = _peak(np.linspace(400, 600, 201))
    y[50] = -1e6                                           # negative spike
    y_in = y.copy()
    d = sa.Data('t', x, y, info=None, data={})
    assert isinstance(d.x, np.ndarray) and isinstance(d.y, np.ndarray)
    assert np.array_equal(y, y_in)                          # caller's array not modified
    assert d.y.min() > -1e6                                 # spike removed in the Data


@pytest.mark.parametrize('y', [np.array([]), np.full(10, np.nan)])
def test_check_spike_empty_or_nan(y):
    d = sa.Data('t', np.arange(y.size, dtype=float), y, info=None, data={})
    assert not d.check_spike()


def test_plot_norm_window_list():
    x = np.linspace(400, 600, 401)
    d = sa.Data('t', x, _peak(x), info=None, data={})
    fig, ax = d.plot(norm=[510, 530], d_sub=False)
    assert np.isclose(ax.lines[-1].get_ydata().max(), 1.0)


def test_int_lims_default_and_runtime_change():
    x = np.linspace(300, 500, 401)
    d = sa.Data('t', x, _peak(x, center=390, sigma=10), info=None, data={})
    d.integrate(fit=False)
    assert d.int_lims == (325, 480)
    old = sa.constants.INT_LIMS
    try:
        sa.constants.INT_LIMS = (340, 460)
        d.integrate(fit=False, force=True)
        assert d.int_lims == (340, 460)
    finally:
        sa.constants.INT_LIMS = old


def test_plot_pl_all_restores_x(examples):
    p = sa.Project()
    p.open_pl(os.path.join(examples, 'trpl'), 'W1_P#2_SEQ_117_127_CCD50_G1_1')
    x0 = [d.x.copy() for d in p.data.values()]
    p.plot_pl_all()
    assert all(np.array_equal(a, d.x) for a, d in zip(x0, p.data.values()))
