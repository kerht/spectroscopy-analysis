"""Fits on synthetic data with known parameters (need lmfit)."""
import numpy as np

import spectroscopy_analysis as sa


def _voigt_like(x, center, height, sigma):
    return height * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))


def test_fit_peaks_single():
    x = np.linspace(480, 560, 801)
    y = _voigt_like(x, 520.7, 1000, 1.5)
    d = sa.Data('Si', x, y, info=None, data={})
    r = d.fit_peaks(fit_lims=[(510, 530)], shapes=[sa.gauss], plot=False)
    assert abs(r.params['peak1_center'].value - 520.7) < 0.01
    assert abs(r.params['peak1_sigma'].value - 1.5) < 0.01


def test_fit_peaks_with_baseline():
    x = np.linspace(480, 560, 801)
    y = _voigt_like(x, 520.7, 1000, 1.5) + 0.5 * x + 100
    d = sa.Data('Si', x, y, info=None, data={})
    r = d.fit_peaks(fit_lims=[(505, 535)], shapes=[sa.gauss], baseline=True, plot=False)
    assert abs(r.params['peak1_center'].value - 520.7) < 0.01
    assert abs(r.params['bl_slope'].value - 0.5) < 0.01


def test_models_are_independent():
    a, b = sa.gauss(), sa.gauss()
    a.set_param_hint('center', value=520, min=480, max=560)
    assert 'center' not in b.param_hints                      # no shared state
