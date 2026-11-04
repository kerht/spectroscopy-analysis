"""Unit conversions: round trips and known values."""
import numpy as np

import spectroscopy_analysis as sa


def test_cm_nm_round_trip():
    shift = np.array([0.0, 520.7, 776.0, 964.0])
    nm = sa.cmtonm(shift, ecc=633)
    assert np.allclose(sa.nmtocm(nm, ecc=633), shift)
    assert np.isclose(nm[0], 633)                      # zero shift = laser line


def test_cm_ev_round_trip_preserves_area():
    shift = np.linspace(500, 540, 401)
    y = np.exp(-(shift - 520) ** 2 / 2)
    ev, y_ev = sa.cmtoev(shift, y, ecc=633)
    back, y_back = sa.evtocm(ev, y_ev, ecc=633)
    assert np.allclose(back, shift)
    assert np.allclose(y_back, y)
    area_cm = np.trapezoid(y, shift) if hasattr(np, 'trapezoid') else np.trapz(y, shift)
    area_ev = abs(np.trapezoid(y_ev, ev) if hasattr(np, 'trapezoid') else np.trapz(y_ev, ev))
    assert np.isclose(area_cm, area_ev, rtol=1e-3)


def test_nm_ev():
    ev, _ = sa.nmtoev(np.array([1239.8]), np.array([1.0]))
    assert np.isclose(ev[0], 1.0)
    nm, _ = sa.evtonm(np.array([2.0]), np.array([1.0]))
    assert np.isclose(nm[0], 619.9)
