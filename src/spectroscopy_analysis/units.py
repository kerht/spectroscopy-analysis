"""Unit conversions between Raman shift (cm-1), wavelength (nm) and energy (eV).

``ecc`` is the excitation (laser) wavelength in nm. Functions returning (x, y)
also rescale the intensity so that the integrated intensity is preserved.
"""

from scipy.constants import c, e, h


def cmtoev(x, y, ecc=633):
    """Convert a Raman spectrum from Raman shift (cm-1) to photon energy (eV).

    The Raman shift is turned into the absolute energy of the scattered
    photon, E = h*c*(1/lambda_exc - shift). The intensity is rescaled by the
    constant Jacobian 1/(h*c), so that the integrated intensity is preserved.

    Parameters
    ----------
    x : array_like
        Raman shift (cm-1).
    y : array_like
        Intensity per cm-1.
    ecc : float, optional
        Excitation (laser) wavelength in nm. Default 633.

    Returns
    -------
    x, y : ndarray
        Energy (eV) and intensity per eV.

    Note: ``c`` must be the scipy value in m/s; the factor 1e2 converts cm-1
    to m-1.
    """
    x = (1/ecc*1e7 - x)*h*c/e*1e2
    y = y/(1239.8*1E-7)
    return x, y


def cmtonm(x, ecc=633):
    """Convert Raman shift (cm-1) to the absolute wavelength (nm) of the
    scattered light, for an excitation wavelength ``ecc`` (nm).
    Only x is converted (no intensity rescaling).
    """
    x = 1/(1/ecc - x/1e7)
    return x


def evtocm(x, y, ecc=633):
    """Convert a spectrum from photon energy (eV) to Raman shift (cm-1).

    Inverse of :func:`cmtoev`. ``ecc`` is the excitation wavelength in nm.
    Returns the Raman shift (cm-1) and the intensity per cm-1.
    """
    x = 1/ecc*1e7 - x*e/h/c/1e2
    y = y*(1239.8*1E-7)
    return x, y


def evtonm(x, y):
    """Convert a spectrum from energy (eV) to wavelength (nm).

    Uses lambda[nm] = 1239.8 / E[eV] and rescales the intensity with the
    Jacobian |dE/dlambda| = E**2 / 1239.8, so that the integrated intensity is
    preserved. Returns (x_nm, y_per_nm).
    """
    y = y*(x**2)/1239.8
    x = 1239.8/x
    return x, y


def nmtoev(x, y):
    """Convert a spectrum from wavelength (nm) to energy (eV).

    Uses E[eV] = 1239.8 / lambda[nm] and rescales the intensity with the
    Jacobian |dlambda/dE| = lambda**2 / 1239.8, so that the integrated
    intensity is preserved. Returns (x_eV, y_per_eV).
    """
    y = y*(x**2)/1239.8
    x = 1239.8/x
    return x, y


def nmtocm(x, ecc=633):
    """Convert an absolute wavelength (nm) to Raman shift (cm-1) for an
    excitation wavelength ``ecc`` (nm). Inverse of :func:`cmtonm`.
    """
    x = 1e7*(1/ecc-1/x)
    return x
