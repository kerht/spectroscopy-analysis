"""spectroscopy_analysis: load, process, fit and plot spectroscopic data.

Raman (single spectra and z-maps, LOPC fit), steady-state and time-resolved
photoluminescence (PL, TRPL), absorption and XPS depth profiles.

Typical use::

    import spectroscopy_analysis as sa
    sa.apply_style()                            # optional: the package's plot look

    proj = sa.Project()
    d = proj.open_file('spectrum.txt', meas_type='Raman', ecc=633)
    d.plot()
    r = d.fit_peaks(fit_lims=[(510, 530)], shapes=[sa.voigt])

Package layout
--------------
constants  material constants (4H-SiC, LOPC fit), default integration window
units      cm-1 <-> nm <-> eV conversions
models     lmfit model classes (gauss, voigt, baseline_line, ...)
style      colour/marker cycles, apply_style()
metadata   standard metadata: register columns by name, file-name patterns
data       Data: one spectrum, with the methods of every technique
project    Project: a collection of measurements, with the loaders
raman, pl, trpl, fitting, plotting, io
           the methods of Data and Project, grouped by technique
xps        XPS depth profiles from VAMAS files
"""
__version__ = '2.0.0'

from .constants import omega_TO, omega_LO, C, GAMMA, eps_inf, m_eff, eps_0
# INT_LIMS is read from spectroscopy_analysis.constants when a method runs:
# change it there, sa.constants.INT_LIMS = (a, b), not as sa.INT_LIMS.
from . import constants
from .units import cmtoev, cmtonm, evtocm, evtonm, nmtoev, nmtocm
from .models import (gauss, lorentz, voigt, baseline_line, constant, poly_degree,
                     baseline_poly, exp, as_model)
from .utils import moving_average
from . import metadata
from .style import apply_style
from .raman import lopc
from .trpl import rescale_factor, plot_tau, plot_tau_hist, merge_short_long
from .data import Data
from .project import Project
from .xps import XPS, el_BE
