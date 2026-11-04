"""Material constants (4H-SiC, used by the LOPC fit) and the default PL/TRPL integration window."""

from scipy.constants import m_e


# ---- 4H-SiC constants for the LOPC fit (see lopc() and Data.fit_lopc).
# Two literature sets are kept; uncomment the one to use.
# Set in use: J. Appl. Phys., Vol. 90, No. 10 (2001)

omega_TO = 774.85     # TO mode frequency (cm^-1)
omega_LO = 962.05     # LO mode frequency (cm^-1)
# omega_TO = 775.80       # TO mode frequency (cm^-1) , from my data
# omega_LO = 963.05       # LO mode frequency (cm^-1), , from my data
C = 0.35                # Faust-Henry coefficient
GAMMA = 4.94            # Phonon damping constant (cm^-1)
eps_inf = 6.78          # high freq dielectric constant
m_eff = 0.29*m_e        # effective mass, longitudinal
eps_0 = 8.854e-12       # Vacuum permittivity [F/m]

# Alternative set: Crystals 2019, 9, 428; doi:10.3390/cryst9080428 (2019)

# omega_TO = 777     # TO mode frequency (cm^-1)
# omega_LO = 964.2     # LO mode frequency (cm^-1)
# # omega_TO = 775.80       # TO mode frequency (cm^-1) , from my data
# # omega_LO = 963.05       # LO mode frequency (cm^-1), , from my data
# C = 0.43                # Faust-Henry coefficient
# GAMMA = 4.94            # Phonon damping constant (cm^-1)
# eps_inf = 6.78          # high freq dielectric constant
# m_eff = 0.48*m_e      # effective mass, longitudinal
# eps_0 = 8.854e-12       # Vacuum permittivity [F/m]

# ---- Default integration window for PL / TRPL spectra (nm)
# Used by every method that integrates an emission spectrum when no window is
# given: Data.integrate, Data.fit_exciton, Project.plot_pl, plot_pl_mean,
# rescale_factor, merge_short_long. Change it here (or in a script with
# ``spectroscopy_analysis.constants.INT_LIMS = (a, b)``) to change it everywhere.
INT_LIMS = (325, 480)
