"""Quick tour of spectroscopy_analysis on the example data in examples/data.

Run it from the repository root:

    python examples/quickstart.py

or open it in VS Code / Spyder and run the ``# %%`` cells one at a time.
"""
# %% Setup
import os

import matplotlib.pyplot as plt

import spectroscopy_analysis as sa

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
sa.apply_style('paper')            # or 'poster', or leave matplotlib's defaults

# %% Raman: Si reference, peak fit with a linear baseline
proj = sa.Project()
si = proj.open_file(os.path.join(DATA, 'raman', 'Si_reference.txt'),
                    meas_type='Raman', ecc=633, verbose=False)
fig, ax = plt.subplots()
r = si.fit_peaks(fit_lims=[(505, 535)], shapes=[sa.voigt], baseline=True, ax=ax)
print(f"Si peak: {r.params['peak1_center'].value:.2f} cm-1, "
      f"FWHM {r.params['peak1_fwhm'].value:.2f} cm-1")

# %% Raman: z-scan (one spectrum per depth)
zmap = sa.Project()
spectra = zmap.open_map(os.path.join(DATA, 'raman', 'z_scan.txt'), info=None, data=zmap.data)
fig, ax = plt.subplots()
for d in spectra[::10]:
    d.plot(ax, d_sub=False)

# %% PL map loaded through the lab register (folder named by date)
# open_data reads reg_mis_<date>.xlsx and needs the original LabSpec file
# (.l6s, .l6m or .0) next to the text export: only its existence is checked.
os.chdir(DATA)
pl = sa.Project()
pl_map = pl.open_data('261001', 7)
fig, ax = plt.subplots()
for d in pl_map:
    d.plot(ax, xscale='nm', norm='norm_max', d_sub=False)

# %% TRPL: one series, decay and lifetime
trpl = sa.Project()
trpl.open_pl(os.path.join(DATA, 'trpl'), 'W1_P#1_SEQ_100_160_CCD50_G1_1')
fig, ax = plt.subplots()
trpl.plot_pl(ax, fit_method='exp', norm=True)

# %% TRPL: short (G1) + long (G3) series of the same sample, merged
short, long_ = sa.Project(), sa.Project()
short.open_pl(os.path.join(DATA, 'trpl'), 'W1_P#2_SEQ_117_127_CCD50_G1_1')
long_.open_pl(os.path.join(DATA, 'trpl'), 'W1_P#2_SEQ_121_226_CCD50_G3_1')
merged = sa.merge_short_long(short, long_)
fig, ax = plt.subplots()
merged.plot_pl(ax, norm=False, t_int=[1, 2], g_width=1, fit_method='exp')

# %% Absorption: Jasco and Avantes exports
fig, ax = plt.subplots()
sa.Project().open_abs(os.path.join(DATA, 'absorption', 'jasco'), 'W1_P#3').plot(ax, xscale='eV')
sa.Project().open_abs(os.path.join(DATA, 'absorption', 'avantes'), 'W1_P#4',
                      instrument='avantes').plot(ax, xscale='eV')

# %% XPS depth profile
xps = sa.XPS()
xps.load_data('W1P1.vms', path=os.path.join(DATA, 'xps'))
xps.plot_xps(bg_type='tougaard', shift=True)
at_percent, depths = xps.fit_blocks(thickness_per_block=2.5)

plt.show()
