# spectroscopy-analysis

Python package for loading, analysing and plotting spectroscopic data: **Raman**, **photoluminescence** (steady-state and time-resolved), **absorption** and **XPS** depth profiles. The physical models (LOPC fit, constants) are set up for 4H-SiC.

You load measurements into a `Project`, each spectrum becomes a `Data` object, and you plot or fit it with one method call. XPS depth profiles have their own `XPS` class.

## Supported measurements

| Technique | What the package does |
|---|---|
| **Raman** (single spectra and z-maps) | Dark subtraction, spike check, peak shift to the TO mode, baseline subtraction, integration, multi-peak fit, **LOPC fit** of the LO-phonon–plasmon coupled mode (carrier density and mobility), epi/substrate peak separation along z |
| **PL** | Exciton peak fit and integration, quantum-yield metrics (QY, k<sub>r</sub>, k<sub>nr</sub>), before/after comparison |
| **TRPL** (time-resolved PL) | Lifetime fits (`'exp'`, `'two-exp'`, `'line'`, `'conv'` with IRF convolution), merging short and long gate measurements, lifetime plots and histograms, summaries with uncertainties |
| **Absorption** | Reads Jasco and Avantes spectrometer exports |
| **XPS** (VAMAS `.vms` files) | Shirley or Tougaard background, binding-energy correction on adventitious carbon (C 1s), peak fits per block, **atomic % vs depth** |
| **Any text/CSV file** | Structure detected automatically (delimiter, decimal comma, header, layout, units) |

## Installation

Requires **Python 3.12 or newer**.

```bash
git clone https://github.com/kerht/spectroscopy-analysis.git
cd spectroscopy-analysis
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[test]"
```

`-e` installs the package in *editable* mode: changes to the code in `src/` are used immediately, and `import spectroscopy_analysis` works from any folder. `[test]` also installs pytest. Then check the installation:

```bash
pytest
python examples/quickstart.py
```

## Repository contents

| Path | Content |
|---|---|
| `src/spectroscopy_analysis/` | the package (see *Package layout* below) |
| `examples/quickstart.py` | a tour of every technique on the example data, in `# %%` cells |
| `examples/data/` | example measurements: Raman, PL, TRPL, absorption, XPS, a lab register |
| `tests/` | pytest tests on the example data and on synthetic spectra |
| `.github/workflows/tests.yml` | runs the tests and the example script on GitHub at every push |
| `pyproject.toml` | package metadata and dependencies |

## Quick start

```python
import matplotlib.pyplot as plt
import spectroscopy_analysis as sa

sa.apply_style()                                   # optional: the package's plot look

# --- Raman: any text file, then a peak fit ---
proj = sa.Project()
si = proj.open_file('examples/data/raman/Si_reference.txt', meas_type='Raman', ecc=633)
r = si.fit_peaks(fit_lims=[(505, 535)], shapes=[sa.voigt], baseline=True)
print(r.params['peak1_center'].value)

# --- Raman / PL measured with the lab register ---
proj.open_data('231212', 30)                       # measurement 30 of folder 231212
proj.open_folder('231212', meas_type='Raman')      # every measurement of the folder
s = proj.data['23121230']                          # spectra are stored by file name
s.fit_lopc()                                       # LOPC fit

# --- TRPL ---
trpl = sa.Project()
trpl.open_pl('examples/data/trpl', 'W1_P#1_SEQ_100_160_CCD50_G1_1')
trpl.plot_pl(fit_method='exp')                     # decay + lifetime
groups = sa.Project().open_folder('examples/data/trpl', meas_type='trpl')   # {wafer: {'short', 'long', 'other'}}

# --- Absorption ---
a = sa.Project().open_abs('examples/data/absorption/avantes', 'W1_P#4', instrument='avantes')

# --- XPS ---
xps = sa.XPS()
xps.load_data('W1P1.vms', path='examples/data/xps')
xps.plot_xps(bg_type='tougaard', shift=True)       # background + C 1s correction
at_percent, depths = xps.fit_blocks()              # atomic % vs depth
plt.show()
```

`examples/quickstart.py` runs all of this on the example data.

## Package layout

| Module | Content |
|---|---|
| `constants` | 4H-SiC constants of the LOPC model; `INT_LIMS`, the default PL/TRPL integration window |
| `units` | `cmtoev`, `cmtonm`, `evtocm`, `evtonm`, `nmtoev`, `nmtocm` |
| `models` | lmfit model classes: `gauss`, `lorentz`, `voigt`, `baseline_line`, `baseline_poly`, `constant`, `exp`; `as_model` |
| `style` | colour/marker cycles and `apply_style()` |
| `data` | `Data`: one spectrum, with the methods of every technique |
| `project` | `Project`: a collection of measurements, with the loaders |
| `io` | loader methods of `Project` (`open_data`, `open_folder`, `open_pl`, `open_abs`, `open_file`, …) |
| `raman`, `pl`, `trpl`, `fitting`, `plotting` | the methods of `Data` and `Project`, grouped by technique |
| `xps` | `XPS` |

`Data` and `Project` are single classes: the technique modules only group their methods in the code. Everything is available from the top level (`sa.Data`, `sa.Project`, `sa.voigt`, `sa.nmtoev`, …).

### lmfit models

`gauss`, `lorentz`, `voigt`, `baseline_line`, `baseline_poly`, `constant` and `exp` are model **classes**: call them to get a new, independent model.

```python
model = lmfit.Model(sa.lopc) + sa.baseline_line()     # note the ()
model = sa.gauss(prefix='p1_') + sa.gauss(prefix='p2_')
```

Arguments such as `bl` (`integrate`, `fit_exciton`, `plot_pl`, …), `shape` (`fit_si`), `shapes` and `baseline` (`fit_peaks`) accept either a class (`bl=sa.baseline_poly`) or a model you created (`bl=lmfit.models.QuadraticModel()`); a model you pass is not modified.

### Integration window

PL/TRPL spectra are integrated over **325–480 nm** unless a method is given `int_lims`. To change the default everywhere:

```python
sa.constants.INT_LIMS = (340, 460)
```

## Opening any text or CSV file

Files that don't follow the lab conventions (other instruments, exports from colleagues, Excel CSVs) can be opened with `Project.open_file`. The structure is detected by `Project.read_file`: delimiter, decimal comma, header and footer lines, encoding, and layout (`y`, `xy`, `multi` columns, `map` with x in the first row, `transposed`).

```python
sa.Project.read_file('export.csv')               # only print the detected structure

proj = sa.Project()
d = proj.open_file('export.csv', meas_type='Raman', ecc=633)
d.plot(norm='norm_max')                          # all Data methods work

for d in proj.open_file('zscan.csv', meas_type='Raman'):   # several spectra -> list of Data
    d.plot(ax)

proj.open_file('odd.txt', header_lines=3, decimal=',', x_unit='nm')   # correct the detection
```

Units and axis labels are read from the file (column names such as `Raman shift (cm-1)` or `Wavelength [nm]`, a units row, a `# Columns: ...` comment, or `XUNITS`/`YUNITS` header keys) and used by `Data.plot`. Points keep the order of the file (`sort=True` sorts by increasing x), missing values are dropped and no dark subtraction is applied.

## Data folder and file conventions

The lab loaders rely on these names.

**Raman / PL (LabRAM)** — `open_data`, `open_folder(meas_type='Raman')`
- Folder name starts with the date, optionally followed by a space and a name: `231212` or `231212 user`.
- The folder contains the measurement register `reg_mis_<date>.xlsx` with a column `N` (measurement number). See `examples/data/261001/`.
- Each measurement is `<date><NN>.txt` (e.g. `23121230.txt`), next to the original file from the instrument: `.l6s` (spectrum), `.l6m` (map) or `.0`. Only the existence of the original file is checked, to choose between spectrum and map.

**TRPL** — `open_pl`, `open_folder(meas_type='trpl')`
- File name: `wafer_sample_meastype_tstart_tstop_CCDx_Gx_prognum.txt`, e.g. `W1_P#2_SEQ_117_127_CCD50_G1_1.txt`. The number after `#` in the sample name is the position used by `plot_tau`.
- Export as *Multiple column (header)* with *Comma* delimiter.
- `G1` = short gate, `G3` = long gate.

**Absorption** — `open_abs`, `open_folder(meas_type='absorption', instrument=...)`
- `.txt` (any case) exports from **Jasco** (tab-separated, Windows-1252, decimal comma) or **Avantes** (`;`-separated; header length detected). Choose with `instrument='jasco'` (default) or `'avantes'`.

**XPS** — `XPS.load_data`
- VAMAS `.vms` files. With `save_metadata=True`, `load_data` also writes `<file>_metadata.csv`.

## Physical parameters

The LOPC model uses 4H-SiC constants from *J. Appl. Phys.* **90**, 10 (2001): ω<sub>TO</sub> = 774.85 cm⁻¹, ω<sub>LO</sub> = 962.05 cm⁻¹, Faust–Henry coefficient C = 0.35, Γ = 4.94 cm⁻¹, ε<sub>∞</sub> = 6.78, m* = 0.29 m<sub>e</sub>. An alternative set from *Crystals* **9**, 428 (2019) is in `constants.py`, commented out.

## Plot style

Importing the package does **not** change matplotlib's settings. `sa.apply_style()` applies the package's look: serif fonts, ticks on all four sides, 9.6 × 7.2 in figures, 300 dpi export and large labels (`'poster'`, the default), or the same with normal font sizes (`'paper'`). For LaTeX fonts, uncomment the `text.usetex` lines in `style.py` (slower, needs a LaTeX installation).

## Origin

This repository comes from a private one that I developed and used for the data analysis of my Ph.D. It is a repackaging of that code: all references to samples and other project-specific information have been removed, and the example data have been anonymised.

## License

MIT. See [LICENSE](LICENSE).
