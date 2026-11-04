"""XPS depth profiles from VAMAS (.vms) files."""

import lmfit
import lmfitxps.backgrounds   # the submodule must be imported explicitly: lmfitxps/__init__.py is empty
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import vamas as v
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter


# ---- XPS: reference binding-energy windows (eV) per element/chemical state
el_BE = {
    # 'element/chemical state/orbital': [min_val,max_val]
    'AdC 1s': [284.8, 284.8],
    'C 1s': [280.7, 283.0],
    'N-Si 1s': [397.0, 398.0],
    'N-C 1s': [400.0, 402.0],
    'O-Si-C 1s': [531.8, 532.6],
    'O-Si 1s': [533.0, 534.0],
    'Si-O 2p': [102.1, 103.5],
    'Si-C 2p': [100.0, 101.1]
}

# adventitious carbon C 1s, used as binding-energy reference
el_BE['AdC'] = 284.8


class XPS:
    """XPS depth profile from a VAMAS (.vms) file.

    Typical use::

        s = XPS()
        s.load_data('sample.vms', path='240718')
        s.plot_xps(bg_type='tougaard')          # computes the backgrounds
        at_percent, depths = s.fit_blocks(thickness_per_block=2.5)

    Attributes
    ----------
    data : vamas.Vamas
        The loaded file (one block per element and sputter step).
    info : dict
        General header info.
    blocks : pandas.DataFrame
        One row per block (species, units, number of points, ...).
    bg : dict
        Cached backgrounds, keyed '<element>_<block index>'.
    results : dict
        Depths, element list and atomic % from :meth:`fit_blocks`.
    fit_results : dict
        lmfit result per block.
    """
    def __init__(self):
        self.info = {}         # general (header) info
        self.blocks = pd.DataFrame()  # block-level info
        self.data = None
        self.bg = {}           # cached backgrounds
        self.sample = ''
        self.results = {}      # atomic % results
        self.fit_results = {}  # fit results per block

    # ---- 1. Load data & extract metadata
    def load_data(self, file, path=None, save_metadata=False):
        """Load XPS VAMAS file and extract header + block info into a DataFrame."""
        import datetime
        if path is None:
            path = os.getcwd()
        filepath = os.path.join(path, file)

        print(f"Loading XPS file: {filepath}")
        self.data = v.Vamas(filepath)
        self.sample = os.path.splitext(file)[0]
        header = self.data.header

        # ---- Safe header extraction ----
        def safe_get(obj, *names):
            """Try multiple possible attribute names, return None if not found."""
            for name in names:
                if hasattr(obj, name):
                    return getattr(obj, name)
            return None

        # ---- General info ----
        self.info = {
            "experiment_type": safe_get(header, "experiment_type", "experiment_mode"),
            "num_blocks": safe_get(header, "num_blocks"),
            "operator_name": safe_get(header, "operator_identifier", "operator_name"),
            "experiment_identifier": safe_get(header, "experiment_identifier", "experiment_id"),
            "instrument_model": safe_get(header, "instrument_model_identifier", "instrument_model"),
            "analysis_source_label": safe_get(header, "analysis_source_label"),
            "analysis_source_energy": safe_get(header, "analysis_source_energy", "source_energy"),
            "comment": safe_get(header, "comment", "comments"),
            "date": safe_get(header, "date"),
            "time": safe_get(header, "time"),
        }

        # ---- Optional: merge date/time into datetime ----
        if self.info.get("date") and self.info.get("time"):
            try:
                self.info["datetime"] = datetime.datetime.strptime(
                    f"{self.info['date']} {
                        self.info['time']}", "%d-%b-%Y %H:%M:%S"
                )
            except Exception:
                pass

        # ---- Per-block info ----
        block_data = []
        for i, blk in enumerate(self.data.blocks):
            block_data.append({
                "index": i,
                "species": getattr(blk, "species_label", ""),
                "block_identifier": getattr(blk, "block_identifier", f"block_{i}"),
                "analysis_mode": getattr(blk, "analysis_mode", ""),
                "scan_mode": getattr(blk, "scan_mode", ""),
                "x_label": getattr(blk, "x_label", ""),
                "x_units": getattr(blk, "x_units", ""),
                "y_label": getattr(blk.corresponding_variables[0], "y_label", ""),
                "y_units": getattr(blk.corresponding_variables[0], "y_units", ""),
                "x_start": getattr(blk, "x_start", np.nan),
                "x_step": getattr(blk, "x_step", np.nan),
                "num_points": int(getattr(blk, "analysis_width_x", 0)) + 1,
                "comment": getattr(blk, "comment", ""),
            })

        self.blocks = pd.DataFrame(block_data)

        print(f"Loaded {len(self.blocks)
                        } data blocks for sample '{self.sample}'.")
        print(self.blocks[["index", "species",
              "analysis_mode", "scan_mode"]].head())

        # ---- Optional: save metadata to CSV ----
        if save_metadata:
            meta_path = os.path.join(path, f"{self.sample}_metadata.csv")
            self.blocks.to_csv(meta_path, index=False)
            print(f"Metadata saved to: {meta_path}")

        return self.data

    # ---- 2. Plot spectra (with optional background subtraction)

    def plot_xps(self, bg_type='tougaard', cache_bg=True, bg=None, shift=True):
        """Plot all spectra for each element. Run load_data() first.
        Performs BE correction based on Adventitious Carbon (C 1s) using
        the raw peak maximum (not the Voigt fit).
        """

        if self.data is None:
            raise RuntimeError("No data loaded. Run load_data() first.")

        data = self.data
        N = range(data.header.num_blocks)

        elements = []
        ax_map = {}
        x, y = {}, {}

        # ---- Background handling ----
        if bg is None:
            if cache_bg and self.bg:
                bg = self.bg
            else:
                bg = {}
                self.bg = bg
        else:
            self.bg = bg

        # ---- Prepare subplots for each element ----
        for blk in data.blocks:
            e = blk.species_label
            if e not in elements:
                elements.append(e)
                ax_map[e] = plt.subplots()

        # ---- BE Correction (based on C 1s raw peak) ----
        shift_val = 0.0
        if shift:
            print("Performing BE correction based on Adventitious Carbon (C 1s)...")

            # Find all C 1s blocks
            c_blocks = [blk for blk in data.blocks if blk.species_label == 'C']
            if not c_blocks:
                print("Warning: No C 1s block found. Skipping BE correction.")
            else:
                # Use the *last* C block (you said it has the peak)
                blkC = c_blocks[-1]
                xC = np.array([
                    blkC.x_start + j * blkC.x_step
                    for j in range(int(blkC.analysis_width_x) + 1)
                ])
                yC = np.array(blkC.corresponding_variables[0].y_values)

                # Compute background and subtract it
                if bg_type == 'tougaard':
                    bgC = lmfitxps.backgrounds.tougaard_calculate(xC, yC, td=0)[
                        0]
                elif bg_type == 'shirley':
                    # shirley_calculate returns the background array itself
                    # (tougaard_calculate returns (array, B)): no [0] here
                    bgC = lmfitxps.backgrounds.shirley_calculate(xC, yC, maxit=100)
                else:
                    raise ValueError(
                        "bg_type must be 'tougaard' or 'shirley'.")

                yC_corr = yC - bgC

                # Find raw peak maximum position
                idx_max = np.argmax(yC_corr)
                xC_peak = xC[idx_max]
                print(f"xC start: {xC[0]:.2f}, end: {
                      xC[-1]:.2f}, step: {blkC.x_step}")
                print(f"Carbon peak maximum at {xC_peak:.2f} eV")

                # Compute shift to set C 1s = 284.8 eV
                shift_val = 284.8 - xC_peak
                print(f"→ Shifting all spectra by {shift_val:.2f} eV")

        # ---- Plot spectra for each element ----

        for e in elements:
            fig, ax = ax_map[e]
            for i in N:
                blk = data.blocks[i]
                if blk.species_label != e:
                    continue

                key = f"{e}_{i}"
                y[key] = np.array(blk.corresponding_variables[0].y_values)
                x_vals = np.array([
                    blk.x_start + j * blk.x_step
                    for j in range(int(blk.analysis_width_x) + 1)
                ])

                # Apply BE correction shift
                if shift:
                    x_vals = x_vals + shift_val

                # ---- Background calculation ----
                if bg_type is not None:
                    if key not in bg:
                        if bg_type == 'tougaard':
                            bg[key] = lmfitxps.backgrounds.tougaard_calculate(
                                x_vals, y[key], td=0)
                        elif bg_type == 'shirley':
                            # stored as (array, None), like tougaard's (array, B),
                            # so bg[key][0] is the background for both types
                            bg[key] = (lmfitxps.backgrounds.shirley_calculate(
                                x_vals, y[key], maxit=100), None)
                    ax.plot(x_vals, y[key] - bg[key][0], label=f"{e}_{i}")
                else:
                    ax.plot(x_vals, y[key], label=f"{e}_{i}")

            ax.set_xlabel(f"{blk.x_label} ({blk.x_units})")
            ax.set_ylabel("Counts/s")
            fig.suptitle(f"{e}")
            ax.legend(fontsize=6)

        self.bg = bg
        return ax_map

    # ---- 3. Fit peaks and compute atomic %
    def fit_blocks(self, plot=True, plot_fit=False, thickness_per_block=1.0, int_type='fit', smooth=True):
        """Fit every block and compute the atomic % vs depth.

        For each block: subtract the background computed by :meth:`plot_xps`,
        then take the area of a Voigt fit (``int_type='fit'``) or of the data
        (``'raw'``). Areas are divided by the relative sensitivity factors (Si, C,
        N, O) and normalized to 100 % at each depth step.
        Depth = step index * ``thickness_per_block`` (nm).

        Parameters
        ----------
        plot : bool
            Plot the atomic % profile.
        plot_fit : bool
            Plot all fits, one figure per element.
        thickness_per_block : float
            Material removed per sputter step (nm).
        int_type : {'fit', 'raw'}
        smooth : bool
            Draw a Savitzky-Golay smoothed profile.

        Returns
        -------
        at_percent : ndarray, shape (n_depths, n_elements)
            Columns in the order of ``self.results['elements']``.
        depths : ndarray
        """
        if self.data is None or not self.bg:
            raise RuntimeError(
                "Run load_data() and plot_xps() first to compute backgrounds.")

        data = self.data
        bg = self.bg
        elements = list({blk.species_label for blk in data.blocks})
        N = range(data.header.num_blocks)

        intensities = {e: [] for e in elements}
        self.fit_results = {}

        for i in N:
            blk = data.blocks[i]
            e = blk.species_label
            key = f"{e}_{i}"

            y_raw = np.array(blk.corresponding_variables[0].y_values)
            x_vals = np.array([
                blk.x_start + j * blk.x_step
                for j in range(int(blk.analysis_width_x) + 1)
            ])

            # Subtract background if available
            y_corr = y_raw - bg[key][0] if key in bg else y_raw

            # Ensure increasing x
            if x_vals[0] > x_vals[-1]:
                x_vals = x_vals[::-1]
                y_corr = y_corr[::-1]

            if int_type == 'fit':
                # fit
                # model = lmfit.models.LorentzianModel()
                model = lmfit.models.VoigtModel()
                params = model.guess(y_corr, x=x_vals)
                result = model.fit(y_corr, params, x=x_vals)
                self.fit_results[key] = result

                # Integrate
                y_fit = result.best_fit
                area = trapezoid(y_fit, x=x_vals)
                # area = simpson(y_fit, x=x_vals)
            elif int_type == 'raw':
                area = trapezoid(y_corr, x_vals)
                # area = simpson(y_corr, x=x_vals)
            else:
                raise RuntimeError("int_type must be either 'fit' or 'raw'.")
            intensities[e].append(max(area, 0))

        depths = np.arange(len(intensities[elements[0]])) * thickness_per_block

        # RSFs correction
        # relative sensitivity factors of the instrument
        RSFs = {
            'Si': 0.368,
            'C': 0.314,
            'N': 0.499,
            'O': 0.733
        }

        for e in elements:
            for j in range(len(intensities[e])):
                intensities[e][j] /= RSFs[e]

        # Normalize per depth
        intensities_arr = np.nan_to_num(
            np.array([intensities[e] for e in elements]))
        intensities_arr = intensities_arr.T
        sums = np.sum(intensities_arr, axis=1, keepdims=True)
        sums[sums == 0] = 1
        at_percent = (intensities_arr / sums) * 100

        if plot:
            fig, ax = plt.subplots()
            ax.set_title(
                f"Depth Profile - Atomic Concentrations (sample {self.sample})")
            ax.set_xlabel("Depth (nm)")
            ax.set_ylabel("Atomic %")
            ax.set_xlim(left=depths[0], right=depths[-1])
            ax.set_ylim(0, 100)
            for j, e in enumerate(elements):
                # Original data
                original_data = at_percent[:, j]

                # Apply Savitzky-Golay smoothing
                # window_length must be odd and >= polyorder + 2
                window_length = min(11, len(original_data) if len(
                    original_data) % 2 == 1 else len(original_data) - 1)
                if window_length >= 5 and smooth == True:  # Minimum reasonable window
                    smoothed_data = savgol_filter(
                        original_data, window_length=window_length, polyorder=2)
                    ax.plot(depths, smoothed_data, label=f"{
                            e} (smoothed)", linewidth=2)
                    # Apply Gaussian smoothing
                    # smoothed_data = gaussian_filter1d(at_percent[:, j], sigma=1.5)
                    # ax.plot(depths, smoothed_data, label=f"{e} (smoothed)")
                else:
                    ax.plot(depths, original_data, label=e, linewidth=2)

            ax.legend()

        self.results = {
            "depths": depths,
            "elements": elements,
            "atomic_percent": at_percent
        }

        if plot_fit:
            if int_type == 'fit':
                for e in elements:
                    fig, ax = plt.subplots()
                    fig.suptitle(f"{e} - Best Fits")
                    for i in N:
                        key = f"{e}_{i}"
                        if data.blocks[i].species_label != e:
                            continue
                        result = self.fit_results.get(key)
                        if result:
                            result.plot_fit(ax=ax)
                    ax.set_xlabel("Binding Energy (eV)")
                    ax.set_ylabel("Intensity (a.u.)")
                    for line in ax.get_lines():
                        line.set_linewidth(0.8)
                        line.set_markersize(1)
                        line.set_alpha(0.6)
            else:
                raise RuntimeError("No fit performed. Plot_fit aborted")

        return at_percent, depths
