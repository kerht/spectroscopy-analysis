"""Steady-state PL: emission integral with exciton baseline, quantum-yield metrics."""

import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from scipy.integrate import simpson, trapezoid

from . import constants
from .units import evtonm, nmtoev
from .models import as_model, baseline_line, gauss


class DataPLMixin:
    """pl methods of :class:`Data` (mixed into it, not used alone)."""

    def integrate(self, fit=True, int_lims=None, plot_fit=False, bl=baseline_line,
                  force=False):
        """Integrated emission of a PL/TRPL spectrum.

        The spectrum is cut to ``int_lims`` (nm); if ``fit`` is True the baseline
        is removed with :meth:`fit_exciton` first. The integral (trapezoid rule)
        is stored in ``self.I`` and the limits used in ``self.int_lims``.

        The result is cached: once integrated, later calls return the stored
        ``self.I``, even with different limits, unless ``force=True``. The
        cache is what keeps the normalized integrals set by
        :func:`merge_short_long`, so do not force a merged series.

        Parameters
        ----------
        int_lims : tuple, optional
            Integration window (nm). Default: the module constant
            ``INT_LIMS`` (325, 480), shared by the whole library.
        force : bool
            Recompute the integral even if it was already computed.

        Returns
        -------
        float
        """
        if int_lims is None:
            int_lims = constants.INT_LIMS

        # self.y=self.y_raw
        roi = np.where((self.x >= int_lims[0]) & (self.x <= int_lims[1]))
        x_cut = np.array(self.x, dtype=float)
        y_cut = np.array(self.y, dtype=float)
        x_cut = x_cut[roi]
        y_cut = y_cut[roi]
        if fit is True:
            self.y_int = self.fit_exciton(
                fit_lims=int_lims, plot=plot_fit, bl=bl)
        else:
            self.y_int = y_cut

        # y_smoothed = moving_average(self.y_int, w=10)
        # edge=int(len(y_smoothed)-len(self.y_int))
        # x_smoothed = x_cut[-edge:edge]
        if self.int_status == False or force:
            # integrate the emission over int_lims (first time, or forced)
            I = trapezoid(self.y_int, x=x_cut)
            # I_smoothed = trapezoid(y_smoothed, x=x_smoothed)
            self.I = I
            self.int_status = True
            self.int_lims = tuple(int_lims)

        # self.I_error = abs(I-I_smoothed)
        # error = np.std(y_cut)
        # self.I_error = error
        # self.I= ufloat(I, error)

        # elif self.int_status == True:
            # print(f'Measure {self.name} already integrated')

        return self.I

    def fit_exciton(self, fit_lims=None, report=False, plot=False, bl=baseline_line):
        """Remove the background under the PL emission with a peak + baseline fit.

        Works in eV: the spectrum is converted from nm, a Gaussian plus the
        baseline model ``bl`` is fitted within ``fit_lims`` (nm) and the fitted
        baseline is subtracted. If there is no clear signal, or the fit looks
        unreliable (baseline subtraction too weak or too strong, weak or poorly
        determined peak, peak at the window edge), only the mean is subtracted
        instead.

        Parameters
        ----------
        fit_lims : [min, max], optional
            Window in nm. Default: ``INT_LIMS`` (325, 480).
        report, plot : bool
            Print the fit report / plot fit and residuals.
        bl : lmfit model class or model
            Baseline model (default: linear). A class (``baseline_line``,
            ``baseline_poly``, ...) or a model created by the caller.

        Returns
        -------
        ndarray
            Baseline-subtracted intensity, converted back to per-nm, for the
            points of ``self.x`` inside the window.
        """

        if fit_lims is None:
            fit_lims = constants.INT_LIMS
        x, y = np.array(self.x, dtype=float), np.array(self.y, dtype=float)
        x, y = nmtoev(x, y)
        roi = np.where((x >= (1239.8/fit_lims[1])) & (x <= (1239.8/fit_lims[0])))
        x_cut = x[roi]
        y_cut = y[roi]

        # ---- early “no signal” check
        noise = np.std(y_cut)
        signal_amp = np.ptp(y_cut)
        if noise == 0 or signal_amp < 3 * noise:
            data_bl_sub = y_cut - np.mean(y_cut)
            if plot:
                fig, ax = plt.subplots()
                ax.plot(x_cut, y_cut, 'o')
                ax.axhline(np.mean(y_cut), color='r', linestyle=':')
            x_cut_nm, data_bl_sub_nm = evtonm(x_cut, data_bl_sub)
            return data_bl_sub_nm

        peak = gauss()
        bl = as_model(bl)       # class (default) or model given by the caller
        signal = peak + bl
        params = signal.make_params()
        params.update(bl.guess(y_cut, x=x_cut))
        params.update(peak.guess(y_cut, x=x_cut))

        signal_result = signal.fit(y_cut, params, x=x_cut)
        fitted_baseline = bl.eval(signal_result.params, x=x_cut)
        data_bl_sub = y_cut - fitted_baseline
        status = True

        # Existing stats
        orig_amp = np.ptp(y_cut)
        res_amp  = np.ptp(data_bl_sub)
        orig_std = np.std(y_cut)
        res_std  = np.std(data_bl_sub)
        orig_mean = np.mean(y_cut)
        res_mean  = np.mean(data_bl_sub)
        bl_mean   = np.mean(fitted_baseline)
        bl_std    = np.std(fitted_baseline)

        # Under-subtraction (your existing logic or modified)
        under = abs(res_mean / (np.max(data_bl_sub) + 1e-30)) > 0.6

        # Over-subtraction: baseline too high or residuals explode
        over  = (
            res_mean < -4 * max(orig_std, bl_std, 1e-30)
            or res_amp > 3 * orig_amp + 1e-30
            or bl_mean > orig_mean + 4 * max(orig_std, bl_std, 1e-30)
        )

        if under or over:
            data_bl_sub = y_cut - np.mean(y_cut)
            status = False

        # ---- model-based sanity checks
        amp = signal_result.params['amplitude'].value
        amp_err = signal_result.params['amplitude'].stderr or 0
        baseline_level = np.mean(fitted_baseline)
        center = signal_result.params['center'].value

        if (
            abs(amp) < 0.05 * (abs(baseline_level) + noise)
            or amp_err > 0.5 * abs(amp)
            or abs(center - x_cut.min()) < 0.02 * (x_cut.max() - x_cut.min())
            or abs(center - x_cut.max()) < 0.02 * (x_cut.max() - x_cut.min())
        ):
            data_bl_sub = y_cut - np.mean(y_cut)
            status = False

        if plot:
            fig, ax = plt.subplots(2, 1)
            signal_result.plot_fit(ax[0])
            ax[0].plot(x_cut, fitted_baseline, linestyle=':')
            ax[0].plot(x_cut, data_bl_sub, linestyle=':')
            ax[0].axhline(0, color='black', linestyle=':')
            signal_result.plot_residuals(ax[1])
            if not status:
                fig.suptitle('Background subtraction aborted', color='red')

        if report:
            print(signal_result.fit_report())

        x_cut, data_bl_sub = evtonm(x_cut, data_bl_sub)
        return data_bl_sub

    def get_corrected_area(self, x_range):
        """Area of the spectrum in an energy window after a linear baseline.

        The spectrum (nm) is converted to eV, cut to ``x_range`` (eV), a straight
        line through the two end points is subtracted, and the remainder is
        integrated (Simpson). The absolute value is returned because the eV axis
        is in decreasing order.

        Returns
        -------
        float
            Area, or 0.0 if fewer than 2 points are in the window.
        """
        # convert the spectrum from nm to eV
        x_ev, y_ev = nmtoev(self.x, self.y)

        # points inside the energy window
        idx = np.where((x_ev >= x_range[0]) & (x_ev <= x_range[-1]))
        x_sub = x_ev[idx]
        y_sub = y_ev[idx]

        if len(x_sub) < 2:
            return 0.0

        # linear baseline through the end points of the window
        x1, x2 = x_sub[0], x_sub[-1]
        y1, y2 = y_sub[0], y_sub[-1]

        m = (y2 - y1) / (x2 - x1)
        q = y1 - m * x1

        baseline = m * x_sub + q
        y_corr = y_sub - baseline

        # Simpson integration
        return np.abs(simpson(y_corr, x=x_sub))  # abs: the eV axis is decreasing


class ProjectPLMixin:
    """pl methods of :class:`Project` (mixed into it, not used alone)."""

    def calculate_qy_metrics(self, tau_df, mask_Raman, mask_PL, eta):
            """PL quantum yield and recombination rates for every spectrum of the project.

            For each Data (a PL spectrum in nm, label/SAMPLE = "wafer sample"):
            1. The PL and Raman areas are computed in eV after a linear baseline
               (:meth:`Data.get_corrected_area`) over ``mask_PL`` and ``mask_Raman``.
            2. QY = (A_PL / A_Raman) / eta.
            3. With the lifetime tau (column 'tau2' of ``tau_df``):
               k_r = QY / tau and k_nr = 1/tau - k_r.

            Parameters
            ----------
            tau_df : pandas.DataFrame
                Lifetimes indexed by (wafer, samp); values formatted as 'value±err'
                in ns.
            mask_Raman, mask_PL : array_like
                Energy windows (eV).
            eta : float
                Normalization factor (see :meth:`compare_pre_post`).

            Returns
            -------
            dict
                SAMPLE -> {'QY', 'kr', 'knr', 'tau'} (tau in s, rates in 1/s).
            """
            results = {}
            for k, d in self.data.items():
            #     label = d.label if d.label else d.info.SAMPLE
            #     if not label or ' ' not in label:
            #         continue

                # wafer, samp = label.split(' ')[0], label.split(' ')[1]
                wafer, samp = d.info.SAMPLE.split(' ')[0], d.info.SAMPLE.split(' ')[1]

                # 1. Raman area without baseline correction (overwritten below)
                x_ev, y_ev = nmtoev(d.x, d.y)
                idx_ram = np.where((x_ev >= mask_Raman[0]) & (x_ev <= mask_Raman[-1]))
                area_raman = np.abs(simpson(y_ev[idx_ram], x=x_ev[idx_ram]))

                # 2. PL and Raman areas with linear baseline (Data.get_corrected_area)
                area_pl = d.get_corrected_area(mask_PL)
                area_raman = d.get_corrected_area(mask_Raman)

                # 3. Quantum yield
                qy = (area_pl / area_raman) / eta

                # 4. Lifetime from tau_df (strings formatted 'value ± error', in ns)
                try:
                    # look up (wafer, samp) in the MultiIndex; 'stage' is handled by
                    # compare_pre_post, so tau_df is assumed to be filtered already
                    # tau_str = str(tau_df.xs((wafer, samp), level=['wafer', 'samp'])['tau_avg'].values[0])
                    tau_str = str(tau_df.xs((wafer, samp), level=['wafer', 'samp'])['tau2'].values[0])
                    tau_val = float(tau_str.split('±')[0]) * 1e-9 # ns -> s

                    kr = qy / tau_val
                    knr = (1.0 / tau_val) - kr

                    # results[label] = {'QY': qy, 'kr': kr, 'knr': knr, 'tau': tau_val}
                    results[d.info.SAMPLE] = {'QY': qy, 'kr': kr, 'knr': knr, 'tau': tau_val}
                except Exception as e:
                    # print(f"Errore nel calcolo parametri per {label}: {e}")
                    print(f"Errore nel calcolo parametri per {d.info.SAMPLE}: {e}")

            return results

    @staticmethod
    def compare_pre_post(list_pre, list_post, tau_df,
                         mask_Raman=np.array([3.71, 3.73]),
                         mask_PL=np.array([2.85, 3.55]),
                         B=1.5e-12, delta_n=1e17):
        """Compare QY, k_r and k_nr before and after a treatment.

        Runs :meth:`calculate_qy_metrics` on the 'Pre' and 'Post' projects, using
        the rows of ``tau_df`` with the matching 'stage', and prints a table with
        the Post/Pre ratios for the samples present in both lists.

        Parameters
        ----------
        list_pre, list_post : list of Project
        tau_df : pandas.DataFrame
            Lifetimes with a 'stage' level/column ('Pre' or 'Post').
        mask_Raman, mask_PL : array_like
            Energy windows (eV).
        B : float
            Radiative recombination coefficient (cm3/s).
        delta_n : float
            Excess carrier density (cm-3). eta = B * delta_n.

        Returns
        -------
        dict
            {'pre': results, 'post': results}.
        """
        eta = B * delta_n
        all_pre_results = {}
        all_post_results = {}

        for p in list_pre:
            # filter tau_df on 'stage' (index level or column)
            if 'stage' in tau_df.index.names:
                df_stage = tau_df.xs('Pre', level='stage', drop_level=False)
            else:
                df_stage = tau_df[tau_df['stage'] == 'Pre']
            all_pre_results.update(p.calculate_qy_metrics(df_stage, mask_Raman, mask_PL, eta))

        # same for the Post projects
        for p in list_post:
            if 'stage' in tau_df.index.names:
                df_stage = tau_df.xs('Post', level='stage', drop_level=False)
            else:
                df_stage = tau_df[tau_df['stage'] == 'Post']
            all_post_results.update(p.calculate_qy_metrics(df_stage, mask_Raman, mask_PL, eta))

        # print the comparison table
        print(f"\n{'Sample':<10} | {'Metric':<6} | {'Pre':<8} | {'Post':<8} | {'Ratio'}")
        print("-" * 50)

        common_samples = set(all_pre_results.keys()) & set(all_post_results.keys())
        for s in natsorted(common_samples):
            pre, post = all_pre_results[s], all_post_results[s]

            print(f"\n{s}")
            for m in ['QY', 'kr', 'knr']:
                v_pre = pre[m]
                v_post = post[m]
                ratio = v_post / v_pre
                fmt = ".2%" if m == 'QY' else ".2e"
                print(f"{'':<10} | {m:<6} | {v_pre:{fmt}} | {v_post:{fmt}} | {ratio:.2f}x")

        return {'pre':all_pre_results, 'post': all_post_results}
