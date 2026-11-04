"""Raman: LOPC line shape, TO-mode calibration, Si and LOPC fits, z-scans."""

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import warnings
from math import isclose
from scipy.constants import c, e
from scipy.integrate import simpson

from .constants import C, GAMMA, eps_0, eps_inf, m_eff, omega_LO, omega_TO
from .models import baseline_line, voigt


def lopc(x, S, omega_p, gamma_p, GAMMA=GAMMA, omega_TO=omega_TO, omega_LO=omega_LO):
    # auxiliary terms of the dielectric-model expression
    """Line shape of the LO-phonon-plasmon coupled (LOPC) Raman mode.

    Standard dielectric-model expression for the coupled mode in a polar
    n-type semiconductor (Faust-Henry coefficient ``C`` and high-frequency
    dielectric constant ``eps_inf`` are taken from the module constants).
    Used as an lmfit model function by :meth:`Data.fit_lopc`.

    Parameters
    ----------
    x : array_like
        Raman shift (cm-1).
    S : float
        Intensity scale factor.
    omega_p : float
        Plasma frequency (cm-1).
    gamma_p : float
        Plasmon damping constant (cm-1).
    GAMMA, omega_TO, omega_LO : float, optional
        Phonon damping, TO and LO frequencies (cm-1). Defaults: module
        constants.

    Returns
    -------
    ndarray
        Raman intensity at each x.
    """
    A = omega_p**2*gamma_p*(omega_TO**2-x**2)-x**2 * \
        GAMMA*(x**2+gamma_p**2-omega_p**2)
    B = omega_p**2*(gamma_p*(omega_LO**2-omega_TO**2)+GAMMA*(omega_p**2-2*x**2))\
        + x**2*GAMMA*(x**2+gamma_p**2)
    D = omega_p**2*gamma_p*((omega_TO**2-x**2)**2+x**2*GAMMA**2)\
        + x**2*GAMMA*(omega_LO**2-omega_TO**2)*(x**2+gamma_p**2)
    N = gamma_p*omega_TO**2*x-gamma_p*x**3-GAMMA*x**3
    M = omega_TO**2*x**2-x**4+gamma_p*GAMMA*x**2
    L = gamma_p*omega_LO**2*x+GAMMA*omega_p**2*x-GAMMA*x**3-gamma_p*x**3
    P = gamma_p*x**2*GAMMA+omega_LO**2*x**2 - \
        x**4-omega_p**2*omega_TO**2+omega_p**2*x**2
    # print(f'\n\nA={A[0]}\nB={B[0]}\nD={D[0]}\nN={N[0]}\nM={M[0]}nL={L[0]}\nP={P[0]}\n\n')

    # fitting profile
    I = -(S/eps_inf)*(1+(2*A*C*omega_TO**2)/D+(B*C**2*omega_TO**4)/(D*(omega_LO**2-omega_TO**2)))\
        * (N*P-M*L)/(P**2+L**2)
    return I


class DataRamanMixin:
    """raman methods of :class:`Data` (mixed into it, not used alone)."""

    def data_shift(self, omega_TO=omega_TO):
        """Calibrate the Raman shift on the 4H-SiC TO mode.

        Fits a Voigt peak between 750 and 790 cm-1 and shifts x so that the peak
        sits at ``omega_TO``. The applied shift (cm-1) is stored in
        ``self.shift``.

        Returns
        -------
        ndarray or None
            Shifted x; None if there are no points in 750-790 cm-1.
        """
        # fit TO(E2) mode (new model: the hints below stay in this fit)
        model = voigt()
        # if self.type == 'data_raw':
        #     print('\nIn "data_shift" method: Data type is "data_raw".')

        model.set_param_hint('center', value=omega_TO)
        model.set_param_hint('sigma', value=2, min=0.1)
        model.set_param_hint('amplitude', value=max(self.y), min=0)

        roi = np.where((self.x >= 750) & (self.x <= 790))
        # print(roi)
        if len(roi[0]) == 0:
            print(
                '\nIn "data_shift" method: No data in range of interest [750,790].\n')
            return
        x_cut = self.x[roi]
        y_cut = self.y[roi]

        params = model.make_params()

        self.shift_result = model.fit(y_cut, params, x=x_cut)
        # print(result.fit_report())
        # plt.plot(x_cut, y_cut, 'o')
        # plt.plot(x_cut, result.init_fit, '.', label='initial fit')
        # x_plot = np.linspace(min(x_cut), max(x_cut), 1000)
        # plt.plot(x_plot, result.eval(x=x_plot), '-', label='best fit')

        self.shift = self.shift_result.best_values['center'] - omega_TO
        self.x = self.x - self.shift
        return self.x

    def fit_si(self, fit_lims=[480, 560], shape=voigt, report=False):
        """Fit the Si Raman peak (~520 cm-1). Deprecated: use :meth:`fit_peaks`.

        Kept so that old calls keep working; it now calls::

            self.fit_peaks(n_peaks=1, fit_lims=[fit_lims], shapes=[shape],
                           plot=True, report=report)

        i.e. the same model and center bounds (the fit window) as before.
        Differences from the old version:
          - starting values are estimated from the data (the model's
            ``guess()``) instead of the fixed center 520, sigma 2, amplitude
            80000;
          - parameter names have the prefix ``peak1_`` (``peak1_center``
            instead of ``center``);
          - the plot is drawn on a new figure, not on the current axes.

        Parameters
        ----------
        fit_lims : [xmin, xmax]
        shape : lmfit model class or model
            Peak shape (default: Voigt).
        report : bool
            Print the fit report.

        Returns
        -------
        lmfit ModelResult
        """
        warnings.warn("Data.fit_si is deprecated and will be removed in a future "
                      "version; use Data.fit_peaks(fit_lims=[(xmin, xmax)], "
                      "shapes=[voigt]) instead.", FutureWarning, stacklevel=2)
        return self.fit_peaks(n_peaks=1, fit_lims=[tuple(fit_lims)], shapes=[shape],
                              plot=True, report=report)

    def fit_lopc(self, ax=None, fit_lims=[850, 1100], plot=True, init=True):
        """Fit the LO-phonon-plasmon coupled mode and derive carrier density and
        mobility.

        The x axis is first calibrated on the TO mode (:meth:`data_shift`), then
        :func:`lopc` + a constant is fitted between ``fit_lims`` with the
        Nelder-Mead method. From the plasma frequency omega_p and the plasmon
        damping gamma_p (converted from cm-1 to rad/s):

            n  = eps_0 * eps_inf * m_eff * omega_p**2 / e**2   (cm-3)
            mu = e / (m_eff * gamma_p)                          (cm2/Vs)

        Results are stored in ``self.doping`` (cm-3) and ``self.mu``.

        Uncertainties: with Nelder-Mead, lmfit estimates them only if
        ``numdifftools`` is installed.

        Parameters
        ----------
        fit_lims : [xmin, xmax]
            Fit window (cm-1).
        plot : bool
            Plot the fit and the constant baseline.
        init : bool
            Also plot the initial guess.

        Returns
        -------
        lmfit ModelResult
        """

        # if ax is None:
        #     fig, ax = plt.subplots()
        # else:
        #     fig = ax.figure()

        if not self.shift:
            self.data_shift()

        x = self.x
        y = self.y

        roi = np.where((x >= fit_lims[0]) & (x <= fit_lims[1]))
        x_cut = x[roi]
        y_cut = y[roi]

        # bl = baseline_poly
        bl = baseline_line()
        # model = lmfit.Model(lopc) + bl
        model = lmfit.Model(lopc) + lmfit.models.ConstantModel()

        # initial values
        S_0 = max(y_cut)
        omega_p_0 = 100
        gamma_p_0 = 500
        GAMMA_0 = 4.94

        model.set_param_hint('S', value=S_0)
        model.set_param_hint('omega_p', value=omega_p_0, min=0)
        model.set_param_hint('gamma_p', value=gamma_p_0, min=0)
        model.set_param_hint('GAMMA', value=GAMMA_0, vary=False)
        model.set_param_hint('omega_TO', value=omega_TO, vary=False)
        model.set_param_hint('omega_LO', value=omega_LO, vary=False)
        # model.set_param_hint('slope',
        # value=(y_cut[-1]-y_cut[0])/(x_cut[-1]-x_cut[0]))   # linear baseline
        # model.set_param_hint('intercept', value=y_cut[0])                       # linear baseline
        params = model.make_params()
        # params.update(bl.guess(y_cut, x=x_cut))

        self.fit_result = model.fit(y_cut, params, x=x_cut, method='nelder')
        print('\n'+self.fit_result.fit_report())
        # print(self.fit_result.best_values)
        if self.fit_result.success:
            self.omega_p_value = self.fit_result.uvars['omega_p']
            self.gamma_p_value = self.fit_result.uvars['gamma_p']

            # cm-1 -> rad/s: omega = 2*pi*c*nu, with c in m/s and 1e2 for cm-1 -> m-1
            omega_p_rad_s = 2 * np.pi * c * 1e2 * self.omega_p_value
            gamma_p_rad_s = 2 * np.pi * c * 1e2 * self.gamma_p_value

            if plot:
                self.fit_result.plot(show_init=init)
                ax = plt.gca()
                # baseline=self.fit_result.eval_components()['linear']
                baseline = self.fit_result.eval_components()['constant']
                ax.plot(x_cut, baseline, label='baseline')

            # Carrier concentration in m^-3
            n_m3 = (eps_0 * eps_inf * m_eff * omega_p_rad_s**2) / e**2

            # Convert to cm^-3
            n_cm3 = n_m3 / 1e6
            # mobility mu = e / (m_eff * gamma_p), x 1e4 for m2/Vs -> cm2/Vs
            mu = 1E4*(e/(gamma_p_rad_s*m_eff))

            self.doping = n_cm3
            self.mu = mu
            print(f'\nn={"{:e}".format(self.doping)}\tmu={self.mu}\n')
        else:
            print('Fit aborted!')

        return self.fit_result


class ProjectRamanMixin:
    """raman methods of :class:`Project` (mixed into it, not used alone)."""

    def analyze_si_z_scan(self, x_min=510, x_max=530, z_indices=None):
        """Integrated and maximum intensity of a Si peak along a z-scan map.

        Parameters
        ----------
        x_min, x_max : float
            Window (cm-1) around the peak.
        z_indices : list
            z value of each spectrum, in the order of ``self.data``.

        Returns
        -------
        z_vals, I_int, I_max : ndarray
            Sorted by z.
        """

        # use the first spectrum as reference x grid
        first_key = next(iter(self.data))
        x_full = np.asarray(self.data[first_key].x, dtype=float)
        mask = (x_full >= x_min) & (x_full <= x_max)
        x_cut = x_full[mask]

        I_int = []
        I_max = []
        z_vals = []

        for i, (k, d) in enumerate(self.data.items()):
            # retrieve y and apply same mask
            y_full = np.asarray(d.y, dtype=float)
            y_cut = y_full[mask]

            # numeric integral + max
            I_int.append(simpson(y_cut, x_cut))
            I_max.append(np.max(y_cut))
            z_vals.append(z_indices[i])

        z_vals = np.array(z_vals)
        I_int = np.array(I_int)
        I_max = np.array(I_max)

        # sort by z
        order = np.argsort(z_vals)
        return z_vals[order], I_int[order], I_max[order]

    def init_epi_sub_model(self, x_fit, y, x_min, x_max, model_type='gaussian'):
        """Build the epi + substrate + linear background model for the LO region.

        Two peaks (Gaussian or Lorentzian) with fixed centers, the epilayer LO
        mode at ``omega_LO`` and the substrate (LOPC-shifted) mode at 978 cm-1,
        plus a linear background. Used by :meth:`fit_epi_sub_vs_z`.

        Returns
        -------
        model, params : lmfit Model and Parameters
        """
        if model_type == 'gaussian':
            peak_epi = lmfit.models.GaussianModel(prefix='epi_')
            peak_sub = lmfit.models.GaussianModel(prefix='sub_')
        elif model_type == 'lorentzian':
            peak_epi = lmfit.models.LorentzianModel(prefix='epi_')
            peak_sub = lmfit.models.LorentzianModel(prefix='sub_')
        else:
            raise ValueError("model_type must be 'gaussian' or 'lorentzian'")

        baseline = lmfit.models.LinearModel(prefix='bkg_')
        model = peak_epi + peak_sub + baseline

        pars = model.make_params()
        y_max = y.max()
        x_center = x_fit[np.argmax(y)]

        # epi
        pars['epi_center'].set(value=omega_LO, vary=False)
        pars['epi_sigma'].set(value=2, min=0)
        pars['epi_amplitude'].set(value=0.5 * y_max * (x_max - x_min), min=0)

        # sub
        pars['sub_center'].set(value=978, vary=False)
        pars['sub_sigma'].set(value=2, min=0)
        pars['sub_amplitude'].set(value=0.5 * y_max * (x_max - x_min), min=0)

        pars['bkg_intercept'].set(value=np.median(y))
        pars['bkg_slope'].set(value=0.0)

        return model, pars

    def fit_epi_sub_vs_z(self, x_min=900, x_max=1100,
                         rel_tol=0.05, make_plots=False,
                         model_type='gaussian', norm=False):
        """Separate epilayer and substrate LO contributions along a z-scan.

        Every spectrum of the project is fitted between x_min and x_max with
        :meth:`init_epi_sub_model`; the peak areas are integrated and plotted
        vs. z. The z value is read from each Data label, which must look like
        'z=-5'. Prints whether the total area stays constant within ``rel_tol``.

        Parameters
        ----------
        make_plots : bool
            Also plot every single fit.
        model_type : {'gaussian', 'lorentzian'}
        norm : bool
            Plot the areas normalized by the total area instead of raw areas.

        Returns
        -------
        tuple
            (z_vals, epi_I, sub_I, tot_I, epi_rel, sub_rel, local_max,
            global_max), all sorted by z.
        """

        # plot title: the project's sample name, if any
        label = getattr(self, 'sample', '') or ''

        # reference x, mask
        ref_key = next(iter(self.data))
        x_full = np.asarray(self.data[ref_key].x, dtype=float)
        mask = (x_full >= x_min) & (x_full <= x_max)
        x_fit = x_full[mask]

        epi_I = []
        sub_I = []
        tot_I = []
        z_vals = []
        local_max_vals = []
        global_max_vals = []

        for d in self.data.values():
            # z index from label like "z=-5"
            try:
                z = int(str(d.label).split('=')[1])
            except Exception:
                raise ValueError(f"Data '{d.name}' needs a label like 'z=...'. Got '{d.label}'")

            y_full = np.asarray(d.y, dtype=float)
            y = y_full[mask]

            global_max_vals.append(np.max(y_full))
            local_max_vals.append(np.max(y))

            model, pars = self.init_epi_sub_model(x_fit, y, x_min, x_max, model_type)
            result = model.fit(y, pars, x=x_fit)
            comps = result.eval_components(x=x_fit)
            epi = comps['epi_']
            sub = comps['sub_']
            bkg = comps['bkg_']
            total = result.best_fit

            I_epi = simpson(epi, x_fit)
            I_sub = simpson(sub, x_fit)
            I_tot = simpson(y - bkg, x_fit)

            z_vals.append(z)
            epi_I.append(I_epi)
            sub_I.append(I_sub)
            tot_I.append(I_tot)

            if make_plots:
                fig, ax = plt.subplots()
                ax.plot(x_fit, y, 'k.', label=f'data z={z}')
                ax.plot(x_fit, total, 'r-', label='total fit')
                ax.plot(x_fit, epi, 'b--', label='epi peak')
                ax.plot(x_fit, sub, 'g--', label='sub peak')
                ax.set_xlabel('Raman shift (cm$^{-1}$)')
                ax.set_ylabel('intensity (a.u.)')
                ax.set_title(f'{self.sample} z={z}' if hasattr(self, 'sample') else f'z={z}')
                ax.legend()
                fig.tight_layout()

        z_vals = np.array(z_vals)
        order = np.argsort(z_vals)
        z_vals = z_vals[order]
        epi_I = np.array(epi_I)[order]
        sub_I = np.array(sub_I)[order]
        tot_I = np.array(tot_I)[order]
        local_max_vals = np.array(local_max_vals)[order]
        global_max_vals = np.array(global_max_vals)[order]

        epi_rel = epi_I / tot_I
        sub_rel = sub_I / tot_I

        I_ref = np.median(tot_I)
        area_ok = np.all([isclose(I, I_ref, rel_tol=rel_tol) for I in tot_I])
        print(f"Project '{getattr(self, 'sample', 'Raman map')}': "
              f"total area median={I_ref:.3g}, "
              f"min={tot_I.min():.3g}, max={tot_I.max():.3g}, "
              f"area_constant_within_{rel_tol*100:.1f}%={area_ok}")

        if norm:
            fig1, ax1 = plt.subplots()
            ax1.plot(z_vals, epi_rel, 'o-', label='epi (norm)')
            ax1.plot(z_vals, sub_rel, 's-', label='sub (norm)')
            ax1.set_xlabel('z index')
            ax1.set_ylabel('relative intensity (norm by I_tot)')
            ax1.set_title(f"{label} (normalized)")
            ax1.legend()
            fig1.tight_layout()
            ax1.set_xlim(z_vals[0], z_vals[-1])
        else:
            fig2, ax2 = plt.subplots()
            ax2.plot(z_vals, epi_I, 'o-', label='epi')
            ax2.plot(z_vals, sub_I, 's-', label='sub')
            ax2.set_xlabel('Distance from surface (μm)')
            ax2.set_ylabel('Intensity (a.u.)')
            ax2.set_title(f"{label}")
            ax2.legend()
            ax2.set_ylim(bottom=-100)
            ax2.set_yticklabels('')
            fig2.tight_layout()
            ax2.set_xlim(z_vals[0], z_vals[-1])

        return (z_vals, epi_I, sub_I, tot_I,
                epi_rel, sub_rel, local_max_vals, global_max_vals)

    @staticmethod
    def plot_peak_max(results_list, names, norm=False):
        """Plot the LO-region peak maximum vs. z for several samples.

        Parameters
        ----------
        results_list : list of tuple
            Outputs of :meth:`fit_epi_sub_vs_z`.
        names : list of str
            Legend labels, one per result.
        norm : bool
            Divide by the global spectrum maximum (TO peak).
        """
        fig, ax = plt.subplots(figsize=(9, 6))

        for data, name in zip(results_list, names):
            z_vals = data[0]
            local_max = data[6]
            global_max = data[7]

            if norm:
                y_to_plot = local_max / global_max
                ylabel = "Peak Max (norm)"
                title = "Peak Max (normalized by TO) vs Z"
            else:
                y_to_plot = local_max
                ylabel = "Peak Max (a.u.)"
                title = "Raw Peak Max vs Z"

            ax.plot(z_vals, y_to_plot, 'o-', label=name)
            ax.set_ylim(bottom= -max(y_to_plot)/10)

        ax.set_xlabel('Distance from surface (μm)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        if len(results_list) > 0:
            ax.set_xlim(results_list[0][0][0], results_list[0][0][-1])
