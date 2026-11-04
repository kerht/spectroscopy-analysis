"""Time-resolved PL: decay curves, lifetime fits, short/long series, lifetime plots."""

import lmfit
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from termcolor import colored
from uncertainties import ufloat, unumpy

from . import constants
from .models import baseline_line, constant, exp
from . import style as cs


def rescale_factor(proj_short, proj_long, int_lims=None):
    """Scale factor to join a "short" and a "long" TRPL series of one sample.

    The short series (gate G1, fine time steps) and the long series (gate G3,
    coarser steps) are measured separately. Every spectrum of both series is
    integrated; the function returns the normalized intensity of the short
    series at the delay where the long series starts:
    ``I_short[t_start_long - t_start_short] / max(I_short)``.

    Assumes 1 ns steps in the short series (the start-time difference is used
    directly as an index).

    Parameters
    ----------
    proj_short, proj_long : Project
        TRPL projects (one series each) of the same sample.
    int_lims : tuple, optional
        Integration window (nm). Default: ``INT_LIMS`` (325, 480).

    Returns
    -------
    float
        Factor to multiply the normalized long series by.
    """
    I1 = []
    I2 = []
    for k1, d1 in proj_short.data.items():
        i = d1.integrate(int_lims=int_lims)
        t1_start = d1.info['t_start']
        I1.append(i)
    for k2, d2 in proj_long.data.items():
        i = d2.integrate(int_lims=int_lims)
        t2_start = d2.info['t_start']
        I2.append(i)

    # norm1=max(I1)
    # norm2=max(I2)
    try:
        norm1 = max(I1)
    except ValueError:
        print(colored(f"'{proj_short}'", 'green',
                      attrs=['bold']), "is an empty sequence!")
    try:
        norm2 = max(I2)
    except ValueError:
        print(colored(f"'{proj_short}'", 'green',
                      attrs=['bold']), "is an empty sequence!")

    idx = int(t2_start)-int(t1_start)
    # multiply the starting intensity of the "long" data set by the
    # normalized intensity of the "short" set at the same delay
    rescale_norm = (I1[idx]/norm1)
    # rescale_norm = (I1[idx]/I2[0])*(norm2/norm1)
    return rescale_norm


def plot_tau(proj_list, ax=None, labels=None, color=None, err=None):
    """Plot lifetime vs. position on the wafer for a list of TRPL projects.

    The position is the integer after '#' in the sample name (e.g. 'P#2' -> 2).
    Points are sorted by position.

    Parameters
    ----------
    proj_list : list of Project
        One project per measured position; each must already have a fitted
        ``tau`` on its first Data.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if None.
    labels : list of str, optional
        Sample labels (currently not drawn on the axis).
    color : str, optional
        Marker color.
    err : float, optional
        Fixed error bar (ns), e.g. the repeatability of the measurement.
        Default None: the fit uncertainty of each lifetime.

    Returns
    -------
    list of ufloat
        Lifetimes (ns) with their error bars, sorted by position.
    """
    lifetimes_val = []
    lifetimes_err = []
    positions = []

    # 1. Gather lifetimes and error values along with their actual geometric positions
    for proj in proj_list:
        try:
            tau = list(proj.data.items())[0][1].tau
            lifetimes_val.append(tau.n)
            lifetimes_err.append(tau.s if err is None else err)
        except Exception as e:
            lifetimes_val.append(list(proj.data.items())[0][1].tau)
            lifetimes_err.append(0)
            print(e)

        # Extract the geometric position integer from sample string (e.g., 'P#2' -> 2)
        try:
            pos_str = proj.info[0]['samp'].split('#')[-1]
        except Exception:
            pos_str = proj.info['samp'].split('#')[-1]
        try:
            positions.append(int(pos_str))
        except Exception:
            positions.append(1)

    # 2. Build or override text labels matching the project list
    if labels is None:
        labels = []
        for proj in proj_list:
            try:
                labels.append(proj.info[0]['wafer'] + ' ' + proj.info[0]['samp'])
            except Exception as e:
                labels.append(proj.info['wafer'] + ' ' + proj.info['samp'])

    # 3. Sort all arrays simultaneously based on the geometric position value (x)
    # This prevents values from simply ordering in increasing magnitude from left to right
    sorted_combined = sorted(zip(positions, lifetimes_val, lifetimes_err, labels), key=lambda item: item[0])
    x, lifetimes_val, lifetimes_err, labels = zip(*sorted_combined)

    # Convert tuples back to lists/numpy arrays for plotting compatibility
    x = np.array(x)
    lifetimes_val = list(lifetimes_val)
    lifetimes_err = list(lifetimes_err)
    labels = list(labels)

    # 4. Handle plotting canvas setup
    if ax is None:
        fig, ax = plt.subplots()

    ax.errorbar(x, lifetimes_val, yerr=lifetimes_err, fmt='o', color=color)
    ax.set_xlim(left=min(x) - 2, right=max(x) + 2)
    ax.set_xticks(sorted(set(x)), minor=False)     # one tick per position
    ax.set_xticklabels([])                          # no labels
    ax.grid(visible=True, which='major')
    ax.set_ylabel('lifetime (ns)')
    ax.set_xlabel('position')

    lifetimes = []
    for i, j in zip(lifetimes_val, lifetimes_err):
        lifetimes.append(ufloat(i, j))

    return lifetimes


def plot_tau_hist(proj_list, ax=None, bins='auto', label=None, color=None):
    """Histogram of the lifetimes of a list of TRPL projects of one wafer.

    Parameters
    ----------
    proj_list : list of Project
        Projects with a fitted ``tau`` on their first Data.
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created if None.
    bins : int or str, optional
        Passed to ``ax.hist``. Default 'auto'.
    label : str, optional
        Legend label. Default: the wafer code of the first project.
    color : str, optional
        Bar color. Default: matplotlib's next color.

    Returns
    -------
    ndarray
        Lifetime values (ns).
    """
    lifetimes_val = []
    sample=proj_list[0].info[0].wafer

    for proj in proj_list:
        try:
            # print(proj.info)
            lifetimes_val.append(list(proj.data.items())[0][1].tau.n)
        except Exception as e:
            lifetimes_val.append(list(proj.data.items())[0][1].tau)
            print(e)

    lifetimes_val = np.array(lifetimes_val)

    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(lifetimes_val, bins=bins, color=color, density=False,
            alpha=0.7, edgecolor='black', linewidth=0.8)

    # KDE overlay
    # from scipy.stats import gaussian_kde
    # kde = gaussian_kde(lifetimes_val)
    # x_range = np.linspace(lifetimes_val.min(), lifetimes_val.max(), len(lifetimes_val))
    # ax.plot(x_range, kde(x_range), color='black', linewidth=2, alpha=0.9)

    # Styling
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_linewidth(0.8)
    # ax.spines['bottom'].set_linewidth(0.8)

    # ax.tick_params(axis='both', which='major', labelsize=11, width=0.8)
    # ax.tick_params(axis='x', direction='out', pad=6)
    # ax.tick_params(axis='y', direction='out', pad=6)

    # Grid (subtle)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Tight limits
    ax.margins(x=0.02, y=0)

    ax.set_xlabel('lifetime (ns)')
    ax.set_ylabel('counts')
    ax.legend([label if label is not None else sample])
    return lifetimes_val


def merge_short_long(proj_short, proj_long, int_lims=None):
    """Merge a short (G1) and a long (G3) TRPL series into one Project.

    Steps:
    1. Join the two ``data`` dicts and drop the last 6 spectra of the short
       series (overlap with the long one). Note: they are also removed from
       ``proj_short``.
    2. Integrate all spectra, normalize each series to its maximum and scale
       the long series so that it continues from the last short point.
    3. Build a common time axis (1 ns steps for the short series, 3 ns steps
       for the long series), starting at 0 at the short series start.

    Parameters
    ----------
    proj_short, proj_long : Project
        Short (G1) and long (G3) series of the same sample.
    int_lims : tuple, optional
        Integration window (nm). Default: ``INT_LIMS`` (325, 480), the same
        as :meth:`Project.plot_pl`. (Before, the merged series were always
        integrated over 320-520 nm.)

    Returns
    -------
    Project
        Merged project; each Data has ``I`` (normalized integral) and
        ``time`` (ns) set.
    """
    from .project import Project   # here, to avoid a circular import
    proj_merged = Project()

    proj_merged.data = proj_short.data | proj_long.data

    for i in range(6):
        k = list(proj_short.data.keys())[-1]
        del proj_merged.data[k]
        del proj_short.data[k]

    for k, d in proj_merged.data.items():
        d.data_all = proj_merged.data

    for k, d in proj_merged.data.items():
        d.I = d.integrate(int_lims=int_lims)

    I1 = []
    I2 = []
    for k, d in proj_merged.data.items():
        if '_G1' in k:
            I1.append(d.I)
        if '_G3' in k:
            I2.append(d.I)
    I1 = I1/max(I1)
    I2 = I2/max(I2)
    res = (I1[-1]/max(I1))
    I2 = I2*res
    I = [item for lst in [I1, I2] for item in lst]
    for i, (k, d) in zip(range(len(proj_merged.data.items())), proj_merged.data.items()):
        d.I = I[i]

    t_int = np.append(np.linspace(int(proj_short.info[0]['t_start']), int(proj_short.info[0]['t_stop'])-6,
                                  (int(proj_short.info[0]['t_stop']) - int(proj_short.info[0]['t_start'])-6)+1),
                      np.linspace(int(proj_long.info[0]['t_start']), int(proj_long.info[0]['t_stop']),
                                  int((int(proj_long.info[0]['t_stop']) - int(proj_long.info[0]['t_start']))/3+1)))
    t_int = t_int - int(proj_short.info[0]['t_start'])

    for k, d in proj_merged.data.items():
        d.time = t_int
    proj_merged.info = proj_short.info
    return proj_merged


class DataTRPLMixin:
    """trpl methods of :class:`Data` (mixed into it, not used alone)."""

    def plot_lifetime(self, ax=None, norm=False, fit=False, plot=False, fit_method='exp',
                      t_start=0, t_stop=0, g_width=0, shift=0, rescale=False,
                      mean=False, n_exp=2, yscale='linear', align=True, baseline=True, tau='avg'):
        """Build, plot and fit the TRPL decay of the series this spectrum belongs to.

        Run :meth:`integrate` on all spectra first (``Project.plot_pl`` does it):
        the decay is made of the integrals ``I`` of every Data in ``data_all``.

        Steps:
        1. Time axis from ``t_start``, ``t_stop``, ``g_width`` (read from info if
           0), unless ``self.time`` is set (merged series).
        2. Collect I, normalize to the maximum (``norm``), multiply by
           ``rescale``; values <= 0 are replaced by a small positive number.
        3. Fit (if ``fit``) with ``fit_method``:
             'line'    : straight line on ln(I); tau = -1/slope.
             'exp'     : single exponential (+ constant if ``baseline``),
                         amplitude fixed to 1 (use with norm=True).
             'two-exp' : two exponentials (+ constant).
             'conv'    : exponentials convolved with a Gaussian IRF
                         (:meth:`fit_conv`); with ``align`` the time axis is
                         shifted so the decay starts at 0.
           ``tau`` chooses the lifetime stored in ``self.tau`` for two components:
           'avg' (intensity-weighted, sum(a*t**2)/sum(a*t)), '0' or '1' (a single
           component).
        4. Plot data (markers) and fit (line) on ``ax``, in ln scale for
           yscale='log'.

        Plot styles come from ``ax.style``: if not set, an iterator over
        ``style.all_styles``; a list of (color, marker, linestyle) tuples is
        used in order; a single tuple is used as is.

        Parameters
        ----------
        plot : bool
            Also open a separate figure with fit and residuals.
        shift : float
            Added to the time axis (ns).
        mean : bool
            Use ``self.y_mean`` instead of the integrals (legacy).
        n_exp : int
            Number of exponentials for 'conv'.

        Returns
        -------
        fig, ax
        """

        self.current_fit_method = fit_method

        if self.meas_type == 'trpl':
            if t_start == 0 or t_stop == 0 or g_width == 0:
                try:
                    t_start = int(self.info['t_start'])
                    t_stop = int(self.info['t_stop'])
                    g_width = int(self.info['g_width'])
                except Exception as e:
                    print("Error occurred in method", colored('plot_lifetime', 'red',
                                                              attrs=['bold']), "file", colored(f"'{self.name}'", 'cyan',
                                                                                               attrs=['bold']), ":", type(e), e)
                    t_start = t_start
                    t_stop = t_stop
                    if type(e) == KeyError:
                        print('Detail: t_start/t_stop/g_width in info dict not found')

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # ---- Plot style (color, marker, linestyle) from ax.style
        if not hasattr(ax, 'style'):
            if fit:
                ax.style = iter(cs.all_styles)
            else:
                ax.style = iter(cs.color_ls)
            if fit:
                try:
                    color, marker, ls = next(ax.style)
                except Exception as e:
                    print("Error occurred in plot style assignation:", type(e), e)
                    # Optionally restart if all styles used
                    ax.style = iter(cs.all_styles)
                    color, marker, ls = next(ax.style)
            else:
                # marker=''
                try:
                    color, ls = next(ax.style)
                except Exception as e:
                    print("Error occurred in plot style assignation:", type(e), e)
                    # Optionally restart if all styles used
                    ax.style = iter(cs.color_ls)
                    color, ls = next(ax.style)

        elif isinstance(ax.style, list):
            # a list of (color, marker, linestyle): used in order, one per call
            ax.style = iter(ax.style)
            color, marker, ls = next(ax.style)

        elif type(ax.style) == tuple:
            try:
                color, marker, ls = ax.style
            except Exception as e:
                print("Error occurred in plot style assignation:", type(e), e)

        else:
            try:
                color, marker, ls = next(ax.style)
            except Exception as e:
                print("Error occurred in plot style assignation:", type(e), e)
                # Optionally restart if all styles used
                ax.style = iter(cs.all_styles)
                color, marker, ls = next(ax.style)

        if self.label is None:
            label = self.name.split('_')
            for i in range(1, 5):
                try:
                    label.pop(-1)
                except Exception as e:
                    print("Error occurred in method", colored('plot_lifetime', 'red',
                                                              attrs=['bold']), "file", colored(f"'{self.name}'", 'cyan',
                                                                                               attrs=['bold']), ":", type(e), e)
                    pass
            label = '_'.join(label)
            self.label = label
        else:
            label = self.label

        if t_start == 0 or t_stop == 0 or g_width == 0:
            print('Error: Missing time interval (ns) and gate width')
            return fig, ax

        # ---- Time axis (ns), starting at 0 (+ shift)
        if self.time is None:
            time = np.linspace(t_start, t_stop, int(
                (t_stop-t_start)/g_width)+1)
            time = time-t_start+shift
        else:
            time = self.time
            time = time+shift

        # ---- Decay curve: integrals of all spectra of the series
        if not mean:
            I = []
            # before calling this function the whole array of integrals is needed,i.e. using integrate
            if len(self.data) == 1:
                for k, d in self.data_all.items():
                    I.append(d.I)
            else:
                for k, d in self.data.items():
                    I.append(d.I)

            I = np.array(I)
            norm_factor = max(I)
            self.norm = norm_factor
            if norm_factor != I[0]:
                print('I_max is not the first spectrum')

            if norm:
                I = I/norm_factor

            if rescale != False:
                I = I*rescale

            for i in range(len(I)):
                if I[i] <= 0:
                    # I[i] = np.abs(I[i])
                    I[i] = min(np.abs(I[i]), np.e**(-6))
                    # I[i]=4e-2
                    print(
                        f"Negative value found in integration. Set to {I[i]}")
            # print(I)

            I_log = np.log(I)

        if mean:
            I = self.y_mean
            I_log = np.log(I)
            self.I = I

        # ---- Fit of the decay
        if fit:
            if fit_method == 'line':
                model = baseline_line()
                model.set_param_hint('slope', value=-0.03)
                model.set_param_hint('intercept', value=0)

            elif fit_method == 'exp':
                # # single exponential
                model = exp()

                if baseline is True:
                    model += constant()
                    model.set_param_hint('c', value=0.01)

                model.set_param_hint('amplitude', value=1, vary=False, min=0)
                model.set_param_hint('decay', value=10, min=0)

            elif fit_method == 'two-exp':
                # two exponentials
                exp1 = lmfit.models.ExponentialModel(prefix='exp1_')
                exp2 = lmfit.models.ExponentialModel(prefix='exp2_')
                model = exp1 + exp2

                if baseline is True:
                    model += constant()
                    model.set_param_hint('c', value=0.01)

                model.set_param_hint('exp1_amplitude', value=0.2, min=0, max=1)
                model.set_param_hint('exp2_amplitude', value=0.8, min=0, max=1)
                model.set_param_hint('exp1_decay', value=1, min=0, max=20)
                model.set_param_hint('exp2_decay', value=10, min=0, max=50)
            elif fit_method == 'conv':
                fit_result = self.fit_conv(
                    x=time, y=I, n=n_exp, baseline=baseline, tau_type=tau)
                # to shift the maximum at 0 ns
                if align:
                    time = time-fit_result.params['mid']+shift

            if fit_method != 'conv':
                params = model.make_params()

            if fit_method == 'line':
                fit_result = model.fit(I_log, params, x=time)
                # fit_result.data = np.exp(fit_result.data)
                self.tau = -1/(fit_result.uvars['slope'])

            elif fit_method == 'exp':
                # single exponential
                fit_result = model.fit(I, params, x=time)
                self.tau = fit_result.uvars['decay']

            elif fit_method == 'two-exp':
                # two exponentials
                # params['exp1_amplitude'].set(expr='1 - exp2_amplitude')
                fit_result = model.fit(I, params, x=time)
                try:
                    a1 = fit_result.uvars['exp1_amplitude']
                    tau1 = fit_result.uvars['exp1_decay']
                    a2 = fit_result.uvars['exp2_amplitude']
                    tau2 = fit_result.uvars['exp2_decay']
                except Exception as e:
                    print("Error occurred in method", colored('plot_lifetime', 'red',
                                                              attrs=['bold']), "file", colored(f"'{self.name}'", 'cyan',
                                                                                               attrs=['bold']), ":", type(e), '\nDetails: ', e)
                    a1 = fit_result.params['exp1_amplitude'].value
                    a2 = fit_result.params['exp2_amplitude'].value
                    tau1 = fit_result.params['exp1_decay'].value
                    tau2 = fit_result.params['exp2_decay'].value

                if tau == '0':
                    self.tau = tau1
                if tau == '1':
                    self.tau = tau2
                if tau == 'avg':
                    tau_mean = (a1*tau1**2+a2*tau2**2)/(a1*tau1+a2*tau2)
                    self.tau = tau_mean
                # ampl_weighted_tau = (a1*tau1+a2*tau2)/(a1+a2)
                # self.tau = ampl_weighted_tau


            self.fit_result = fit_result

            # print(f'{self.label} fit result: \n{fit_result.fit_report()}\n\n')
            # print(f'{self.label} best values: \n{fit_result.best_values}\n\n')
            print(f"{self.info['wafer']} {self.info['samp']} fit result: \n{
                  self.fit_result.fit_report(show_correl=False)}\n")
            print(f"{self.info['wafer']} {self.info['samp']
                                          } lifetime τ = {format(self.tau, '.4f')} ns\n")
            print("-" * 50)
            if plot:
                if fit_method == 'exp' or fit_method == 'two-exp':
                    fig_fit, ax_fit = plt.subplots(2, 1)
                    fit_result.plot_fit(ax_fit[0], title=f"{
                                        self.info['wafer']} {self.info['samp']}")
                    fit_result.plot_residuals(
                        ax_fit[1], title=f"{self.info['wafer']} {self.info['samp']}")
                    ax_fit[0].set_yscale('log')
                if fit_method == 'two-exp':
                    func1 = fit_result.eval_components()['exp1_']
                    func2 = fit_result.eval_components()['exp2_']
                    ax_fit[0].plot(time, func1, linestyle=':',
                                   label='exp1', color='red')
                    ax_fit[0].plot(time, func2, linestyle=':',
                                   label='exp2', color='blue')
                    ax_fit[0].legend()
                elif fit_method == 'conv':
                    fit_result.plot()
                    exp1 = fit_result.eval_components()['f0_']
                    plt.plot(time, exp1, color='red', linestyle=':')

                    if n_exp == 2:
                        exp2 = fit_result.eval_components()['f1_']
                        plt.plot(time, exp2, color='blue', linestyle=':')
                    # print(fit_result.fit_report())
                    # print(fit_result.params['avg_decay'])
                    ax_fit = plt.gca()
                    ax_fit.set_yscale('log')
                    ax_fit.set_ylim(top=3, bottom=10**(-3))
                    ax_fit.set_xlim(left=time[0], right=time[-1])
                # ax.plot(time, np.exp(fit_result.init_fit), '--', label='initial fit')
                # ax.plot(time, np.exp(fit_result.eval(x=time)), '-',
                #         label='best fit, τ = {:.2f}'.format(self.tau))
                ax_fit.set_ylabel('log(I/I(0))')
        # ---- Plot of data and fit on ax
        if yscale == 'log':
            if fit:
                # data: markers only, color from cycle
                data_line, = ax.semilogy(
                    time, I,
                    label=label,
                    base=np.e,
                    marker=marker,
                    linestyle='',
                    color=color
                )

                # fit: same color, unique linestyle
                ax.semilogy(
                    time, self.fit_result.best_fit,
                    # label=label + '_fit',
                    label=None,
                    base=np.e,
                    marker='',
                    color=color,
                    linestyle=ls
                )
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(cs.ticks))
                # locmin = mtick.LogLocator(base=np.e,
                #                           subs=np.arange(start=np.e/5, stop=np.e,
                #                                          step=np.e/5), numticks=999)
                # ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(mtick.NullFormatter())
                ax.set_ylabel('log(I/I(0))')
                ax.set_ylim(top=np.e**(0), bottom=np.e**(-5))

            else:
                ax.semilogy(
                    time, I,
                    label=label,
                    base=np.e,
                    marker='o',
                    linestyle='',
                    color=color
                )
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(cs.ticks))
                # locmin = mtick.LogLocator(base=np.e,
                #                           subs=np.arange(start=np.e/5, stop=np.e,
                #                                          step=np.e/5), numticks=999)
                # ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(mtick.NullFormatter())
                ax.set_ylabel('log(I)')
                ax.set_ylim(top=np.e**(0), bottom=np.e**(-5))

        elif yscale == 'linear':
            if fit:
                data_line, = ax.plot(
                    time, I,
                    label=label,
                    marker=marker,
                    linestyle='',
                    color=color
                )

                ax.plot(
                    time, self.fit_result.best_fit,
                    label=label + '_fit',
                    color=color,
                    linestyle=ls,
                    marker=''
                )
            else:
                ax.plot(
                    time, I,
                    label=label,
                    marker=marker,
                    linestyle='',
                    color=color
                )

        ax.set_xlabel('Time (ns)')
        # ax.set_ylabel('Intensity (AU)')

        ax.legend()
        ax.set_xmargin(0)
        ax.set_ymargin(0)
        ax.grid(visible=True)
        # ax.set_xlim(left=time[0], right=time[-1])

        colors = []
        labels = []
        for line in ax.lines:
            c = line.get_color()
            l = line.get_label()
            colors.append(c)
            labels.append(l)

        for l, c in zip(labels, colors):
            self.label_color_dict[l] = c
            # self.color = c

        return fig, ax

    def fit_conv(self, x, y, n=2, fitlims=None, tau=None,
                 irf_sigma=None, offset=0, offset_vary=True, baseline=True, tau_type='avg'):
        """Fit a TRPL decay with exponentials convolved with a Gaussian IRF.

        Model: sum of ``n`` exponentials starting at a step (``mid``), convolved
        with a Gaussian instrument response, plus a constant if ``baseline``.
        Initial decay times come from the 1/e time of the data (doubling for each
        further component) unless ``tau`` is given. The IRF width starts at the
        gate width.

        Derived parameters added to the fit:
          f<i>_ampl_rel     : fraction of the emission in component i,
          avg_decay         : intensity-weighted lifetime sum(a*t**2)/sum(a*t),
          ampl_weighted_tau : amplitude-weighted lifetime sum(a*t)/sum(a),
          offset            : shift of the IRF from the center of the time window.

        ``self.tau`` is set from ``tau_type`` ('avg', 'ampl_weighted', '0',
        '1'); for n=2 the relative amplitudes are stored in ``self.ampl_rel``.
        Needs ``self.norm``, set by :meth:`plot_lifetime`.

        Parameters
        ----------
        x, y : ndarray
            Time (ns) and intensity.
        n : int
            Number of exponentials.
        tau : list of float, optional
            Initial decay times.
        irf_sigma : float, optional
            Fix the IRF width instead of fitting it.
        offset, offset_vary :
            Initial IRF offset and whether it is fitted.

        Returns
        -------
        lmfit ModelResult
        """

        def convolve(arr, kernel):
            """Convolve ``arr`` with a normalized ``kernel`` and return an array of the
            same length as ``arr`` (edges padded with the end values).
            """
            # npts = arr.size
            npts = min(arr.size, kernel.size)
            pad = np.ones(npts)
            tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
            kernel /= np.sum(kernel)
            out = np.convolve(tmp, kernel, mode='valid')
            noff = int((len(out) - npts) / 2)
            return out[noff:noff+npts]

        def jump(x, mid):
            """Step function: 0 before ``mid``, 1 from ``mid`` on."""
            o = np.zeros(x.size)
            imid = max(np.where(x <= mid)[0])
            o[imid:] = 1.0
            return o

        def offset_exponential(x, amplitude=1, offset=0, decay=1):
            """Exponential decay starting at ``offset``."""
            decay = lmfit.lineshapes.not_zero(decay)
            return amplitude * np.exp(-(x-offset)/decay)

        # y/=max(y)

        # ---- Model: (sum of exponentials * step) convolved with a Gaussian IRF
        maxoffset = x[-1]-x[0]
        t_mid = (x[-1]+x[0])/2
        irf_mod = lmfit.models.GaussianModel(prefix='irf_')

        decays = [lmfit.Model(offset_exponential, prefix='f%i_' % i)
                  for i in range(n)]
        model = decays[0]
        for decay in decays[1:]:
            model += decay
        model *= lmfit.Model(jump)
        model = lmfit.CompositeModel(model, irf_mod, convolve)

        # initial guesses from the data: peak and 1/e time
        ymax = y.max()
        tmax = x[y.argmax()]
        tdecay = x[x > tmax]
        ydecay = y[x > tmax]
        ttau = tdecay[ydecay < ymax/np.e][0] - tmax

        if baseline is True:
            model += lmfit.models.ConstantModel()
            y_slice = y[x < tmax]
            if y_slice.size > 4:
                ybl = np.mean(y_slice[:-4])
            elif y_slice.size > 0:
                ybl = np.mean(y_slice)
            else:
                ybl = 0.0
            # ybl = np.mean(y[x < tmax][:-4])

        try:
            step = int(self.info['g_width'])
        except KeyError:
            step = int(self.info[0]['g_width'])

        if tau is None:
            tau = [ttau*(2**i) for i in range(n)]

        pars = model.make_params()

        if baseline is True:
            pars['c'].set(value=ybl, vary=True)

        for i in range(n):
            pars['f%i_amplitude' % i].set(value=ymax/n,
                                          vary=True, min=0)
            pars['f%i_decay' % i].set(value=tau[i],
                                      vary=True, min=0, max=x[-1]*10)
            # if irf is None:
            pars['f%i_offset' % i].set(value=tmax, expr='f0_offset',
                                       vary=False)
            # else:
            # pars['f%i_offset' % i].set(value=tmax*2, expr='f0_offset',
            # vary=True, min=-maxoffset, max=maxoffset)

        # AMPLITUDE CONSTRAINT
        # if n == 2:
        #     pars['f1_amplitude'].set(value=ymax/n, min=0, max=1, vary=True)
        #     pars['f0_amplitude'].set(expr='1 - f1_amplitude')

        pars['f0_offset'].expr = None
        pars['mid'].expr = 'f0_offset'
        # if irf is None:
        # pars['irf_center'].set(value=tmax, vary=offset_vary,
        #                         min=x[0], max=x[-1])
        pars['irf_center'].set(value=t_mid-offset, vary=offset_vary,
                               min=t_mid-maxoffset, max=t_mid+maxoffset)
        pars['irf_amplitude'].set(value=1, vary=False)
        if irf_sigma is None:
            pars['irf_sigma'].set(value=step, vary=True,
                                  min=0, max=step*10)
        else:
            pars['irf_sigma'].set(value=irf_sigma, vary=False)

        # ---- Derived parameters: relative amplitudes and average lifetimes
        terms = ['f%i_amplitude*f%i_decay' % (i, i) for i in range(n)]
        taus = ['f%i_decay' % i for i in range(n)]
        den = '('+'+'.join(terms)+')'
        num = '('+'+'.join([term+'**2' for term in terms])+')'
        for i, term in enumerate(terms):
            pars.add('f%i_ampl_rel' % i, expr=term+'/'+den)
        pars.add('avg_decay', expr=num+'/'+den)
        num_amp = '(' + '+'.join(terms) + ')'
        amps = '('+'+'.join(['f%i_amplitude' %i for i in range(n)])+')'
        pars.add('ampl_weighted_tau', expr=num_amp+'/'+amps)
        pars.add('offset', expr='%f-irf_center' % t_mid)

        # debug for NaN, inf parameters
        # for name, par in pars.items():
        #     print(f"{name}: value={par.value}, min={par.min}, max={par.max}")

        # Poisson-like weights (computed but currently not used in the fit)
        var_area = y/ self.norm

        # lmfit expects weights that multiply residuals; use inverse std dev in area units:
        weights = 1.0 / np.sqrt(var_area)
        # print("weights: min,median,max:", np.min(weights), np.median(weights), np.max(weights))
        # print("weights finite? any NaN/inf:", np.any(~np.isfinite(weights)))
        # print("y: min,median,max:", np.min(y), np.median(y), np.max(y))


        fitres = model.fit(y, pars, x=x, # weights=weights,
                            method='least_squares', max_nfev=1e16)

        try:
            if tau_type == '1':
                self.tau = fitres.uvars['f1_decay']
            elif tau_type == '0':
                self.tau = fitres.uvars['f0_decay']
            elif tau_type == 'avg':
                self.tau = fitres.uvars['avg_decay']
            elif tau_type == 'ampl_weighted':
                self.tau = fitres.uvars['ampl_weighted_tau']
        except Exception as e:
            print(f"Exception {
                  e}: could not determine tau from fit. Setting to 0+-1")
            self.tau = ufloat(0, 1)
        if n == 2:
            self.ampl_rel = {'f0':fitres.uvars['f0_ampl_rel'],
                             'f1':fitres.uvars['f1_ampl_rel']
                             }
        return fitres

    def summarize_lifetime_fit(self):
        """Summary of the last lifetime fit of this spectrum.

        Finds every exponential component in ``self.fit_result`` (parameters
        ending in '_decay' with a matching '_amplitude') and returns them with
        uncertainties.

        Returns
        -------
        dict
            'fit_method', 'tau_avg', 'tau1', 'amplitude_1', 'tau2',
            'amplitude_2' (None where missing).
        """
        if not hasattr(self, "fit_result"):
            return {"tau_avg": getattr(self, "tau", None)}

        res = self.fit_result
        p = res.params

        def _u(name):
            if name in res.uvars: return res.uvars[name]
            if name in p:
                par = p[name]
                return ufloat(par.value, par.stderr or 0.0)
            return None

        # Auto-detect ALL possible components (any order)
        taus, amps = [], []

        # Find all exponential components (expN_, fN_)
        for key in p:
            if "_decay" in key:
                tau = _u(key)
                # Try both common naming conventions for amplitudes
                amp_key = key.replace("_decay", "_amplitude")
                if amp_key not in p:
                    amp_key = key.replace("_decay", "_amp")

                amp = _u(amp_key)
                if tau is not None and amp is not None:
                    taus.append(tau)
                    amps.append(amp)

        # tau_avg from self.tau or intensity-weighted average
        tau_avg = getattr(self, "tau", None)
        if not tau_avg and len(taus) >= 2:
            amps_n = unumpy.nominal_values(amps)
            taus_n = unumpy.nominal_values(taus)
            tau_avg = np.average(taus_n, weights=amps_n)
            tau_avg = ufloat(tau_avg, 0.1 * tau_avg)  # rough error

        # Pack first 2 components (tau1/tau2 or None)
        result = {
            "fit_method": getattr(self, "current_fit_method", "N/A"), # Capture the method
            "tau_avg": tau_avg
            }

        for i in range(2):
            result[f"tau{i+1}"] = taus[i] if i < len(taus) else None
            result[f"amplitude_{i+1}"] = amps[i] if i < len(amps) else None

        return result


class ProjectTRPLMixin:
    """trpl methods of :class:`Project` (mixed into it, not used alone)."""

    def plot_pl(self, ax=None, norm=True, rescale=False, fit=True, fit_method='two-exp',
                plot_fit_exciton=False, plot_fit=False, t_int=[], int_lims=None,
                g_width=0, shift=0, n_exp=2, bl=baseline_line, yscale='log',
                baseline=False, tau='avg', force=False):
        """Integrate every TRPL spectrum of the project and plot/fit the decay.

        Each spectrum is integrated over ``int_lims`` (nm) after the exciton
        baseline fit (:meth:`Data.integrate`); then
        :meth:`Data.plot_lifetime` is called on the first spectrum, which collects
        the integrals of the whole series.

        Parameters
        ----------
        t_int : [t_start, t_stop], optional
            Time window (ns); read from the file name if empty.
        fit_method : {'line', 'exp', 'two-exp', 'conv'} or None
            None -> 'line' for yscale='log', 'exp' for yscale='linear'.
        int_lims : tuple, optional
            Integration window (nm). Default: the module constant
            ``INT_LIMS`` (325, 480).
        force : bool
            Recompute the integrals even if the spectra were already
            integrated (e.g. to try a different ``int_lims``). Leave False
            for series merged with :func:`merge_short_long`.
        Other parameters are passed to :meth:`Data.plot_lifetime`.
        """
        if int_lims is None:
            int_lims = constants.INT_LIMS

        if t_int == []:
            try:
                t_int = [self.info[0]['t_start'], self.info[0]['t_stop']]
            except Exception as e:
                print("Error occurred in method", colored('plot_pl', 'red',
                                                          attrs=['bold']), "project", colored(f"'{self}'", 'green',
                                                                                              attrs=['bold']), ":", type(e), e)

        t_start = t_int[0]
        t_stop = t_int[1]

        # print(self.info, type(self.info))
        keys = []

        # warn once if stored integrals were computed over other limits
        if not force:
            n_stale = sum(1 for d in self.data.values()
                          if d.int_status and getattr(d, 'int_lims', None) != tuple(int_lims))
            if n_stale:
                print(colored('plot_pl', 'red', attrs=['bold']) +
                      f": {n_stale} spectra were already integrated with other limits; "
                      f"using the stored integrals, not int_lims={tuple(int_lims)}. "
                      "Pass force=True to recompute (not for merged short/long series).")

        for k, d in self.data.items():
            keys.append(str(k))
            d.integrate(int_lims=int_lims, plot_fit=plot_fit_exciton, bl=bl,
                        force=force)

        if fit_method == None:
            if yscale == 'log':
                fit_method = 'line'
            if yscale == 'linear':
                fit_method = 'exp'

        # data = self.data[k[0]]
        self.data[keys[0]].plot_lifetime(ax=ax, norm=norm, fit=fit, fit_method=fit_method,
                                         t_start=t_start, t_stop=t_stop,
                                         g_width=g_width, rescale=rescale, plot=plot_fit,
                                         shift=shift, n_exp=n_exp, yscale=yscale,
                                         baseline=baseline, tau=tau)

    def plot_pl_all(self, ax=None, norm=False, fit=False, shift=True):
        """Plot all spectra of a TRPL series side by side.

        Each spectrum is drawn shifted along x by 1.05 times the spectral range,
        so the series reads as intensity vs. delay; x tick labels are the spectrum
        indices. The x of every spectrum is restored afterwards; y and the raw
        data are not touched.

        Returns
        -------
        fig, ax
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        tick_positions = []

        first = next(iter(self.data.values()))
        x_range = np.max(first.x)-np.min(first.x)
        # y_range=np.max(self.data.values()[0].x)-np.min(self.data.values()[0].x)
        for i, (k, d) in enumerate(self.data.items()):
            # spectrum i is drawn shifted by (i + 1) * 1.05 * range: the same
            # layout as before, when all spectra shared one x array and each
            # in-place shift added up. Now every Data owns its x, so the shift
            # is computed explicitly and x is restored exactly afterwards
            # (before, spectra 2, 3, ... stayed shifted after the plot).
            x_orig = d.x
            if shift is True:
                d.x = x_orig + (i + 1) * 1.05 * x_range     # new array
            d.plot(ax, norm=norm)
            center = (d.x.min() + d.x.max()) / 2
            tick_positions.append(center)
            # tick_labels.append(str(i)) # Label starting from 0
            d.x = x_orig
        ax.legend().remove()
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xticks(tick_positions, minor=False)
        ax.set_xticklabels(list(range(1,len(self.data)+1)), fontsize=8)
        ax.set_yticklabels([])
        ax.set_xlabel('---- delay (ns) ---->')
        ax.set_ylabel('')
        ax.set_ylim(bottom=0)
        ax.minorticks_off()
        ax.grid(axis='x')
        ax.tick_params(
            top=False,
            labeltop=False,
            right=False,
            labelright=False
            )
        for l in ax.lines:
            l.set_linewidth(1.2)
            l.set_linestyle('-')
            l.set_markersize(0.2)
            l.set_color('blue')
        return fig, ax

    def plot_pl_mean(self, folder, name, n_mean, ax=None, dark=False, color=None,
                     t_int=[], g_width=0, shift=0, plot_fit_exciton=False, plot_fit=False,
                     fit_method='two-exp', int_lims=None, label=None,
                     norm=True, fit=True, n_exp=2, bl=baseline_line, yscale='log',
                     baseline=False, tau='avg'):
        """Combine repeated TRPL acquisitions and plot/fit the resulting decay.

        Loads ``<name>_1`` ... ``<name>_<n_mean>`` from ``folder``, sums the
        corresponding spectra (same delay) of all repetitions, stores the result
        as a new series and calls :meth:`plot_pl` on it. The spectra are summed,
        not averaged: this does not matter once the decay is normalized.

        Parameters
        ----------
        folder, name : str
            Folder and common file name (without the _i suffix).
        n_mean : int
            Number of repeated acquisitions.
        int_lims : tuple, optional
            Integration window (nm). Default: ``INT_LIMS`` (325, 480).
        Other parameters are passed to :meth:`plot_pl`.

        Returns
        -------
        Project
            Project holding the summed series (also stored in ``self.data``).
        """
        from .data import Data   # here, to avoid a circular import
        from .project import Project   # here, to avoid a circular import

        self.proj_all = []
        names = []

        for i in range(n_mean):
            self.proj_all.append(Project())
            names.append(name+'_%i' % (i+1))
            self.proj_all[i].open_pl(
                folder, names[i], color=color, label=label)
            self.info = self.proj_all[i].info

        for i in self.proj_all:
            for k, d in i.data.items():
                try:
                    t_int = [d.info['t_start'], d.info['t_stop']]
                except Exception as e:
                    print("Error occurred in method", colored('plot_pl_mean', 'red',
                                                              attrs=['bold']), "project", colored(f"'{self}'", 'green',
                                                                                                  attrs=['bold']), ":", type(e), e)

        t_start = t_int[0]
        t_stop = t_int[1]

        y_all = [[] for _ in range(n_mean)]

        first_key = list(self.proj_all[0].data.keys())[0]
        n_spectra = len(self.proj_all[0].data)
        X = self.proj_all[0].data[first_key].x
        Y = np.zeros((n_spectra, len(X)))  # Shape (n_spectra, len(X))

        # sum the spectra with the same delay over all repetitions
        for i in range(len(self.proj_all)):  # Loop over files
            # Loop over n_spectra
            for j, (key, d) in enumerate(self.proj_all[i].data.items()):
                # print(key)
                Y[j] += d.y  # Sum corresponding spectra

        merged_project = Project()

        # print(self.info, type(self.info))
        # self.info = self.info[0].to_dict()
        data_map = {}
        for i in range(n_spectra):
            data_map[name+'_%i' % (i+1)] = Data(name+'_%i' % (i+1), X, Y[i], info=self.info,
                                                data={}, dark=False, color=color, label=label)

        merged_project.data = data_map
        for k, d in merged_project.data.items():
            d.data[k] = d
            d.data_all = data_map
            if label is None:
                d.label = d.name

        self.data = merged_project.data
        merged_project.plot_pl(ax=ax, norm=norm, fit=fit, fit_method=fit_method, t_int=t_int,
                               g_width=g_width, plot_fit=plot_fit, plot_fit_exciton=plot_fit_exciton,
                               n_exp=n_exp, shift=shift, bl=bl, yscale=yscale,
                               baseline=baseline, tau=tau, int_lims=int_lims)

        # ---- old approach: average of the integrals of the repetitions

        # for i in data_map:
        #     i.integrate()

        # data_map[0].plot_lifetime(ax=ax, norm=norm, fit=fit, fit_method=fit_method,
        #                                   t_start=t_start, t_stop=t_stop, plot=plot_fit,
        #                                       g_width=g_width, shift=shift,
        #                                           yscale=yscale)
        # merged_data.integrate()

        # merged_data.plot_lifetime(ax=ax, norm=norm, fit=fit, fit_method=fit_method,
        #                                   t_start=t_start, t_stop=t_stop, plot=plot_fit,
        #                                       g_width=g_width, shift=shift,
        #                                           mean=True, yscale=yscale)

        # return merged_data

        # for i in range(n_mean):
        #     for k, d in self.proj_all[i].data.items():
        #         d.integrate(int_lims=int_lims, plot_fit=plot_fit_exciton)
        #     y_all[i] = [d.I for k,d in self.proj_all[i].data.items()]

        # y_mean = []
        # y_std_deviations = []

        # for k in range(len(y_all[0])):
        #     # mean_k = sum(y_all[j][k] for j in range(n_mean)) / n_mean
        #     mean_k = np.mean([y_all[j][k] for j in range(n_mean)])
        #     y_std_deviations.append(np.std([y_all[j][k] for j in range(n_mean)]))
        #     y_mean.append(mean_k)
        #     y_std_deviations[k] = y_std_deviations[k]/y_mean[k]
        #     # print(y_mean[k], '\t', y_std_deviations[k])

        # if norm:
        #     norm_factor = max(y_mean)
        #     if norm_factor != y_mean[0]:
        #         print('I_max is not the first spectrum')
        #     y_mean= y_mean/norm_factor

        # for i in range(n_mean):
        #     self.data[names[i]] = self.proj_all[i].data[names[i]+'_%i'%(i+1)]
        #     self.data[names[i]].y_mean = y_mean

        # if fit_method == None:
        #     if yscale == 'log':
        #         fit_method = 'line'
        #     if yscale == 'linear':
        #         fit_method == 'exp'

        # self.data[names[0]].plot_lifetime(ax=ax, norm=norm, fit=fit, fit_method=fit_method,
        #                                   t_start=t_start, t_stop=t_stop, plot=plot_fit,
        #                                       g_width=g_width, shift=shift,
        #                                           mean=True, yscale=yscale)

        # results = {}
        # results['means'] = [format(y_mean[i], ".4f") for i in range(len(y_mean))]
        # # results['std_deviations'] = y_std_deviations
        # results['std_deviations'] = [format(y_std_deviations[i], ".2%") for i in range(len(y_std_deviations))]

        # # print(f"Sample {self.proj_all[0].info[0].wafer} {self.proj_all[0].info[0].samp}: \n",\
        #       # results, '\n')

        return merged_project

    @staticmethod
    def summarize_lifetimes(projs, fmt=".1uP"):
        """Table of lifetime-fit results for one or more TRPL projects.

        Uses :meth:`Data.summarize_lifetime_fit` on the first Data of each
        project and prints the table.

        Parameters
        ----------
        projs : Project or list of Project
        fmt : str
            Format for the ufloat values (default '.1uP').

        Returns
        -------
        pandas.DataFrame
        """
        if not isinstance(projs, list):
            projs = [projs]

        rows = []
        for proj in projs:
            try:
                first_d = next(iter(proj.data.values()))
                s = first_d.summarize_lifetime_fit()  # raw uvars

                def _fmt(v):
                    return f"{v:{fmt}}" if v is not None else "-"

                row = {
                    "wafer": getattr(first_d.info, "wafer", "-"),
                    "samp": getattr(first_d.info, "samp", "-"),
                    "method": s.get("fit_method", ""),
                    "tau_avg": _fmt(s["tau_avg"]),
                    "tau1": _fmt(s["tau1"]),
                    "amp_1": _fmt(s["amplitude_1"]),
                    "tau2": _fmt(s["tau2"]),
                    "amp_2": _fmt(s["amplitude_2"]),
                }
                rows.append(row)
            except Exception as e:
                print(f"Skipping {proj.sample}: {e}")
                continue

        df = pd.DataFrame(rows)
        if not df.empty:
            print(f"\nMulti-project summary ({len(df)} projects):")
            print(df.to_string(index=False))

        return df
