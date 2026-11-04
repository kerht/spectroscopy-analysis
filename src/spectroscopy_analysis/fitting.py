"""Generic multi-peak fit (Data.fit_peaks)."""

import copy
import lmfit
import matplotlib.pyplot as plt
import numpy as np

from .models import baseline_line, voigt


class DataFitMixin:
    """fitting methods of :class:`Data` (mixed into it, not used alone)."""

    def fit_peaks(self, n_peaks=1, fit_lims=None, shapes=None, param_hints=None,
                  baseline=None, plot=True, report=False, ax=None):
        """Fit one or more peaks, each inside its own x window.

        Every peak gets its own lmfit model with prefix ``peak1_``,
        ``peak2_``, ... and the models are summed. The fit uses the points
        inside any of the windows (``fit_lims``); each window also bounds the
        center of its peak. With ``baseline`` set, a background model is
        fitted together with the peaks; otherwise the data should be close
        to zero around the peaks, or background-subtracted first.

        Initial values, for each peak:
          - from ``param_hints`` if given;
          - otherwise from the data in the window, with the model's own
            ``guess()`` (height converted to area for each shape).
        In both cases the center is then bounded to its window.

        Parameters
        ----------
        n_peaks : int
            Number of peaks.
        fit_lims : list of (xmin, xmax)
            One window per peak, in the units of ``self.x`` (e.g.
            ``[(510, 530)]`` for the Si peak, ``[(510, 530), (760, 790)]``
            for Si + TO). Points inside the windows are fitted; window ``n``
            also sets min/max of ``peak<n+1>_center``.
        shapes : list of lmfit model classes or models, optional
            One shape per peak, e.g. ``[gauss, voigt]``. A model instance
            (``voigt()``) is accepted too: only its class is used. Default:
            Voigt for every peak.
        param_hints : list of (center, sigma, amplitude), optional
            Initial values, one tuple per peak. ``amplitude`` is the AREA of
            the peak (lmfit convention), not its height. Default: estimated
            from the data.
        baseline : None, bool, lmfit model class or model, optional
            Background fitted together with the peaks (prefix ``bl_``):
              - None or False : no background (default);
              - True          : a straight line (``baseline_line``);
              - a class or factory, e.g. ``baseline_poly`` or
                ``lmfit.models.QuadraticModel``;
              - a model instance, e.g.
                ``lmfit.models.PolynomialModel(degree=3)``: a copy is used,
                so the caller's model is not modified.
            The background is not subtracted beforehand: the peak
            parameters returned are already corrected for it. Its starting
            values come from the first and last points of the fitted region
            (5 %, at least 3 per side). One background is used for all
            windows: use one window, or windows close together (far-apart
            windows share one background across the gap; fit them with
            separate calls instead).
        plot : bool
            Plot the whole spectrum (raw data) with the initial and best fit
            over the fitted region, and the background if fitted.
        report : bool
            Print the lmfit fit report.
        ax : matplotlib Axes, optional
            Axes to plot on. Default: a new figure.

        Returns
        -------
        lmfit ModelResult
            Peak parameters are named ``peak1_center``, ``peak1_sigma``,
            ``peak1_amplitude``, ``peak1_height``, ``peak1_fwhm``, ...
            Background parameters depend on the model: ``bl_slope`` and
            ``bl_intercept`` for a line, ``bl_c0``, ``bl_c1``, ... for a
            polynomial.

        Examples
        --------
        >>> r = d.fit_peaks(fit_lims=[(510, 530)], shapes=[voigt])
        >>> r.params['peak1_center'].value

        >>> r = d.fit_peaks(fit_lims=[(510, 530)], baseline=True)
        >>> r = d.fit_peaks(fit_lims=[(500, 540)], baseline=baseline_poly)
        """
        shapes = shapes or [voigt] * n_peaks          # default: Voigt
        param_hints = param_hints or []
        # a model instance is accepted too: build a new one of the same class
        # (with the prefix), so the caller's model is never modified
        peaks = [(type(shapes[n]) if isinstance(shapes[n], lmfit.Model) else shapes[n])
                 (prefix=f"peak{n+1}_") for n in range(n_peaks)]

        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        mask = np.zeros(x.size, dtype=bool)

        model = peaks[0]
        for s in peaks[1:]:
            model += s

        params = model.make_params()

        for n in range(n_peaks):
            lo, hi = fit_lims[n]
            x_w, y_w, w = self.get_roi(lo, hi)
            mask |= w
            pre = f"peak{n+1}_"

            if param_hints:
                cen, sig, amp = param_hints[n]
                params[pre + 'center'].set(value=cen)
                # abs: the spacing is negative on decreasing axes (e.g. eV)
                params[pre + 'sigma'].set(value=sig, min=np.abs(np.diff(x_w)).min())
                params[pre + 'amplitude'].set(value=amp, min=0)
            else:
                params.update(peaks[n].guess(y_w, x=x_w))

            params[pre + 'center'].set(min=lo, max=hi)

        x_cut, y_cut = x[mask], y[mask]           # the points that are fitted

        if baseline is not None and baseline is not False:
            if baseline is True:
                bl = baseline_line(prefix="bl_")           # default: straight line
            elif isinstance(baseline, lmfit.Model):
                bl = copy.deepcopy(baseline)               # keeps options like degree;
                bl.prefix = "bl_"                          # the caller's model is untouched
            else:
                bl = baseline(prefix="bl_")                # class, or baseline_poly (partial)
            model += bl
            # start the line from the background: a few points at each end of
            # the fitted region (5 %, at least 3), not through the peaks
            k = max(3, x_cut.size // 20)
            x_edges = np.concatenate([x_cut[:k], x_cut[-k:]])
            y_edges = np.concatenate([y_cut[:k], y_cut[-k:]])
            params.update(bl.guess(y_edges, x=x_edges))

        fit_result = model.fit(y_cut, params, x=x_cut)

        if report:
            print(fit_result.fit_report())

        if plot:
            if ax is None:
                fig, ax = plt.subplots()
            # whole spectrum as raw data, fit drawn over the fitted region
            x_plot = np.linspace(x_cut.min(), x_cut.max(), 1000)
            ax.plot(x, y, 'o')
            ax.plot(x_cut, fit_result.init_fit, '--', label='initial fit')
            ax.plot(x_plot, fit_result.eval(x=x_plot), '-', label='best fit')
            if baseline:
                comps = fit_result.eval_components(x=x_plot)
                ax.plot(x_plot, comps['bl_'], ':', label='baseline')

        return fit_result
