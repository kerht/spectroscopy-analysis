"""Data.plot: unit conversions, normalisation and styling of a spectrum."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
from termcolor import colored

from .units import cmtoev, cmtonm, evtocm, evtonm, nmtocm, nmtoev
from . import style as cs


class DataPlotMixin:
    """plotting methods of :class:`Data` (mixed into it, not used alone)."""

    def plot(self, ax=None, norm=False, shift=False, d_sub=True, baseline=False,
             fit=False, xlims=None, xscale=None, timeout=10, label=None):
        """Plot the spectrum.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            New figure if None.
        norm : False, 'norm_max', list/ndarray or tuple
            False      : raw intensity.
            'norm_max' : divide by the maximum.
            [xmin, xmax] (list or array) : divide by the maximum in that window.
            (xmin, xmax) (tuple)         : divide by the area in that window.
        shift : bool
            Calibrate x on the TO mode first (:meth:`data_shift`).
        d_sub : bool
            Subtract the dark first (always False for TRPL).
        xscale : str, optional
            'cm-1', 'nm', 'eV' (no conversion) or a conversion: 'cm-1 to eV',
            'cm-1 to nm', 'eV to nm', 'eV to cm-1', 'nm to cm-1', 'nm to eV'.
            Conversions involving cm-1 use ``self.ecc``.
            Default (None): the units of the data if known (``self.x_unit``,
            set by :meth:`Project.open_file`), otherwise 'cm-1' as before.
            When x is not converted, the axis labels read from the file
            (``self.x_label``, ``self.y_label``) are used if available.
        xlims : (xmin, xmax), optional
            x-axis limits.
        label : str, optional
            Legend label; stored in ``self.label``.
        baseline, fit, timeout :
            Currently unused.

        Lines with the same color and linestyle get a different linestyle
        automatically.

        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Plot's labels and colors
        if self.meas_type == 'trpl':
            if self.label is None:
                label = self.name.split('_')
                for i in range(1, 4):
                    try:
                        label.pop(-2)
                    except Exception as e:
                        print("Error occurred in method", colored('plot', 'red',
                                                                  attrs=['bold']), "file", colored(f"'{self.name}'", 'cyan',
                                                                                                   attrs=['bold']), ":", type(e), e)
                label = '_'.join(label)
                self.label = label
            d_sub = False
            # xscale = 'nm'

        if label is not None:
            self.label= label
        elif label is None and self.label is None:
            label = self.name
            self.label = label
        else:
            label = self.label

        # if label is None:
        #     if self.label is None:
        #         label = self.name
        #         self.label = label
        #     else:
        #         label = self.label

        if d_sub:
            self.subtract_dark()

        if shift:
            self.data_shift()

        # default x scale: native units of the data if known, else 'cm-1'
        # (previous default); 'native' = unknown units, no conversion
        if xscale is None:
            if self.x_unit in ('cm-1', 'nm', 'eV'):
                xscale = self.x_unit
            elif self.x_unit is None and self.x_label is None:
                xscale = 'cm-1'
            else:
                xscale = 'native'

        # x-y units conversions
        if xscale in ('cm-1', 'eV', 'nm', 'native'):
            x = self.x
            y = self.y
        elif xscale == 'cm-1 to eV':
            x, y = cmtoev(self.x, self.y, ecc=self.ecc)
            print('x units converted from cm-1 to eV, intensity adjusted.')
        elif xscale == 'cm-1 to nm':
            x = cmtonm(self.x, ecc=self.ecc)
            y = self.y
            print('x units converted from cm-1 to nm')
        elif xscale == 'eV to nm':
            x, y = evtonm(self.x, self.y)
            print('x units converted from eV to nm, intensity adjusted.')
        elif xscale == 'eV to cm-1':
            x, y = evtocm(self.x, self.y, ecc=self.ecc)
            print('x units converted from eV to cm-1, intensity adjusted.')
        elif xscale == 'nm to cm-1':
            x = nmtocm(self.x, ecc=self.ecc)
            y = self.y
            print('x units converted from nm to cm-1')
        elif xscale == 'nm to eV':
            x, y = nmtoev(self.x, self.y)
            print('x units converted from nm to eV, intensity adjusted.')
        else:
            print('x units not converted')
            x = self.x
            y = self.y

        # ---- norm by raw maximum
        if norm == 'norm_max':
           y = y/max(y)

        # ---- norm by maximum within range (must be 'list' or 'ndarray')
        elif type(norm) == np.ndarray or type(norm) == list:
            try:
                norm_range = np.array(norm)
                x_peak = np.where((x >= norm_range[0]) & (x <= norm_range[-1]))
                x_peak, y_peak = x[x_peak], y[x_peak]
                norm = max(y_peak)
                y = y/norm
            except Exception as e:
                print(f"Error occurred in normalization of sample {
                      self.info.SAMPLE}: {e}")

        # ---- norm by aera within range (must be 'tuple')
        elif type(norm) == tuple:
            try:
                norm_range = np.array(norm)
                x_peak = np.where((x >= norm_range[0]) & (x <= norm_range[-1]))
                x_peak, y_peak = x[x_peak], y[x_peak]
                norm = simpson(y_peak, x=x_peak)
                if norm < 0:
                    norm= -norm
                y = y/norm
            except Exception as e:
                print(f"Error occurred in normalization of sample {
                      self.info.SAMPLE}: {e}")

        if fit:
            pass

        if xlims is not None:
            x = x.clip(xlims[0], xlims[1])

        if self.meas_type == 'Raman':
            marker = None
        elif self.meas_type == 'trpl':
            marker = 'o'
        else:
            marker = None

        ax.plot(x, y, label=label, color=self.color,
                marker=marker, linestyle='-')

        if xscale.split(' ')[-1] == 'cm-1':
            ax.set_xlabel('Raman shift (cm$^{-1}$)')
        elif xscale.split(' ')[-1] == 'eV':
            ax.set_xlabel('Energy (eV)')
        elif xscale.split(' ')[-1] == 'nm' or self.meas_type == 'trpl':
            ax.set_xlabel('Wavelenght (nm)')
        ax.set_ylabel('Intensity (AU)')
        if self.meas_type == 'absorption':
            ax.set_ylabel('Absorbance (O.D.)')
            ax.set_xlabel('Energy (eV)')

        # labels read from the file (open_file) win when x is not converted
        if ' to ' not in xscale:
            if self.x_label:
                ax.set_xlabel(self.x_label)
            if self.y_label:
                ax.set_ylabel(self.y_label)

        if xscale == 'nm to eV':
            ax.invert_xaxis()
            fig.show()
        # elif xscale == 'eV to nm':
        #     ax.invert_xaxis()
        #     fig.show()

        if norm:
            ax.set_ylabel('Normalized intensity')
            if norm == 'norm_max':
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
                ax.set_ylim(top=1, bottom=0)

        ax.legend()
        # no fixed color: give a different linestyle to lines that share
        # color and linestyle, and remember label -> color
        if self.color == None:
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

            for line1 in ax.lines:
                for line2 in ax.lines:
                    color1 = line1.get_color()
                    linestyle1 = line1.get_linestyle()
                    color2 = line2.get_color()
                    linestyle2 = line2.get_linestyle()
                    if mcolors.same_color(color1, color2) == True \
                            and linestyle1 == linestyle2 \
                            and line1 != line2:
                        linestyle_index = (cs.linestyles.index(
                            linestyle2) + 1) % len(cs.linestyles)
                        line2.set_linestyle(cs.linestyles[linestyle_index])

            ax.set_xmargin(0)
            ax.set_ymargin(0)
            # ax.autoscale(axis='x', tight=True)
        else:
            colors = []
            labels = []
            for line in ax.lines:
                c = line.get_color()
                l = line.get_label()
                colors.append(c)
                labels.append(l)
            for l, c in zip(labels, colors):
                self.label_color_dict[l] = c

        if xlims is not None:
            x = ax.set_xlim(left=xlims[0], right=xlims[1])
        return fig, ax
