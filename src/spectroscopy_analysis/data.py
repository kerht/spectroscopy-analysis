"""The Data class: one spectrum, with the methods of every technique."""

import numpy as np
import pandas as pd
from scipy.integrate import simpson, trapezoid
from termcolor import colored

from .raman import DataRamanMixin
from .pl import DataPLMixin
from .trpl import DataTRPLMixin
from .fitting import DataFitMixin
from .plotting import DataPlotMixin


class Data(DataPlotMixin, DataRamanMixin, DataPLMixin, DataTRPLMixin, DataFitMixin):
    """One spectrum, or one spectrum of a map or TRPL series.

    Main attributes
    ---------------
    name : str
    x, y : array
        Current data (may be shifted or dark-subtracted).
    x_raw, y_raw : array
        Data as loaded.
    info : pandas Series
        Metadata: lab-register row (Raman), parsed file name (TRPL) or file
        header (absorption).
    meas_type : {'Raman', 'FT-Raman', 'trpl', 'absorption', 'generic', ...}
    type : {'data_raw', 'data', 'dark'}
        'data_raw' becomes 'data' after dark subtraction.
    data : dict
        All Data of the same Project (used to find the dark spectrum).
    data_all : dict
        TRPL only: all spectra of the series (used to build the decay).
    ecc : float
        Raman: excitation wavelength (nm), from the register or given.
    x_unit, y_unit : str or None
        Units read from the file by :meth:`Project.open_file` ('cm-1', 'nm',
        'eV' are recognized for x); None for data loaded otherwise.
    x_label, y_label : str or None
        Axis labels read from the file; used by :meth:`plot` when the x axis
        is not converted.
    I, tau, fit_result
        Results of integration and fits.
    """

    def __init__(self, name, x, y, info, data, dark=False, color=None, label=None,
                 meas_type=None, ecc=None):
        """Store the data, detect the measurement type from ``info`` and remove
        negative spikes.

        The type is detected from ``info``: a lab-register row whose third column
        is 'ND FILT' -> LabRAM Raman, 'RES' -> FT-Raman; a dict -> TRPL, or
        absorption if it contains 'Absorbance'.

        Parameters
        ----------
        meas_type : str, optional
            Force the measurement type ('Raman', 'absorption', 'generic', ...)
            instead of detecting it from ``info``.
        ecc : float, optional
            Laser wavelength (nm), needed for cm-1 <-> nm/eV conversions.
        """

        self.name = name
        # self.date = name[:6]
        self.data = data
        # Always float NumPy arrays, whatever the loader passed (lists, pandas
        # Series, ...): TRPL spectra came as Series with a text index
        # ('1', '2', ...), so selecting a window by position failed (e.g.
        # plot(norm=[a, b])). np.array also makes a copy: each Data owns its
        # x and y, and the caller's arrays are never modified here.
        self.x_raw = np.array(x, dtype=float)
        self.y_raw = np.array(y, dtype=float)
        self.I = 0
        self.int_status = False
        self.time = None

        self.color = color
        self.label = label
        self.label_color_dict = {}  # Dictionary for label-color association
        self.shift = False
        # axis units/labels: set by Project.open_file, None otherwise
        self.x_unit = None
        self.y_unit = None
        self.x_label = None
        self.y_label = None

        # ---- Metadata from the lab register (Raman / FT-Raman)
        if info is not None and type(info) != dict:
            self.info = info
            # the third column of the register identifies the instrument
            if self.info.axes[0][2] == 'ND FILT':
                self.n = info.iloc[0]
                self.sample = info.iloc[1]
                self.ndfilt = info.iloc[2] * 100
                self.acqt = info.iloc[3]
                self.acc = info.iloc[4]
                self.det = info.iloc[5]
                self.obj = info.iloc[6]
                self.grat = info.iloc[7]
                self.ecc = info.iloc[8]
                self.hole = info.iloc[9]
                self.range = info.iloc[10]
                self.spectro = info.iloc[11]
                self.notes = info.iloc[12]
                self.meas_type = 'Raman'

            elif self.info.axes[0][2] == 'RES':
                self.n = info.iloc[0]
                self.sample = info.iloc[1]
                self.res = info.iloc[2]
                self.range = info.iloc[3]
                self.aper = info.iloc[4]
                self.scan_t = info.iloc[5]
                self.scan_v = info.iloc[6]
                self.pow = info.iloc[7]
                self.p_gain = info.iloc[8]
                self.low_pass = info.iloc[9]
                self.ph_res = info.iloc[10]
                self.pos = info.iloc[11]
                self.det = info.iloc[12]
                self.meas_type = 'FT-Raman'

        # if 'eV' in self.info:
        #     self.meas_type = 'PL'

        # else:
        #     self.info = None
            # print('Info = None')

        # ---- Metadata as dict: TRPL (parsed file name) or absorption (header)
        if type(info) == dict:
            self.info = pd.DataFrame.from_dict(info, orient='index')
            if 'wafer' in self.info[0]:
                # self.name = self.info[0].wafer+'_'+self.info[0].samp
                self.meas_type = 'trpl'
            try:
                if any(val in self.info[0].values for val in ['Absorbance', 'ABSORBANCE']):
                    # if 'Absorbance' in self.info[0].values or 'ABSORBANCE' in self.info[0].values:
                    self.meas_type = 'absorption'
                    self.info[0].SAMPLE = self.info[0].Line_1
            except Exception:
                pass
            self.info = self.info[0]

        elif isinstance(info, pd.DataFrame):
            self.meas_type = 'trpl'
            self.info = self.info[0]

        # ---- Explicit type / laser wavelength (e.g. from Project.open_file):
        # they win over the detection above
        if meas_type is not None:
            self.meas_type = meas_type
        elif not hasattr(self, 'meas_type'):
            self.meas_type = 'generic'      # nothing detected: avoid a crash below
        if ecc is not None:
            self.ecc = ecc

        self.x = self.x_raw

        # ---- Data type: raw data, dark, or (absorption) already processed
        if dark is False:
            if self.meas_type != 'absorption':
                self.type = 'data_raw'
            else:
                self.type = 'data'
            self.y = self.y_raw
            # try:
            #     if self.dark:
            #     pass
            # except AttributeError():
            #     pass
            # else:
            self.dark = np.zeros(len(self.y_raw))
        elif dark is True:
            self.type = 'dark'
            self.y = self.y_raw
            self.dark = self.y_raw
        else:
            self.type = 'Not assigned'
            self.y = self.y_raw
            print('\nMeasure %s:' % self.name, 'Data type not assigned.\n')

        # ---- Remove negative spikes: set 5 points on each side of the
        # minimum to 0 until no spike is left. Note: this changes y in place,
        # and y is the same array as y_raw.
        while self.check_spike():
            pts = 5
            i_spike = np.nanargmin(self.y)        # NaN-aware, like check_spike
            x = np.asarray(self.x)  # ensure ndarray

            left_idx = max(i_spike - pts + 1, 0)
            right_idx = min(i_spike + pts, len(x) - 1)

            # boolean mask for region outside spike
            roi = (x < x[left_idx]) | (x > x[right_idx])

            # zero-out spike region (y is always a NumPy array, see above)
            self.y[left_idx:right_idx + 1] = 0

    def get_roi(self, x_min, x_max):
        """Return x,y restricted to [x_min, x_max] as numpy arrays."""
        x = np.asarray(self.x, dtype=float)
        y = np.asarray(self.y, dtype=float)
        mask = (x >= x_min) & (x <= x_max)
        return x[mask], y[mask], mask

    def integrate_window(self, x_min, x_max, method='simps'):
        """Integrate y over x in [x_min, x_max] using simps or trapezoid."""
        x_cut, y_cut, _ = self.get_roi(x_min, x_max)
        if method == 'simps':
            return np.abs(simpson(y_cut, x_cut))
        elif method == 'trapz':
            return np.abs(trapezoid(y_cut, x_cut))
        else:
            raise ValueError("method must be 'simps' or 'trapz'")

    def check_spike(self):
        """Return True if the spectrum has a negative spike: a minimum below zero and
        more than 5 standard deviations below the mean.
        """
        # NaN-aware, like the pandas Series that TRPL spectra used to be:
        # NaN points are ignored, and an empty or all-NaN spectrum (e.g. a
        # non-spectrum file opened by open_pl) has no spike instead of
        # raising "zero-size array to reduction operation".
        if np.all(np.isnan(self.y)):          # also True for an empty array
            return False
        mean = np.nanmean(self.y)
        std = np.nanstd(self.y)
        y_min = np.nanmin(self.y)
        spike = (y_min < mean-5*std)*(y_min < 0)

        return spike

    def subtract_dark(self):
        """Subtract the dark spectrum, once.

        Subtracts the ``dark`` array of every dark measurement (type 'dark')
        in ``self.data``. Only runs on 'data_raw' and turns it into 'data'.
        Other spectra are skipped: their ``dark`` is all zeros, and one of a
        different length (e.g. a file opened with open_file) would otherwise
        stop the loop halfway.
        """
        data = self.data

        if (self.type == 'data_raw'):
            for j in data.values():
                if getattr(j, 'type', None) != 'dark':
                    continue
                try:
                    self.y = self.y - j.dark
                except Exception as e:
                    print("Error occurred in method", colored('subtract_dark', 'red',
                                                              attrs=['bold']), "file", colored(f"'{self.name}'", 'cyan',
                                                                                               attrs=['bold']), ":", type(e), e)
                    if type(e) == AttributeError:
                        print('Dark not assigned! \n')
                    return
            self.type = 'data' if self.type != 'dark' else self.type

    def subtract_baseline(self, x=0, y=0, x_range=[]):
        """Subtract a straight line through the points of (x, y) closest to
        ``x_range[0]`` and ``x_range[1]``. Works on the arguments, not on self.

        Returns
        -------
        ndarray
            Baseline-corrected y.
        """
        # Find indices for the start and end of the range
        idx1 = (np.abs(x - x_range[0])).argmin()
        idx2 = (np.abs(x - x_range[1])).argmin()

        x1, x2 = x[idx1], x[idx2]
        y1, y2 = y[idx1], y[idx2]

        m = (y2 - y1) / (x2 - x1)
        q = y1 - m * x1

        baseline = m * x + q
        y_corrected = y - baseline

        return y_corrected
