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
from . import metadata


class Data(DataPlotMixin, DataRamanMixin, DataPLMixin, DataTRPLMixin, DataFitMixin):
    """One spectrum, or one spectrum of a map or TRPL series.

    Main attributes
    ---------------
    name : str
    x, y : array
        Current data (may be shifted or dark-subtracted).
    x_raw, y_raw : array
        Data as loaded.
    info : pandas Series or None
        Original metadata: lab-register row (Raman), parsed file name (TRPL)
        or file header (absorption).
    meta : dict
        The same metadata with standard keys ('sample', 'meas_type', 'ecc',
        't_start', ...), numbers converted; see
        :mod:`spectroscopy_analysis.metadata`.
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
                 meas_type=None, ecc=None, meta=None):
        """Store the data, detect the measurement type from ``info`` and remove
        negative spikes.

        The metadata in ``info`` (a lab-register row, the TRPL dict from the
        file name, or a header dict) are converted to standard fields in
        ``self.meta`` (see :mod:`spectroscopy_analysis.metadata`). The type
        comes from them: a register with a filter column ('ND FILT') -> LabRAM
        Raman, with a resolution column ('RES') -> FT-Raman, or the register's
        type column if it has one; TRPL dict -> 'trpl'; header with
        'Absorbance' -> 'absorption'; otherwise 'generic'.

        Parameters
        ----------
        meas_type : str, optional
            Force the measurement type ('Raman', 'absorption', 'generic', ...)
            instead of detecting it from ``info``.
        ecc : float, optional
            Laser wavelength (nm), needed for cm-1 <-> nm/eV conversions.
        meta : dict, optional
            Metadata fields to add or correct, with the standard names of
            :mod:`spectroscopy_analysis.metadata` (e.g. ``{'meas_type': 'PL',
            'sample': 'A', 'ecc': 325}``). They win over the fields read from
            ``info``; ``meas_type`` and ``ecc`` win over ``meta``.
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

        # ---- Metadata
        # self.info: the original metadata, as in version 1.0.0 (register row,
        # or a Series built from the TRPL / header dict).
        # self.meta: the same, with standard keys (see the metadata module).
        from_frame = isinstance(info, pd.DataFrame)
        if isinstance(info, dict):
            info = (pd.DataFrame.from_dict(info, orient='index') if info
                    else pd.DataFrame({0: pd.Series(dtype=object)}))
        if isinstance(info, pd.DataFrame):
            info = info[0]
        self.info = info

        detected = metadata.standardize(info)
        if detected.get('meas_type') == 'absorption':
            try:                       # absorption: the first header line is the sample
                self.info.SAMPLE = self.info.Line_1
            except Exception:
                pass
        if from_frame and detected.get('meas_type') is None:
            detected['meas_type'] = 'trpl'          # merged TRPL series

        # register row: the fields become attributes (d.n, d.sample, d.ecc,
        # d.ndfilt, ...) with the values of the register, as in version 1.0.0
        if detected.get('source') == 'register':
            where = metadata.register_columns(info.index)
            for key, w in where.items():
                if w is None or key == 'meas_type':
                    continue
                value = metadata.register_value(info, w)
                if key == 'ndfilt':
                    try:
                        value = value * 100     # register fraction -> %
                    except TypeError:
                        pass
                setattr(self, key, value)
        elif detected.get('sample') is not None:
            self.sample = detected['sample']

        # fields given by the user win over the detected ones
        user = dict(meta or {})
        if meas_type is not None:
            user['meas_type'] = meas_type
        if ecc is not None:
            user['ecc'] = ecc
        self.meta = {**detected, **user}
        if user:
            self.meta['source'] = (detected.get('source', '') + '+user').lstrip('+')
        for key, value in user.items():
            if key in metadata.REGISTER_ALIASES and key != 'meas_type':
                setattr(self, key, value)

        self.meas_type = self.meta.get('meas_type') or 'generic'
        self.meta['meas_type'] = self.meas_type

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
