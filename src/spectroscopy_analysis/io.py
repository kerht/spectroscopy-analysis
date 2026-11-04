"""Loaders: lab register, Raman/PL text exports, maps, TRPL, absorption, generic CSV."""

import numpy as np
import os
import pandas as pd
from termcolor import colored

from .trpl import rescale_factor
from .data import Data


class ProjectIOMixin:
    """Loader methods of :class:`Project` (mixed into it, not used alone)."""

    def get_date(self, folder):
        """Return the date part of a folder name.

        Folders are named ``YYMMDD`` or ``YYMMDD <user>``; the date must be
        separated from any further text by a space.
        """
        date = folder.split()[0]
        # date must be separated with space from the "user" name

        return date

    def open_table(self, folder):
        """Read the lab register ``reg_mis_<date>.xlsx`` of a measurement folder.

        Cleans the measurement-number column 'N' (keeps digits only, as int) and
        the column names, then stores the table in ``self.tables[date]``.

        Returns
        -------
        pandas.DataFrame
            One row per measurement (row N-1 describes measurement N).
        """
        date = self.get_date(folder)
        table = pd.read_excel(folder + os.sep + 'reg_mis_%s.xlsx' % date)
        # keep only the digits of the measurement number (e.g. 'N12' -> 12)
        table['N'] = table['N'].apply(
            lambda x: ''.join(filter(str.isdigit, str(x))))
        # Converts elements of 'N' column to integer
        table['N'] = table['N'].astype(int)
        # clean column names: remove line breaks and replace the Symbol-font
        # micro sign (U+F06D) with 'u'
        table = table.rename(columns={col:
                                      col.replace('\r', ' ').replace('',
                                                                     'u')
                                      for col in table.columns})
        self.tables[date] = table

        return table

    def open_data(self, folder, number, dark=False, color=None, label=None):
        """Load one Raman measurement by its number in the lab register.

        The data are read from the text export ``<date><NN>.txt``; the type of the
        original file in the folder decides how it is read:
        ``.l6s`` (LabSpec single spectrum) and ``.0`` (FT-Raman) -> spectrum,
        ``.l6m`` (LabSpec map) -> list of spectra.

        Parameters
        ----------
        folder : str
            Measurement folder (e.g. '231211').
        number : int
            Measurement number (column 'N' of the register).
        dark : bool or str, optional
            True if this measurement is a dark; a string is the name of the dark
            measurement to load and subtract later.
        color, label : optional
            Plot color and legend label.

        Returns
        -------
        Data, list of Data, or None
            None if no matching file is found.
        """
        ls = sorted(os.listdir(folder))
        date = self.get_date(folder)
        table = self.open_table(folder)
        # register row of this measurement (row N-1 describes measurement N)
        info = table.iloc[number-1]
        self.info = info
        file = os.sep.join([folder, date + '%02i' % number + '.txt'])
        # the data are always read from the .txt export; the original file
        # type (.l6s single spectrum, .l6m map, .0 FT-Raman) decides how
        if date + '%02i' % number + '.l6s' in ls:
            data = self.open_spectrum(file, info, data=self.data, dark=dark,
                                      color=color, label=label)
        elif date + '%02i' % number + '.l6m' in ls:
            data = self.open_map(file, info, data=self.data, dark=dark,
                                 color=color, label=label)
        elif date + '%02i' % number + '.0' in ls:
            data = self.open_spectrum(file, info, data=self.data, dark=dark,
                                      color=color, label=label)
        else:
            print('\nError!',
                  'File %s not found in %s.' % (file.split(os.sep)[-1],
                                                folder))
            data = None
        return data

    def open_spectrum(self, file, info, data, dark=False, color=None, label=None):
        """Read a two-column text file (x, y) into a :class:`Data` object.

        If ``dark`` is a string, the dark spectrum ``<date>/<dark>.txt`` is loaded
        too and stored in ``self.data`` as a dark Data, so that
        :meth:`Data.subtract_dark` can use it.

        Returns
        -------
        Data
        """

        # reads the file and save it in "data"
        name = file.split(os.sep)[-1].split('.')[0]
        data = np.genfromtxt(file)
        x = data[:, 0]
        y = data[:, 1]
        # dark given by name: load it too, so subtract_dark() can find it
        if dark != False and dark != True:
            date = file.split(os.sep)[-1].split('.')[0][0:6]
            d = os.sep.join([date, dark])
            data_dark = np.genfromtxt(d + '.txt')
            x_dark = data_dark[:, 0]
            y_dark = data_dark[:, 1]
            dark_name = dark
            dark_data = Data(dark, x_dark, y_dark, info, data=self.data, dark=True,
                             color=color, label=label)
            self.data[dark_name] = dark_data
            data = Data(name, x, y, info, data=self.data, dark=False,
                        color=color, label=label)
        else:
            data = Data(name, x, y, info, data=self.data, dark=dark,
                        color=color, label=label)

        self.data[name] = data
        # self.data[name].type = 'raman'
        return data

    def open_map(self, file, info, data, dark=False, color=None, label=None):
        """Read a LabSpec map export (e.g. a z-scan) into one Data per spectrum.

        Expected layout: first row = x axis; each following row = one spectrum,
        with its coordinate in the first column. Spectra are named
        ``<name>_1``, ``<name>_2``, ...

        Returns
        -------
        list of Data
        """

        # reads the file and save it in "data"
        name = file.split(os.sep)[-1].split('.')[0] + '_' + str(1)
        x = np.genfromtxt(file, max_rows=1)
        data = np.genfromtxt(file, skip_header=1)[:, 1:]
        data = [data[i, :] for i in range(data.shape[0])]

        data_map = []
        for i, y in enumerate(data):
            name = file.split(os.sep)[-1].split('.')[0] + '_' + str(i+1)
            data_map.append(Data(name, x, y, info, data=None, dark=dark,
                                 color=color, label=label))

            self.data[name] = data_map[-1]
            # data_map.data= self.data
        for i in data_map:
            i.data = self.data
        return data_map

    def open_folder(self, folder, exclude=None, dark_number=None, color=None,
                    label=None, meas_type='Raman', instrument='jasco'):
        """Load all measurements of a folder.

        Parameters
        ----------
        folder : str
            Measurement folder.
        exclude : list, optional
            Measurement numbers (Raman) or file names (absorption, trpl) to skip.
        dark_number : int, optional
            Raman only: number of the dark measurement.
        color, label : optional
            Raman and absorption: plot color (and Raman label) for all spectra.
        meas_type : {'Raman', 'trpl', 'absorption'}
            'Raman'      : every measurement listed in the lab register.
            'trpl'       : every .txt file named
                           ``wafer_sample_meastype_tstart_tstop_CCDx_Gx_prognum``,
                           one Project per file, grouped by wafer (first field of
                           the name) and gate: G1 = 'short', G3 = 'long', other
                           gates = 'other'. Files whose name does not follow the
                           format are skipped. Rescaling factors between the
                           short and long series of each sample are stored in
                           ``self.rescale[wafer][sample]``.
            'absorption' : every .txt file (any case), read with
                           :meth:`open_abs`.
        instrument : {'jasco', 'avantes'}, optional
            Absorption only: passed to :meth:`open_abs`. Default 'jasco'.

        Returns
        -------
        Raman / absorption : the last loaded Data.
        trpl : dict
            ``{wafer: {'short': {name: Project}, 'long': {name: Project},
            'other': {name: Project}}}``.
        """
        exclude = exclude or []

        # ---- Raman: every measurement listed in the lab register
        if meas_type == 'Raman':
            table = self.open_table(folder)

            for number in table['N']:
                if number not in exclude:
                    if (number == dark_number):
                        data = self.open_data(folder, dark_number, dark=True,
                                              color=color, label=label)
                    else:
                        data = self.open_data(folder, number, dark=False,
                                              color=color, label=label)

            return data

        # ---- TRPL: group files by wafer and gate (G1 = short, G3 = long)
        if meas_type == 'trpl':
            from .project import Project      # here, to avoid a circular import

            groups = {}
            for f in sorted(os.listdir(folder)):
                name, ext = os.path.splitext(f)
                if ext.lower() != '.txt' or name in exclude:
                    continue
                parts = name.split('_')
                if len(parts) < 8:            # not wafer_sample_..._Gx_prognum
                    continue
                wafer, sample = parts[0], parts[0] + '_' + parts[1]
                gate = {'G1': 'short', 'G3': 'long'}.get(parts[6], 'other')
                proj = Project()
                proj.sample = sample
                proj.open_pl(folder, name, label='sample')
                groups.setdefault(wafer, {'short': {}, 'long': {}, 'other': {}})
                groups[wafer][gate][name] = proj

            # rescaling factor between the short and long series of each sample
            self.rescale = {}
            for wafer, g in groups.items():
                self.rescale[wafer] = {}
                for p1 in g['short'].values():
                    for p2 in g['long'].values():
                        if p1.sample == p2.sample:
                            try:
                                self.rescale[wafer][p2.sample] = rescale_factor(p1, p2)
                            except Exception as e:
                                print("Error occurred in rescaling factor of ",
                                      colored(f"'{p1.sample}'", 'cyan',
                                              attrs=['bold']), ":", type(e), e)
            return groups

        # ---- Absorption: every .txt file (any case)
        if meas_type == 'absorption':

            ls = sorted(os.listdir(folder))
            names = []
            for i in ls:
                stem, ext = os.path.splitext(i)       # also .TXT, and dots in names
                if ext.lower() == '.txt':
                    names.append(stem)

            data = None                               # returned if no file is loaded
            for i in names:
                if i not in exclude:
                    data = self.open_abs(folder, i, color=color, instrument=instrument)
            return data

    def open_abs(self, folder, name, dark=False, color=None, label=None, instrument='jasco'):
        """Load an absorption spectrum exported by a Jasco or Avantes spectrometer.

        The x axis is converted from wavelength (nm) to energy (eV); the
        absorbance is not rescaled. Header (and Jasco footer) lines are stored in
        ``self.info``.

        Parameters
        ----------
        folder, name : str
            Folder and file name without extension. The extension is
            ``.txt`` in any case (``.txt``, ``.TXT``, ...).
        instrument : {'jasco', 'avantes'}
            'jasco'   : tab separated, decimal comma, 19 header and 44 footer
                        lines, cp1252 encoding.
            'avantes' : ';' separated; the number of header lines is detected
                        with :meth:`read_file` (8 in the AvaSoft exports:
                        title, settings, column names, units, blank line).

        Returns
        -------
        Data
            With ``type = 'absorption'`` and ``meas_type = 'absorption'``.
        """
        info = {}
        file = self._find_txt(folder, name)

        # both instruments: x converted from nm to eV (absorbance not rescaled)
        if instrument == 'avantes':
            # where the numbers start is detected, not assumed: a fixed
            # skip_header=6 read the units row as a NaN point
            report = self.read_file(file, verbose=False)
            data = np.genfromtxt(file, skip_header=report['header_lines'],
                                 delimiter=';', encoding=report['encoding'])
            x = data[:, 0]
            x = 1239.8/x
            y = data[:, -1]                    # last column: absorbance
            for i, line in enumerate(report['header']):
                line = line.strip()
                if not line:
                    continue
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
                else:
                    # Use a generic key for non key:value lines (e.g., title)
                    info[f"Line_{i+1}"] = line

        elif instrument == 'jasco':

            with open(file, 'r', encoding='cp1252') as f:
                lines = f.readlines()

            # --- Parse header (first 19 lines) ---
            for i in range(19):
                line = lines[i].strip()
                if '\t' in line:
                    key, value = line.split('\t', 1)
                    info[key.strip()] = value.strip()
                else:
                    info[f"Line_{i+1}"] = line

            # --- Parse footer (last 44 lines) ---
            footer_start = len(lines) - 44
            for j in range(footer_start, len(lines)):
                line = lines[j].strip()
                if '\t' in line:
                    key, value = line.split('\t', 1)
                    info[key.strip()] = value.strip()
                elif line:
                    info[f"Line_{j}"] = line

            data = pd.read_csv(file,
                               encoding='cp1252',     # or try 'latin1' if needed
                               sep='\t',              # tab-separated values
                               header=18,             # start after metadata
                               skipfooter=44,         # skip footer
                               engine='python',       # required for skipfooter
                               decimal=',',           # comma as decimal separator
                               names=["X", "Y"]
                               )
            data = data.to_numpy()
            x = data[:, 0]
            x = 1239.8/x
            y = data[:, -1]

            # x = data.iloc[:,0]
            # y = data.iloc[:,-1]
            # x = 1239.8/x

        self.info = info

        data = Data(name, x, y, info=self.info, data=None, dark=False,
                    color=color, label=label, meas_type='absorption')

        self.data[name] = data
        self.data[name].type = 'absorption'

        return data

    @staticmethod
    def _find_txt(folder, name):
        """Path of ``folder/name.txt``, matching the extension in any case.

        Windows and macOS ignore the case of file names, Linux does not: an
        AvaSoft export ``name.TXT`` must still be found as ``name``.
        """
        exact = os.path.join(folder, name + '.txt')
        if os.path.isfile(exact):                  # usual case, as before
            return exact
        for f in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(f)
            if stem == name and ext.lower() == '.txt':
                return os.path.join(folder, f)
        raise FileNotFoundError(f"{name}.txt (any case) not found in {folder}")

    def open_pl(self, folder, name, dark=False, color=None, label=None):
        """Load a TRPL (time-resolved PL) file.

        File names must follow
        ``wafer_sample_mestype_tstart_tstop_CCDx_Gx_prognum`` (exported with the
        "Multiple column (header)" and "Comma" delimiter options); the parts are
        parsed into an info dict (``t_start``, ``t_stop``, ``g_width``, ...).

        - 2 columns : a single spectrum -> one Data.
        - otherwise : one spectrum per gate delay. Row 0 holds the wavelengths
          (columns 2:-1), rows 2 onwards the spectra. One Data per delay, named
          ``<name>_1``, ``<name>_2``, ...; each gets ``data_all`` (all spectra of
          the series), needed to build the decay curve.

        Parameters
        ----------
        label : str, optional
            'sample' uses "wafer sample" from the file name as legend label.

        Returns
        -------
        Data or list of Data
        """
        # Save files with the following format:
        # "wafer_sample_mestype_tstart_tstop_CCDx_Gx_prognum"
        # export with "Multiple column (header)" & Delimiter "Comma" options

        ls = sorted(os.listdir(folder))
        names = []
        for i in ls:
            if i.split('.')[-1] == 'txt':
                names.append(i)

        file = os.sep.join([folder]+[name+'.txt'])
        data = pd.read_csv(file)
        info = None

        if label == 'sample':
            try:
                parts = name.split('_')
                label = parts[0] + ' ' + parts[1]
            except Exception as e:
                label = None
                print("Error occurred in method", colored('open_pl', 'red',
                                                          attrs=['bold']), "file", colored(f"'{name}'", 'cyan',
                                                                                           attrs=['bold']), ":", type(e), e)
        else:
            label = label

        # two columns: single spectrum; otherwise one spectrum per gate delay
        if data.shape[1] == 2:
            data = data.to_numpy()
            x = data[:, 0]
            y = data[:, 1]
            info = {}
            try:
                info['wafer'] = name.split('_')[0]
                info['samp'] = name.split('_')[1]
                info['mes_type'] = name.split('_')[2]
                info['t_start'] = name.split('_')[3]
                info['t_stop'] = name.split('_')[4]
                info['CCD'] = name.split('_')[5]
                info['g_width'] = name.split('_')[6].split('G')[-1]
                info['prog_num'] = name.split('_')[-1]
                info['data'] = folder.split('/')[0]
                info['position'] = info['samp'].split('#')[-1]
            except Exception as e:
                print("Error occurred in method", colored('open_pl', 'red',
                                                          attrs=['bold']), "file", colored(f"'{name}'", 'cyan',
                                                                                           attrs=['bold']), ":", type(e), e)

            data = Data(name, x, y, info=info, data=self.data, dark=False,
                        color=color, label=label)

            self.data[name] = data
            self.data[name].type = 'trpl'
            self.info = pd.DataFrame.from_dict(info, orient='index')

            return data

        else:
            x = data.iloc[0, 2:-1]
            y = [0] * (data.shape[0] - 2)
            data_map = []
            info = {}
            try:
                info['wafer'] = name.split('_')[0]
                info['samp'] = name.split('_')[1]
                info['mes_type'] = name.split('_')[2]
                info['t_start'] = name.split('_')[3]
                info['t_stop'] = name.split('_')[4]
                info['CCD'] = name.split('_')[5]
                info['g_width'] = name.split('_')[6].split('G')[-1]
                info['prog_num'] = name.split('_')[-1]
                info['data'] = folder.split('/')[0]
                info['position'] = info['samp'].split('#')[-1]
            except Exception as e:
                print("Error occurred in method", colored('open_pl', 'red',
                                                          attrs=['bold']), "file", colored(f"'{name}'", 'cyan',
                                                                                           attrs=['bold']), ":", type(e), e)

            for i in range(data.shape[0]-2):
                y[i] = data.iloc[i+2, 2:-1]
                name = file.split(os.sep)[-1].split('.')[0] + '_' + str(i+1)
                data_map.append(Data(name, x, y[i], info=info, data={},
                                     dark=False, color=color, label=label))
                self.data[name] = data_map[-1]
                data_map[-1].data = {name: self.data[name]}
                self.data[name].type = 'trpl'
                self.info = pd.DataFrame.from_dict(info, orient='index')
            for i in data_map:
                # dictionary needed for integrations
                i.data_all = self.data

            return data_map

    @staticmethod
    def read_file(path, n_preview=8, verbose=True):
        """Inspect a text/CSV data file and report its structure.

        Nothing is converted into Data here: the report tells open_file (or
        you) how to read the numbers. Static method: it can be called without
        a project, ``Project.read_file('file.csv')``.

        Parameters
        ----------
        path : str
            File to inspect.
        n_preview : int
            Number of lines of the file to show/return.
        verbose : bool
            Print the report.

        Returns
        -------
        dict with keys
            encoding, line_endings, delimiter, decimal, header_lines,
            footer_lines, comment_char, layout ('y', 'xy', 'multi',
            'map', 'transposed'), n_rows, n_columns, column_names, units,
            label_columns (map only), header (list of header lines),
            preview (first lines of the file).
        """
        # ---- 1. Encoding and line endings
        with open(path, 'rb') as fh:
            raw = fh.read()
        if raw.startswith(b'\xef\xbb\xbf'):
            encoding = 'utf-8-sig'                 # Excel "CSV UTF-8" (BOM)
        else:
            try:
                raw.decode('utf-8')
                encoding = 'utf-8'
            except UnicodeDecodeError:
                encoding = 'cp1252'                # Windows instruments (Jasco, ...)
        text = raw.decode(encoding)
        line_endings = 'CRLF' if '\r\n' in text else 'LF'
        lines = text.splitlines()
        while lines and not lines[-1].strip():     # drop blank lines at the end
            lines.pop()

        # ---- 2. Helpers: split a line and test if a token is a number
        def split(line, delim):
            tokens = line.split() if delim is None else line.split(delim)
            return [t.strip().strip('"').strip("'").strip() for t in tokens]

        def is_number(token, decimal):
            if decimal == ',':
                token = token.replace(',', '.')
            try:
                float(token)
                return True
            except ValueError:
                return False

        def numeric_fraction(tokens, decimal):
            """Fraction of fields that are numbers or empty (missing values)."""
            good = [t == '' or t.lower() == 'nan' or is_number(t, decimal) for t in tokens]
            real = [is_number(t, decimal) for t in tokens]
            return sum(good) / len(tokens) if any(real) else 0.0

        # ---- 3. Try each delimiter/decimal pair; keep the one that gives the
        # longest block of consecutive numeric lines with a constant number of
        # fields (the data block).
        candidates = [('\t', '.'), ('\t', ','), (';', '.'), (';', ','),
                      (',', '.'), (None, '.'), (None, ',')]   # None = any whitespace
        best = None
        for delim, decimal in candidates:
            run_start, run_len, run_cols, frac_sum = 0, 0, 0, 0.0
            cur_start, cur_len, cur_cols, cur_frac = 0, 0, None, 0.0
            for i, line in enumerate(lines + ['']):           # '' closes the last run
                tokens = split(line, delim) if line.strip() else []
                frac = numeric_fraction(tokens, decimal) if tokens else 0.0
                if frac >= 0.5 and len(tokens) == cur_cols:
                    cur_len += 1
                    cur_frac += frac
                else:
                    if cur_len > run_len:
                        run_start, run_len, run_cols, frac_sum = cur_start, cur_len, cur_cols, cur_frac
                    if frac >= 0.5:
                        cur_start, cur_len, cur_cols, cur_frac = i, 1, len(tokens), frac
                    else:
                        cur_start, cur_len, cur_cols, cur_frac = i, 0, None, 0.0
            # rows x average numeric fraction; ties -> more columns
            score = (frac_sum, run_cols or 0)
            if best is None or score > best['score']:
                best = dict(score=score, delim=delim, decimal=decimal,
                            start=run_start, n_rows=run_len, n_cols=run_cols)

        if not best or best['n_rows'] == 0:
            raise ValueError(f'{path}: no block of numeric data found')

        delim, decimal = best['delim'], best['decimal']
        start, n_rows, n_cols = best['start'], best['n_rows'], best['n_cols']
        end = start + n_rows
        block = [split(l, delim) for l in lines[start:end]]
        header = lines[:start]
        footer = lines[end:]

        # ---- 4. Layout
        first = block[0]
        lead = 0                                   # leading non-numeric cells of row 0
        while lead < len(first) and not is_number(first[lead], decimal):
            lead += 1
        others_numeric = all(is_number(r[0], decimal) for r in block[1:])
        if n_rows <= 3 and n_cols > 10:
            layout = 'transposed'                  # data stored by rows
        elif lead > 0 and n_rows > 1 and others_numeric:
            layout = 'map'                         # row 0 = x axis
        elif n_cols == 1:
            layout = 'y'                           # intensities only, no x column
        elif n_cols == 2:
            layout = 'xy'
        else:
            layout = 'multi'

        # ---- 5. Column names (and units): non-numeric header lines just above
        # the data with the same number of fields
        names_rows = []
        for line in reversed(header):
            tokens = split(line, delim)
            if len(tokens) == n_cols and numeric_fraction(tokens, decimal) == 0:
                names_rows.insert(0, tokens)
            else:
                break
        column_names = names_rows[0] if names_rows else None
        units = names_rows[1] if len(names_rows) > 1 else None

        comment_char = '#' if header and all(l.startswith('#') for l in header if l.strip()) else None

        report = dict(
            file=path, encoding=encoding, line_endings=line_endings,
            delimiter={'\t': 'tab', None: 'whitespace'}.get(delim, delim),
            decimal=decimal, header_lines=start, footer_lines=len(footer),
            comment_char=comment_char, layout=layout,
            n_rows=n_rows, n_columns=n_cols,
            column_names=column_names, units=units,
            label_columns=lead if layout == 'map' else 0,
            header=header, preview=lines[:n_preview],
        )

        if verbose:
            print(f"File: {path}")
            print(f"  encoding {encoding}, line endings {line_endings}")
            print(f"  delimiter {report['delimiter']!r}, decimal {decimal!r}")
            print(f"  header lines {start}, footer lines {len(footer)}"
                  + (f", comment char {comment_char!r}" if comment_char else ''))
            print(f"  layout '{layout}': {n_rows} rows x {n_cols} columns"
                  + (f", {lead} label column(s)" if layout == 'map' else ''))
            if column_names:
                print(f"  column names: {column_names}" + (f"  units: {units}" if units else ''))
            print("  first lines:")
            for l in report['preview']:
                print('    ' + (l if len(l) < 90 else l[:87] + '...'))
        return report

    def open_file(self, file, meas_type='generic', ecc=None, color=None,
                  label=None, verbose=True, sort=False, **overrides):
        """Open any text/CSV data file as Data object(s).

        The structure is detected with :meth:`read_file`; the numbers are read
        according to the detected layout:
          'y'          : one column -> one Data, x = point index
          'xy'         : x + y -> one Data
          'multi'      : x + several y columns -> one Data per column
          'map'        : row 0 = x, each next row a spectrum -> one Data per row
          'transposed' : data stored by rows -> treated as 'xy'/'multi'

        Points are kept in the order of the file (see ``sort``). Missing
        values (empty cells, NaN) are dropped, because a single NaN makes
        max(), the integrals and the spike check return NaN. Units and axis
        labels are read from the
        column names (e.g. 'Raman shift (cm-1)', 'Wavelength [nm]'), from a
        units row, from a '# Columns: ...' comment or from XUNITS/YUNITS
        header keys; they are stored in ``x_unit``, ``y_unit``, ``x_label``,
        ``y_label`` and used by :meth:`Data.plot`. Header lines go into
        ``info`` (as key/value when possible). No dark subtraction is applied
        (``type`` is set to 'data').

        Parameters
        ----------
        file : str
            Path of the file.
        meas_type : str
            Measurement type given to Data ('Raman', 'absorption', 'generic',
            ...). Decides how plot() and the other methods treat the data.
            Avoid 'trpl': those methods need the TRPL file-name fields.
        ecc : float, optional
            Laser wavelength (nm), needed for cm-1 <-> nm/eV conversions.
        color, label : optional
            Plot color and legend label (label only for single spectra;
            multi-column files and maps use the column names/coordinates).
        verbose : bool
            Print the read_file report.
        sort : bool
            Sort each spectrum by increasing x. Default False (file order).
            Note: with decreasing x (e.g. Jasco exports, eV axes),
            :meth:`Data.integrate` returns a negative value; use sort=True or
            :meth:`Data.integrate_window`, which takes the absolute value.
        **overrides
            Correct any detected value of the report, e.g. ``header_lines=3``,
            ``delimiter=';'``, ``decimal=','``, ``layout='multi'``,
            ``column_names=[...]``, ``units=[...]``. Two extra keys set the
            axes directly: ``x_unit='nm'``, ``x_label=...``, ``y_unit=...``,
            ``y_label=...``.

        Returns
        -------
        Data, or list of Data for multi-column files and maps. They are
        also stored in ``self.data``.
        """
        import re

        def parse(file, r):
            """Read the numbers; return x and a list of (label, y) pairs."""
            sep = {'tab': '\t', 'whitespace': r'\s+'}.get(r['delimiter'], r['delimiter'])
            df = pd.read_csv(file, sep=sep, decimal=r['decimal'], encoding=r['encoding'],
                             skiprows=r['header_lines'], skipfooter=r['footer_lines'],
                             header=None, engine='python', quotechar='"',
                             skip_blank_lines=True)
            df = df.dropna(how='all')               # blank lines at the end
            layout = r['layout']

            if layout == 'transposed':              # rows -> columns, drop label cell
                df = df.T.iloc[1:].reset_index(drop=True)
                layout = 'xy' if df.shape[1] == 2 else 'multi'

            if layout == 'y':                       # intensities only
                y = pd.to_numeric(df.iloc[:, 0], errors='coerce').to_numpy(dtype=float)
                return np.arange(len(y), dtype=float), [('y', y)]

            arr = df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
            if layout in ('xy', 'multi'):           # column 0 = x
                names = r.get('column_names') or [f'y{i}' for i in range(arr.shape[1])]
                return arr[:, 0], [(names[i], arr[:, i]) for i in range(1, arr.shape[1])]
            if layout == 'map':                     # row 0 = x after the label columns
                lead = r.get('label_columns', 1)
                def fmt(v):                         # -10.0 -> '-10'
                    try:
                        return f'{float(v):g}'
                    except (TypeError, ValueError):
                        return str(v)
                return arr[0, lead:], [(fmt(df.iloc[i, lead - 1]), arr[i, lead:])
                                       for i in range(1, arr.shape[0])]
            raise ValueError(f"unknown layout {layout!r}")

        def split_label(text):
            """'Raman shift (cm-1)' -> ('Raman shift', 'cm-1')."""
            if not text:
                return None, None
            m = re.match(r'^\s*(.*?)\s*[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$', str(text))
            if m:
                return (m.group(1) or None), m.group(2)
            return str(text).strip() or None, None

        def norm_unit(unit):
            """Recognize the common x units; leave the others as written."""
            if not unit:
                return None
            u = (unit.strip().lower().replace(' ', '').replace('^', '')
                 .replace('$', '').replace('{', '').replace('}', '').replace('⁻¹', '-1'))
            if u in ('cm-1', '1/cm', 'cm**-1', 'wavenumbers', 'wavenumber'):
                return 'cm-1'
            if u in ('nm', 'nanometers', 'nanometer', 'nanometres'):
                return 'nm'
            if u in ('ev', 'electronvolt', 'electronvolts'):
                return 'eV'
            return unit.strip()

        def make_label(quantity, unit, default_quantity):
            if unit == 'cm-1':
                unit = 'cm$^{-1}$'
            quantity = quantity or default_quantity
            if quantity and unit:
                return f'{quantity} ({unit})'
            return quantity or (f'({unit})' if unit else None)

        # ---- detect the structure, apply manual corrections, read the numbers
        axis_keys = ('x_unit', 'x_label', 'y_unit', 'y_label')
        axis_over = {k: overrides.pop(k) for k in axis_keys if k in overrides}
        report = self.read_file(file, verbose=verbose)
        report.update(overrides)
        x, spectra = parse(file, report)

        # ---- header lines -> info (key/value when possible) + SAMPLE
        base = os.path.splitext(os.path.basename(file))[0]
        info = {'file': file, 'SAMPLE': base}
        for i, line in enumerate(report['header']):
            text = line.lstrip('#').strip()
            for sep_kv in ('\t', ':', '='):
                if sep_kv in text:
                    key, value = text.split(sep_kv, 1)
                    if key.strip():
                        info[key.strip()] = value.strip()
                        break
            else:
                if text:
                    info[f'Line_{i+1}'] = text

        # ---- axis names and units
        n_cols = report['n_columns']
        names = report.get('column_names')
        units = report.get('units')
        if not names:      # e.g. '# Columns: energy (eV), intensity (a.u.)'
            for line in report['header']:
                text = line.lstrip('#').strip()
                text = text.split(':', 1)[1] if ':' in text else text
                parts = [p.strip() for p in text.split(',')]
                if len(parts) == n_cols and all(split_label(p)[1] for p in parts):
                    names = parts
                    break
        x_q, x_u = split_label(names[0]) if names else (None, None)
        y_q, y_u = (split_label(names[1]) if names and len(names) > 1
                    and report['layout'] in ('xy', 'transposed') else (None, None))
        if units:
            x_u = units[0] or x_u
            if len(units) > 1 and report['layout'] in ('xy', 'transposed'):
                y_u = units[1] or y_u
        upper = {k.upper(): v for k, v in info.items()}   # JCAMP/Jasco style
        if not x_u and upper.get('XUNITS'):
            x_u = upper['XUNITS']
        if not y_q and not y_u and upper.get('YUNITS'):
            y_q = upper['YUNITS'].capitalize()
        x_u = norm_unit(x_u)
        default_x = {'cm-1': 'Raman shift', 'nm': 'Wavelength', 'eV': 'Energy'}.get(x_u)
        axes = {'x_unit': x_u, 'x_label': make_label(x_q, x_u, default_x),
                'y_unit': y_u, 'y_label': make_label(y_q, y_u, 'Intensity' if y_u else None)}
        if report['layout'] == 'y' and not axes['x_label']:
            axes['x_label'] = 'Point index'         # no x column in the file
        axes.update(axis_over)
        if 'x_unit' in axis_over and 'x_label' not in axis_over:
            u = norm_unit(axis_over['x_unit'])
            axes['x_unit'] = u
            axes['x_label'] = make_label(None, u, {'cm-1': 'Raman shift', 'nm': 'Wavelength',
                                                    'eV': 'Energy'}.get(u))

        # ---- one Data per spectrum
        created = []
        for col_label, y in spectra:
            ok = ~np.isnan(x) & ~np.isnan(y)        # drop missing values
            xs, ys = x[ok], y[ok]
            if sort:                                # increasing x, on request
                order = np.argsort(xs, kind='stable')
                xs, ys = xs[order], ys[order]
            name = base if len(spectra) == 1 else f'{base}_{col_label}'
            d = Data(name, xs, ys, info=dict(info), data=self.data,
                     color=color, label=label if len(spectra) == 1 else col_label,
                     meas_type=meas_type, ecc=ecc)
            d.type = 'data'                         # already processed: no dark
            for k, v in axes.items():
                setattr(d, k, v)
            self.data[name] = d
            created.append(d)
        return created[0] if len(created) == 1 else created
