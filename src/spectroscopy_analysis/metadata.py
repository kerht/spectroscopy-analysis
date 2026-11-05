"""Measurement metadata: read it from any source, use it with standard names.

The package reads metadata from three kinds of sources:

- a **lab register** (``reg_mis_<date>.xlsx``, one row per measurement),
- a **file name** (TRPL: ``wafer_sample_meastype_tstart_tstop_CCDx_Gx_prognum``),
- a **file header** (absorption, generic text files).

Each source has its own names and layout, but the analysis only needs a few
fields. This module turns any source into one dictionary with **standard
keys** (``Data.meta``)::

    d.meta['sample'], d.meta['meas_type'], d.meta['ecc'], d.meta['t_start'], ...

Values are converted to numbers where they are numbers, and a missing field
is ``None``. The original metadata stay in ``Data.info``, unchanged.

Adapting to another lab
-----------------------
Register columns are found by **name**, through :data:`REGISTER_ALIASES`
(case, spaces, line breaks and units in brackets are ignored), so columns can
be added, moved or renamed. Add the names your register uses::

    import spectroscopy_analysis as sa
    sa.metadata.REGISTER_ALIASES['ecc'].append('Laser line')

TRPL file names are parsed with the regular expression :data:`TRPL_PATTERN`
(named groups become fields). Replace it, or pass ``pattern=`` to the
loaders::

    sa.metadata.TRPL_PATTERN = r'(?P<wafer>[^-]+)-(?P<samp>[^-]+)-...'

Any field can also be given, or corrected, when loading::

    proj.open_data('261001', 7, meta={'meas_type': 'PL'})
"""
import re
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- standard keys
#: Standard keys of ``Data.meta`` (other keys are allowed too).
STANDARD_KEYS = (
    'sample',      # sample name
    'meas_type',   # 'Raman', 'FT-Raman', 'PL', 'trpl', 'absorption', 'generic', ...
    'instrument',  # e.g. 'LabRAM', 'FT-Raman', 'jasco', 'avantes'
    'ecc',         # excitation (laser) wavelength, nm
    'n',           # measurement number in the register
    'wafer',       # TRPL: wafer
    'position',    # TRPL: position on the wafer (number after '#')
    't_start',     # TRPL: first gate delay
    't_stop',      # TRPL: last gate delay
    'g_width',     # TRPL: gate width (step between delays)
    'source',      # 'register', 'filename', 'header' or 'user'
)


# ----------------------------------------------------------- register aliases
#: Standard key -> column names that may hold it in a lab register.
#: Compared after :func:`normalize_name`, so 'Ecc (nm)', 'ECC\n(nm)' and
#: 'ecc' all match 'Ecc'. Lists are searched in order.
REGISTER_ALIASES = {
    'n':          ['N', 'No', 'Num', 'Number', 'Meas', 'Measurement'],
    'sample':     ['SAMPLE', 'Sample name', 'Campione'],
    'meas_type':  ['Type', 'Meas type', 'Measurement type', 'Tipo'],
    'ecc':        ['Ecc', 'Exc', 'Excitation', 'Laser', 'Lambda exc',
                   'Wavelength exc'],
    # LabRAM (Raman / PL)
    'ndfilt':     ['ND FILT', 'ND filter', 'Filter', 'Filtro'],
    'acqt':       ['Acq T', 'Acq time', 'Acquisition time', 'Exposure'],
    'acc':        ['Acc', 'Accumulations', 'Accumulation'],
    'det':        ['DET', 'Detector'],
    'obj':        ['OBJ', 'Objective', 'Obiettivo'],
    'grat':       ['Grat', 'Grating', 'Reticolo'],
    'hole':       ['Hole', 'Pinhole'],
    'range':      ['RANGE', 'Spectral range'],
    'spectro':    ['SPECTRO', 'Spectrometer'],
    'notes':      ['Notes', 'Note', 'Comments', 'Comment'],
    # FT-Raman
    'res':        ['RES', 'Resolution'],
    'aper':       ['APER', 'Aperture'],
    'scan_t':     ['SCAN T', 'Scan time'],
    'scan_v':     ['SCAN V', 'Scan velocity', 'Scan speed'],
    'pow':        ['POW', 'Power', 'Laser power'],
    'p_gain':     ['P GAIN', 'Preamp gain', 'Gain'],
    'low_pass':   ['LOW PASS', 'Low pass', 'Low-pass filter'],
    'ph_res':     ['PH RES', 'Phase resolution'],
    'pos':        ['POS', 'Position'],
}

# Column order of the original registers. Used only as a fallback, when a
# register has one of these layouts (recognised by its third column, as in earlier
# versions) and a field is not found by name.
_LEGACY_LAYOUTS = {
    'ndfilt': ('Raman', 'LabRAM',
               ['n', 'sample', 'ndfilt', 'acqt', 'acc', 'det', 'obj', 'grat',
                'ecc', 'hole', 'range', 'spectro', 'notes']),
    'res': ('FT-Raman', 'FT-Raman',
            ['n', 'sample', 'res', 'range', 'aper', 'scan_t', 'scan_v', 'pow',
             'p_gain', 'low_pass', 'ph_res', 'pos', 'det']),
}


def normalize_name(name, keep_units=True):
    """Simplify a column name for comparison.

    Lower case, only letters and digits; the Symbol-font micro sign and 'µ'
    become 'u'. With ``keep_units=False`` text in brackets is dropped first:
    'Ecc (nm)' -> 'ecc', 'Hole\\n(µm)' -> 'hole'.
    """
    s = str(name).replace('', 'u').replace('µ', 'u').replace('μ', 'u')
    if not keep_units:
        s = re.sub(r'[\(\[\{].*?[\)\]\}]', '', s)
    return re.sub(r'[^0-9a-z]', '', s.lower())


def find_column(columns, aliases):
    """Return the first column whose name matches one of ``aliases``, or None.

    A column matches an alias if their normalised names are equal, with or
    without the units in brackets: 'Ecc (nm)' matches 'Ecc' and 'Ecc (nm)'.
    """
    full = {normalize_name(c): c for c in columns}
    base = {}
    for c in columns:
        base.setdefault(normalize_name(c, keep_units=False), c)
    for alias in aliases:
        a_full = normalize_name(alias)
        a_base = normalize_name(alias, keep_units=False)
        if a_full in full:
            return full[a_full]
        if a_base in base:
            return base[a_base]
    return None


def to_number(value):
    """Convert to int or float when possible; NaN and empty -> None.

    Text that is not a number (e.g. '326-670', 'CCD50') is returned stripped.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, str):
        s = value.strip()
        if s == '' or s.lower() in ('nan', 'none', '---', '-'):
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s.replace(',', '.')) if s.count(',') <= 1 else s
        except ValueError:
            return s
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# ------------------------------------------------------------------- register
def register_layout(columns):
    """Return ``(meas_type, instrument, legacy_order)`` of a register, or Nones.

    The instrument is recognised from the columns it has: a filter column
    ('ND FILT') -> LabRAM Raman; a resolution column ('RES') -> FT-Raman.
    ``legacy_order`` is the column order of the original register, used as
    fallback only when the third column has the original name.
    """
    columns = list(columns)
    third = normalize_name(columns[2]) if len(columns) > 2 else None
    for key, (meas_type, instrument, order) in _LEGACY_LAYOUTS.items():
        legacy = third == normalize_name(REGISTER_ALIASES[key][0])
        if legacy or find_column(columns, REGISTER_ALIASES[key]) is not None:
            return meas_type, instrument, (order if legacy else None)
    return None, None, None


def register_columns(columns, aliases=None):
    """Where each field is in a register: ``{key: column name, position or None}``.

    Every key of ``aliases`` (default :data:`REGISTER_ALIASES`) is looked up
    by column name. If it is not found and the register has the original
    layout (see :func:`register_layout`), its original position (int) is
    used. Otherwise None. A column is assigned to one key only.
    """
    aliases = REGISTER_ALIASES if aliases is None else aliases
    columns = list(columns)
    legacy = register_layout(columns)[2]
    where, used = {}, set()
    for key, names in aliases.items():
        col = find_column([c for c in columns if c not in used], names)
        if col is not None:
            where[key] = col
            used.add(col)
        elif legacy is not None and key in legacy and legacy.index(key) < len(columns):
            where[key] = legacy.index(key)
        else:
            where[key] = None
    return where


def register_value(row, where):
    """Raw cell of a register row, from a column name or position (None -> None)."""
    if where is None:
        return None
    return row.iloc[where] if isinstance(where, int) else row[where]


def from_register(row, aliases=None):
    """Standard metadata from one register row (a pandas Series).

    Fields are found with :func:`register_columns`; missing ones are None.
    ``meas_type`` comes from a type column if the register has one,
    otherwise from the instrument (:func:`register_layout`).
    """
    meas_type, instrument, _ = register_layout(row.index)
    where = register_columns(row.index, aliases)
    meta = {key: to_number(register_value(row, w)) for key, w in where.items()}
    if meta.get('meas_type') is None:
        meta['meas_type'] = meas_type
    meta['instrument'] = instrument
    if meta.get('sample') is not None:
        meta['sample'] = str(meta['sample'])
    meta['source'] = 'register'
    return meta


def register_row(table, number):
    """Return the register row of measurement ``number`` (column 'N').

    Warns and returns None if no row has this number; with several rows, warns
    and returns the first.
    """
    rows = table[table['N'] == number]
    if len(rows) == 0:
        warnings.warn(f"measurement {number} is not in the register "
                      f"(column 'N'); no metadata loaded", stacklevel=3)
        return None
    if len(rows) > 1:
        warnings.warn(f"measurement {number} appears {len(rows)} times in the "
                      f"register; the first row is used", stacklevel=3)
    return rows.iloc[0]


# ------------------------------------------------------------------ file name
#: Default TRPL file-name pattern: wafer_sample_meastype_tstart_tstop_CCDx_Gx_prognum
#: (extra fields before prognum are allowed, as before).
#: Named groups become fields; 'samp' is the sample, 'g_width' the gate number.
TRPL_PATTERN = (r'^(?P<wafer>[^_]+)_(?P<samp>[^_]+)_(?P<mes_type>[^_]+)'
                r'_(?P<t_start>[^_]+)_(?P<t_stop>[^_]+)_(?P<CCD>[^_]+)'
                r'_G(?P<g_width>[^_]+)(?:_.*)?_(?P<prog_num>[^_]+)$')


def parse_name(name, pattern=None, warn=True):
    """Fields of a file name, from a regular expression with named groups.

    Parameters
    ----------
    name : str
        File name without extension.
    pattern : str or compiled regex, optional
        Default :data:`TRPL_PATTERN` (read when the function runs, so it can be
        changed at run time).
    warn : bool
        Warn when the name does not match.

    Returns
    -------
    dict or None
        ``{group: text}``, values as text, as before; None if the
        name does not match.
    """
    pattern = TRPL_PATTERN if pattern is None else pattern
    m = re.match(pattern, name)
    if m is None:
        if warn:
            warnings.warn(f"file name '{name}' does not match the pattern "
                          f"{getattr(pattern, 'pattern', pattern)!r}; no "
                          f"metadata read from it", stacklevel=3)
        return None
    return {k: v for k, v in m.groupdict().items() if v is not None}


def trpl_info(name, folder='', pattern=None):
    """The ``info`` dict of a TRPL file, as built by earlier versions.

    Keys: the groups of the pattern (wafer, samp, mes_type, t_start, t_stop,
    CCD, g_width, prog_num), plus 'data' (first part of the folder path) and
    'position' (number after '#' in the sample). Values are text. An empty
    dict if the name does not match.
    """
    fields = parse_name(name, pattern)
    if fields is None:
        return {}
    info = dict(fields)
    info['data'] = folder.split('/')[0]
    if 'samp' in info and 'position' not in info:
        info['position'] = info['samp'].split('#')[-1]
    return info


# ----------------------------------------------------------------- any source
def standardize(info):
    """Standard metadata from the ``info`` of a Data (any source).

    - register row (Series with an 'N' column or a known layout) -> :func:`from_register`
    - TRPL info (has 'wafer') -> fields of the file name, as numbers
    - file header (absorption, open_file) -> 'SAMPLE' and numeric keys
    - None -> empty
    """
    if info is None:
        return {}
    if isinstance(info, pd.DataFrame):
        info = info[0] if 0 in info.columns else info.iloc[:, 0]
    items = dict(info.items()) if isinstance(info, pd.Series) else dict(info)
    if not items:
        return {}

    if 'wafer' in items:                                   # TRPL file name
        meta = {k: to_number(v) for k, v in items.items()}
        for k in ('wafer', 'samp', 'mes_type', 'CCD', 'prog_num'):
            if items.get(k) is not None:
                meta[k] = str(items[k])
        meta['sample'] = meta.get('samp')
        meta['meas_type'] = 'trpl'
        meta['source'] = 'filename'
        return meta

    if isinstance(info, pd.Series) and (
            'N' in items or register_layout(list(items))[0] is not None):
        return from_register(info)

    meta = {}                                              # file header
    sample = items.get('SAMPLE', items.get('sample'))
    if sample is not None:
        meta['sample'] = str(sample)
    values = [str(v).upper() for v in items.values()]
    if 'ABSORBANCE' in values:
        meta['meas_type'] = 'absorption'
    meta['source'] = 'header'
    return meta
