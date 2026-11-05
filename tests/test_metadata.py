"""Metadata: register read by column name, row chosen by N, file-name pattern, meta=."""
import os
import shutil

import pandas as pd
import pytest

import spectroscopy_analysis as sa
from spectroscopy_analysis import metadata


# ------------------------------------------------------------------ helpers
@pytest.mark.parametrize('columns, aliases, expected', [
    (['N', 'SAMPLE', 'Ecc (nm)'], ['Ecc'], 'Ecc (nm)'),
    (['N', 'sample', 'ECC\n(nm)'], ['Ecc (nm)'], 'ECC\n(nm)'),
    (['N', 'Laser [nm]'], metadata.REGISTER_ALIASES['ecc'], 'Laser [nm]'),
    (['N', 'Hole\n(m)'], ['Hole (um)'], 'Hole\n(m)'),
    (['N', 'SAMPLE'], ['Ecc'], None),
])
def test_find_column(columns, aliases, expected):
    assert metadata.find_column(columns, aliases) == expected


def test_to_number():
    assert metadata.to_number('100') == 100
    assert metadata.to_number('0,5') == 0.5
    assert metadata.to_number(float('nan')) is None
    assert metadata.to_number('326-670') == '326-670'


def test_parse_name_default_and_custom():
    f = metadata.parse_name('W1_P#2_SEQ_117_127_CCD50_G1_1')
    assert f['wafer'] == 'W1' and f['samp'] == 'P#2' and f['g_width'] == '1'
    custom = r'(?P<samp>[^-]+)-(?P<t_start>\d+)-(?P<t_stop>\d+)'
    assert metadata.parse_name('A-5-20', custom) == {'samp': 'A', 't_start': '5',
                                                    't_stop': '20'}
    with pytest.warns(UserWarning, match='does not match'):
        assert metadata.parse_name('random_name') is None


# --------------------------------------------------- register by column name
@pytest.fixture
def lab_folder(tmp_path, examples, monkeypatch):
    """A measurement folder whose register has moved, renamed and extra
    columns, rows out of order, an empty row and a missing measurement."""
    folder = tmp_path / '261002'
    folder.mkdir()
    for n in (3, 5, 9):
        shutil.copy(os.path.join(examples, 'raman', 'Si_reference.txt'),
                    folder / f'2610020{n}.txt')
        (folder / f'2610020{n}.l6s').write_text('')
    table = pd.DataFrame({
        'Sample': ['B', None, 'A', 'C'],
        'No.': ['5', None, '03*', 9],            # number column with another name
        'User': ['me', None, 'me', 'me'],         # extra column
        'Laser (nm)': [532, None, 633, 633],      # laser with another name
        'Filter': [0.5, None, 1.0, 0.1],
        'Notes': ['second', 'comment line', 'first', None],
    })
    table.to_excel(folder / 'reg_mis_261002.xlsx', index=False)
    monkeypatch.chdir(tmp_path)
    return '261002'


def test_register_by_name_and_row_by_n(lab_folder):
    p = sa.Project()
    d = p.open_data(lab_folder, 3)
    assert d.sample == 'A' and d.meta['sample'] == 'A'
    assert d.ecc == 633 and d.meta['ecc'] == 633
    assert d.meas_type == 'Raman'                  # filter column -> LabRAM
    assert d.ndfilt == 100                         # as before: fraction -> %
    assert d.meta['n'] == 3 and d.meta['source'] == 'register'
    assert p.open_data(lab_folder, 5).sample == 'B'   # row 1 of the file


def test_register_empty_rows_skipped(lab_folder):
    table = sa.Project().open_table(lab_folder)
    assert sorted(table['N']) == [3, 5, 9]


def test_missing_measurement_warns(lab_folder, tmp_path, examples):
    shutil.copy(os.path.join(examples, 'raman', 'Si_reference.txt'),
                tmp_path / lab_folder / '26100207.txt')
    (tmp_path / lab_folder / '26100207.l6s').write_text('')
    with pytest.warns(UserWarning, match='not in the register'):
        d = sa.Project().open_data(lab_folder, 7, meta={'ecc': 633})
    assert d.meas_type == 'generic' and d.ecc == 633


def test_meta_overrides_register(lab_folder):
    d = sa.Project().open_data(lab_folder, 9, meta={'meas_type': 'PL', 'sample': 'X'})
    assert d.meas_type == 'PL' and d.sample == 'X'
    assert d.meta['source'] == 'register+user'
    assert d.info['Sample'] == 'C'                 # info is not modified


def test_open_folder_with_renamed_number_column(lab_folder):
    p = sa.Project()
    p.open_folder(lab_folder, meta={'meas_type': 'PL'})
    assert sorted(p.data) == ['26100203', '26100205', '26100209']
    assert all(d.meas_type == 'PL' for d in p.data.values())


def test_original_register_unchanged(examples, monkeypatch):
    monkeypatch.chdir(examples)
    m = sa.Project().open_data('261001', 7)
    d = m[0]
    assert d.meta['instrument'] == 'LabRAM' and d.meta['ecc'] == 325.0
    assert d.n == 7 and d.notes == 'Mappa z (-8; 0)'


def test_legacy_layout_fallback():
    """Original register layout: a field with an unknown name is still found
    by its original position."""
    row = pd.Series([7, 'S', 0.5, 5, 10, 'CCD', '40x', 600, 325, 200, 'r', 's', 'n'],
                    index=['N', 'SAMPLE', 'ND FILT', 'Acq T (s)', 'Acc', 'DET', 'OBJ',
                           'Grat (l/mm)', 'Lambda?', 'Hole', 'RANGE', 'SPECTRO', 'Notes'])
    assert metadata.from_register(row)['ecc'] == 325


# --------------------------------------------------------------------- TRPL
def test_trpl_meta_numbers(examples):
    m = sa.Project().open_pl(os.path.join(examples, 'trpl'), 'W1_P#1_SEQ_100_160_CCD50_G1_1')
    d = m[0]
    assert d.info['t_start'] == '100'              # info: text, as before
    assert d.meta['t_start'] == 100 and d.meta['g_width'] == 1
    assert d.meta['wafer'] == 'W1' and d.meta['sample'] == 'P#1'
    assert d.meta['meas_type'] == 'trpl' and d.meta['source'] == 'filename'


def test_trpl_other_name_with_meta(examples, tmp_path):
    src = os.path.join(examples, 'trpl', 'W1_P#1_SEQ_100_160_CCD50_G1_1.txt')
    shutil.copy(src, tmp_path / 'decay.txt')
    with pytest.warns(UserWarning, match='does not match'):
        m = sa.Project().open_pl(str(tmp_path), 'decay',
                                 meta={'t_start': 100, 't_stop': 160, 'g_width': 1})
    d = m[0]
    assert d.meas_type == 'trpl'
    assert d.info['t_start'] == '100'              # usable by the TRPL methods
    assert d.meta['t_stop'] == 160


def test_trpl_custom_pattern(examples, tmp_path):
    src = os.path.join(examples, 'trpl', 'W1_P#1_SEQ_100_160_CCD50_G1_1.txt')
    shutil.copy(src, tmp_path / 'W1-P#1-100-160-G1.txt')
    pattern = r'(?P<wafer>[^-]+)-(?P<samp>[^-]+)-(?P<t_start>\d+)-(?P<t_stop>\d+)-G(?P<g_width>\d+)'
    m = sa.Project().open_pl(str(tmp_path), 'W1-P#1-100-160-G1', pattern=pattern,
                             label='sample')
    assert m[0].info['samp'] == 'P#1' and m[0].label == 'W1 P#1'
    g = sa.Project().open_folder(str(tmp_path), meas_type='trpl', pattern=pattern)
    assert list(g['W1']['short']) == ['W1-P#1-100-160-G1']


def test_absorption_meta(examples):
    d = sa.Project().open_abs(os.path.join(examples, 'absorption', 'avantes'), 'W1_P#4',
                              instrument='avantes')
    assert d.meta['meas_type'] == 'absorption' and d.meta['instrument'] == 'avantes'
