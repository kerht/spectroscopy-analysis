"""Loaders on the example data in examples/data."""
import os

import numpy as np
import pytest

import spectroscopy_analysis as sa


def test_open_spectrum_raman(examples):
    p = sa.Project()
    d = p.open_spectrum(os.path.join(examples, 'raman', 'Si_reference.txt'), info=None, data=p.data)
    assert isinstance(d.x, np.ndarray) and d.x.dtype == float
    assert d.x.size == d.y.size == 2867
    assert 515 < d.x[np.argmax(d.y)] < 525                 # Si peak


def test_open_map_raman(examples):
    p = sa.Project()
    m = p.open_map(os.path.join(examples, 'raman', 'z_scan.txt'), info=None, data=p.data)
    assert len(m) == 41
    assert all(d.x.size == d.y.size for d in m)


def test_open_data_with_register(examples, monkeypatch):
    monkeypatch.chdir(examples)
    p = sa.Project()
    m = p.open_data('261001', 7)                            # PL z-map, measurement 7
    assert len(m) == 9
    assert str(m[0].sample) == 'sample 1'
    assert float(m[0].ecc) == 325


def test_read_file_avantes_header(examples):
    r = sa.Project.read_file(os.path.join(examples, 'absorption', 'avantes', 'W1_P#4.TXT'),
                             verbose=False)
    assert r['header_lines'] == 8
    assert r['n_columns'] == 5


@pytest.mark.parametrize('folder, name, instrument, n', [
    ('jasco', 'W1_P#3', 'jasco', 601),
    ('avantes', 'W1_P#4', 'avantes', 2313),                # .TXT in upper case
])
def test_open_abs(examples, folder, name, instrument, n):
    d = sa.Project().open_abs(os.path.join(examples, 'absorption', folder), name,
                              instrument=instrument)
    assert d.x.size == n
    assert not np.isnan(d.x).any() and not np.isnan(d.y).any()
    assert d.meas_type == 'absorption'


def test_open_pl_trpl_series(examples):
    p = sa.Project()
    m = p.open_pl(os.path.join(examples, 'trpl'), 'W1_P#1_SEQ_100_160_CCD50_G1_1')
    assert len(m) == 61
    d = m[0]
    assert isinstance(d.x, np.ndarray) and isinstance(d.y, np.ndarray)
    assert d.info['t_start'] == '100' and d.info['g_width'] == '1'


def test_open_folder_trpl_groups(examples):
    p = sa.Project()
    g = p.open_folder(os.path.join(examples, 'trpl'), meas_type='trpl')
    assert list(g) == ['W1']
    assert len(g['W1']['short']) == 6 and len(g['W1']['long']) == 1


def test_open_file_generic(examples):
    d = sa.Project().open_file(os.path.join(examples, 'raman', 'Si_reference.txt'),
                               meas_type='Raman', ecc=633, verbose=False)
    assert d.x.size == 2867
