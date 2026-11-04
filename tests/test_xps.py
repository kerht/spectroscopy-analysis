"""XPS depth profile on the example VAMAS file (needs vamas and lmfitxps)."""
import os

import numpy as np
import pytest

import spectroscopy_analysis as sa


@pytest.mark.parametrize('bg_type', ['tougaard', 'shirley'])
def test_xps_profile(examples, bg_type):
    xps = sa.XPS()
    xps.load_data('W1P1.vms', path=os.path.join(examples, 'xps'))
    assert len(xps.blocks) == 228                         # 4 elements x 57 depths
    xps.plot_xps(bg_type=bg_type, shift=True)
    for b in xps.bg.values():                             # one background array per block
        assert np.ndim(b[0]) == 1
    at, depths = xps.fit_blocks(plot=False, int_type='raw')
    assert at.shape == (57, 4)
    assert np.allclose(at.sum(axis=1), 100)
