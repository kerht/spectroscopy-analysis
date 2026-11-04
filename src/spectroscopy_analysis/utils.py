"""Small numerical helpers."""

import numpy as np


def moving_average(x, w):
    """Centered moving average of ``x`` over ``w`` points.

    The ends are padded by reflection, so the output has the same length as
    the input. If ``w`` is False, ``x`` is returned unchanged.
    """

    if w is False:
        smooth_x = x
    else:
        # with padding
        padding = w // 2
        x_padded = np.pad(x, padding, mode='reflect')
        smooth_x = np.convolve(x_padded, np.ones(w), 'valid') / w
        if w % 2 == 0:
            smooth_x = smooth_x[:-1]

        # # without padding
        # smooth_x = np.convolve(x, np.ones(w), 'valid') / w

    return smooth_x
