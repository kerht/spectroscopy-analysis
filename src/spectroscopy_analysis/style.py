"""Plot styles: colour/marker/linestyle cycles and the matplotlib look.

Importing the package does **not** change matplotlib's settings. Call
:func:`apply_style` once, at the top of a script or notebook::

    import spectroscopy_analysis as sa
    sa.apply_style()              # poster look: large labels and ticks
    sa.apply_style('paper')       # same, with matplotlib's default font sizes
"""
import matplotlib.pyplot as plt
import numpy as np


colors = ['blue', 'green', 'red', 'orange', 'purple', 'pink']
linestyles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (3, 10, 1, 10))]
markers = ['o', 'D', 's', '^', '*', 'x']

# Pair markers with linestyles by index, then cycle colors fastest.
# Used by Data.plot_lifetime (with a fit) as (color, marker, linestyle).
all_styles = [
    (color, marker, ls)
    for marker, ls in zip(markers, linestyles)
    for color in colors
]

# (color, linestyle) pairs, used by Data.plot_lifetime without a fit.
color_ls = [
    (color, ls)
    for ls in linestyles
    for color in colors
]


def ticks(y, pos):
    """Tick labels as powers of e (``e^{-2}``, ``e^{0}``, ...) for ln-scale axes."""
    if y >= 1:
        return r'$e^{{{:.0f}}}$'.format(np.log(y))
    else:
        return r'$e^{{-{:.0f}}}$'.format(abs(np.log(y)))


def apply_style(kind='poster'):
    """Set the matplotlib look used by the package's plots.

    Parameters
    ----------
    kind : {'poster', 'paper'}
        'poster' : serif fonts, ticks on all sides, 9.6 x 7.2 in figures,
                   300 dpi export, large tick labels (24) and axis labels
                   (20).
        'paper'  : the same, with matplotlib's default font sizes.
    """
    if kind not in ('poster', 'paper'):
        raise ValueError("kind must be 'poster' or 'paper'")

    rc = plt.rcParams
    rc['figure.constrained_layout.use'] = True
    rc['savefig.dpi'] = 300
    rc['axes.autolimit_mode'] = 'round_numbers'
    rc['axes.linewidth'] = 0.8
    rc['figure.figsize'] = [9.6, 7.2]          # 4:3, twice the default size
    rc['font.family'] = 'serif'
    rc['font.size'] = 12.0
    rc['grid.linewidth'] = 0.8
    rc['legend.frameon'] = True
    rc['lines.linewidth'] = 1.5
    rc['mathtext.fontset'] = 'dejavuserif'
    rc['xtick.minor.size'] = 2.0
    rc['xtick.minor.visible'] = True
    rc['xtick.top'] = True
    rc['ytick.minor.size'] = 2.0
    rc['ytick.minor.visible'] = True
    rc['ytick.right'] = True
    rc['legend.fontsize'] = 'large'
    rc['errorbar.capsize'] = 5

    # LaTeX fonts: nicer, but slower and they need a LaTeX installation
    # rc['text.latex.preamble'] = '\\usepackage{amsmath} \\usepackage{amssymb}'
    # rc['text.usetex'] = True

    if kind == 'poster':
        rc['xtick.major.size'] = 10
        rc['ytick.major.size'] = 10
        rc['legend.fontsize'] = 17
        rc['xtick.labelsize'] = 24
        rc['ytick.labelsize'] = 24
        rc['axes.labelsize'] = 20
