"""Shared matplotlib configuration for all paper figures."""
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

FONT_FAMILY = 'Times New Roman'
LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 7
LEGEND_FONT_SIZE = 7

# Plot defaults
sns.set_theme(style='ticks', font=FONT_FAMILY, font_scale=0.9,
              rc={
                  'font.family': 'serif',
                  'font.serif': [FONT_FAMILY],
                  'mathtext.fontset': 'custom',
                  'mathtext.rm': FONT_FAMILY,
                  'mathtext.it': f'{FONT_FAMILY}:italic',
                  'mathtext.bf': f'{FONT_FAMILY}:bold',
                  'pdf.fonttype': 42,
                  'ps.fonttype': 42,
                  'axes.labelsize': LABEL_FONT_SIZE,
                  'axes.titlesize': LABEL_FONT_SIZE,
                  'axes.labelweight': 'bold',
                  'axes.titleweight': 'bold',
                  'legend.fontsize': LEGEND_FONT_SIZE,
                  'xtick.labelsize': TICK_FONT_SIZE,
                  'ytick.labelsize': TICK_FONT_SIZE,
                  'axes.labelpad': 1,
                  'xtick.major.pad': 1,
                  'ytick.major.pad': 1,
                  'xtick.major.size': 2.5,
                  'ytick.major.size': 2.5,
                  'xtick.direction': 'in',
                  'ytick.direction': 'in',
                  'figure.dpi': 300,
                  'savefig.dpi': 300,
                  'savefig.bbox': 'tight',
                  'savefig.pad_inches': 0.02,
                  'axes.edgecolor': '.15',
                  'axes.linewidth': 0.6,
                  'axes.spines.top': False,
                  'axes.spines.right': False,
              })

# A4 width = 8.27 inches
FIG_HALF = (8.27 * 0.45, 2.5)
FIG_FULL = (8.27 * 0.9, 2.5)

# Method colors
COLORS = {
    'ours':          '#5B9BD5',  # Blue
    'simpoint':      '#ED7D31',  # Orange
    'abs_surrogate': '#70AD47',  # Green
    'seqpred':       '#FF6B6B',  # Coral red (TaoPredictor baseline)
    'random':        '#BFBFBF',  # Light gray
}

METHOD_NAMES = {
    'ours': 'Our Model',
    'abs_surrogate': 'Abs. Surrogate',
    'simpoint': 'SimPoint',
    'seqpred': 'TaoPredictor',
    'random': 'Random',
}

TASK_KEYS = [
    'cpi', 'branch_mpki', 'l1_mpki', 'l1i_mpki', 'l2_mpki',
    'rob_stall', 'lq_stall', 'sq_stall', 'l2_miss_rate', 'btb_miss_rate',
]

CACHE_TASKS = ('l1_mpki', 'l1i_mpki', 'l2_mpki', 'l2_miss_rate')

TASK_DISPLAY = {
    'cpi': 'CPI',
    'branch_mpki': 'Br. MPKI',
    'l1_mpki': 'L1D MPKI',
    'l1i_mpki': 'L1I MPKI',
    'l2_mpki': 'L2 MPKI',
    'rob_stall': 'ROB Stall',
    'lq_stall': 'LQ Stall',
    'sq_stall': 'SQ Stall',
    'l2_miss_rate': 'L2 MR',
    'btb_miss_rate': 'BTB MR',
}

# Multi-category palettes
ABLATION_PALETTE = ['#5B9BD5', '#FF6B6B', '#ED7D31', '#A580CA', '#70AD47', '#F4A261']
OOD_PALETTE = ['#5B9BD5', '#ED7D31', '#70AD47', '#FF6B6B']


def _style_text(text, fontsize=None, fontweight=None):
    """Apply publication font settings to a matplotlib Text object."""
    if text is None:
        return
    text.set_fontfamily(FONT_FAMILY)
    if fontsize is not None:
        text.set_fontsize(fontsize)
    if fontweight is not None:
        text.set_fontweight(fontweight)


def style_text(text, fontsize=LABEL_FONT_SIZE, fontweight='bold'):
    """Style a standalone figure-level or annotation text object."""
    _style_text(text, fontsize=fontsize, fontweight=fontweight)


def style_axis(ax):
    """Enforce publication fonts on titles, labels, ticks, and legends."""
    _style_text(ax.title, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    _style_text(ax.xaxis.label, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    _style_text(ax.yaxis.label, fontsize=LABEL_FONT_SIZE, fontweight='bold')

    for label in ax.get_xticklabels():
        _style_text(label, fontsize=TICK_FONT_SIZE, fontweight='normal')
    for label in ax.get_yticklabels():
        _style_text(label, fontsize=TICK_FONT_SIZE, fontweight='normal')

    _style_text(ax.xaxis.get_offset_text(), fontsize=TICK_FONT_SIZE, fontweight='normal')
    _style_text(ax.yaxis.get_offset_text(), fontsize=TICK_FONT_SIZE, fontweight='normal')

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            _style_text(text, fontsize=LEGEND_FONT_SIZE, fontweight='normal')
        legend_title = legend.get_title()
        if legend_title and legend_title.get_text():
            _style_text(legend_title, fontsize=LEGEND_FONT_SIZE, fontweight='bold')


def style_figure(fig):
    """Apply publication fonts to a finished figure."""
    for ax in fig.axes:
        style_axis(ax)

    suptitle = getattr(fig, '_suptitle', None)
    for text in fig.texts:
        if text is suptitle:
            continue
        _style_text(text)
    if suptitle is not None:
        _style_text(suptitle, fontsize=LABEL_FONT_SIZE, fontweight='bold')


def rotate_xlabels(ax, rotation=30):
    """Rotate x-axis labels for long benchmark names."""
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')


def savefig(fig, name):
    """Save figure as PDF only."""
    fig.savefig(f'{name}.pdf')
    print(f'Saved {name}.pdf')
    plt.close(fig)
