#!/usr/bin/env python
"""
Shared journal style for the MRes thesis figures.

Target: Communications Chemistry / Lab on a Chip.
  - matplotlib only (no seaborn style sheets)
  - Liberation Sans (metric-compatible with Arial); Arial preferred if present,
    DejaVu Sans as last resort
  - body 7 pt, axis labels 8 pt, panel letters 9 pt bold
  - single column 89 mm, double column 183 mm, converted to inch explicitly
  - top/right spines removed, ticks pointing out, 0.6 pt line weight
  - Okabe-Ito colourblind-safe palette; colour is never the only channel
  - every figure written as PDF (vector, Type 42 embedded fonts) and PNG (600 dpi)
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --------------------------------------------------------------------------
# Physical sizes.  Journals specify millimetres; matplotlib wants inches.
# --------------------------------------------------------------------------
MM_PER_IN = 25.4
SINGLE_COL_MM = 91.5
DOUBLE_COL_MM = 183.0


def mm2in(mm):
    """Millimetres -> inches."""
    return mm / MM_PER_IN


def figsize_mm(w_mm, h_mm):
    """(width_mm, height_mm) -> matplotlib figsize tuple in inches."""
    return (mm2in(w_mm), mm2in(h_mm))


def add_axes_mm(fig, x_mm, y_mm, w_mm, h_mm):
    """Place an Axes using millimetres measured from the figure's bottom-left.

    Using absolute mm (rather than tight_layout) keeps the rendered figure at
    exactly the declared column width, which is what the journal checks.
    """
    W = fig.get_figwidth() * MM_PER_IN
    H = fig.get_figheight() * MM_PER_IN
    return fig.add_axes([x_mm / W, y_mm / H, w_mm / W, h_mm / H])


def text_mm(fig, x_mm, y_mm, s, **kw):
    """Figure-level text positioned in millimetres from the bottom-left."""
    W = fig.get_figwidth() * MM_PER_IN
    H = fig.get_figheight() * MM_PER_IN
    return fig.text(x_mm / W, y_mm / H, s, **kw)


# --------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------
FONT_STACK = ['Arial', 'Liberation Sans', 'DejaVu Sans']

FS_BODY = 7      # tick labels, annotations, legend
FS_LABEL = 8     # axis labels
FS_PANEL = 9     # panel letters (bold)
FS_SMALL = 6     # dense in-figure annotation


def resolved_font():
    """The first font in FONT_STACK actually installed."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in FONT_STACK:
        if name in available:
            return name
    return 'DejaVu Sans'


# --------------------------------------------------------------------------
# Colour.  Okabe-Ito, safe for deuteranopia / protanopia / tritanopia.
# Significance and set membership are additionally encoded by marker fill and
# hatch, so no reader depends on hue alone.
# --------------------------------------------------------------------------
OKABE = {
    'black':     '#000000',
    'orange':    '#E69F00',
    'skyblue':   '#56B4E9',
    'green':     '#009E73',
    'yellow':    '#F0E442',
    'blue':      '#0072B2',
    'vermillion':'#D55E00',
    'purple':    '#CC79A7',
}

INK       = '#1A1A1A'   # near-black for text and rules
GREY      = '#767676'   # half-width ticks
GREY_LT   = '#B0B0B0'   # separators
GREY_XLT  = '#E4E4E4'   # row guides, median half-width hint

LW_AXIS = 0.6
LW_RULE = 0.6


def lighten(hex_colour, amount):
    """Blend a hex colour toward white by `amount` in [0, 1] (same hue family)."""
    h = hex_colour.lstrip('#')
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    out = [round(c + (255 - c) * amount) for c in rgb]
    return '#%02X%02X%02X' % tuple(out)


def darken(hex_colour, amount):
    """Blend a hex colour toward black by `amount` in [0, 1] (same hue family)."""
    h = hex_colour.lstrip('#')
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    out = [round(c * (1.0 - amount)) for c in rgb]
    return '#%02X%02X%02X' % tuple(out)


# --------------------------------------------------------------------------
# stacey_figures.py design spec, shared by Figure 1 and Figure 3.
# Kept from that file apart from the agreed deviations:
#   - Liberation Sans leads the font stack
#   - no broken axes
#   - no savefig.bbox="tight" (the canvas is fixed, so the output carries its
#     true physical size)
#   - the type scale and column widths follow the house spec above rather than
#     stacey_figures.py's own 8/7 pt and 85/180 mm, so all three figures match:
#     body and ticks FS_BODY, axis labels FS_LABEL, panel letters FS_PANEL,
#     widths SINGLE_COL_MM / DOUBLE_COL_MM.
# --------------------------------------------------------------------------
STACEY_FONT_STACK = ['Liberation Sans', 'DejaVu Sans']

STACEY_STYLE = {
    'font.family':       STACEY_FONT_STACK,   # list form keeps per-glyph fallback
    'font.sans-serif':   STACEY_FONT_STACK,
    'font.size':         FS_BODY,
    'axes.labelsize':    FS_LABEL,
    'axes.titlesize':    FS_LABEL,
    'xtick.labelsize':   FS_BODY,
    'ytick.labelsize':   FS_BODY,
    'legend.fontsize':   FS_BODY,
    'axes.linewidth':    0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'lines.linewidth':   1.0,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'savefig.dpi':       600,
    'savefig.pad_inches': 0.0,
}

# Scheme B -- colour-blind safe
STACEY_C = {
    'pdbbind': '#38618C',   # blue        -- PDBbind / primary source
    'hiqbind': '#6E8B7B',   # sage        -- HiQBind / second source
    'gator':   '#B0603D',   # rust        -- GatorAffinity-DB / third source
    'fourth':  '#8A7CA8',   # plum        -- spare
    'ns':      '#5A6B7C',   # slate       -- non-significant comparisons
    'sig':     '#9C4A45',   # brick       -- significant comparisons
    'ref':     '#A0A0A0',   # grey        -- reference lines, secondary text
    'band':    '#E6E4E0',   # warm pale   -- detection-limit band, fills
    'high':    '#38618C',   # blue        -- high-similarity stratum (Fig. 3)
    'low':     '#B0603D',   # rust        -- low-similarity stratum (Fig. 3)
}

MARKER_SIZE = 16

# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------
def apply():
    """Install the house style. Call once at the top of each figure script."""
    _FONT = resolved_font()

    # A missing glyph must never ship as a silent empty box: promote
    # matplotlib's warning to an error so the build stops instead.
    warnings.filterwarnings('error', message=r'Glyph \d+ .* missing from font')
    plt.rcParams.update({
        # fonts -- Type 42 (TrueType) keeps text selectable and embedded in PDF.
        # font.family is the explicit stack, not the alias 'sans-serif':
        # only the list form enables matplotlib's per-glyph fallback.  With the
        # alias, a glyph the first font lacks (Liberation Sans has no U+222A
        # UNION) is silently dropped and ships as an empty box.
        'font.family': FONT_STACK,
        'font.sans-serif': FONT_STACK,
        'font.size': FS_BODY,
        'axes.labelsize': FS_LABEL,
        'axes.titlesize': FS_LABEL,
        'xtick.labelsize': FS_BODY,
        'ytick.labelsize': FS_BODY,
        'legend.fontsize': FS_BODY,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',

        # Keep mathtext ($\Delta$, $\cup$, subscripts) in the same family --
        # otherwise matplotlib silently falls back to DejaVu Sans and the PDF
        # ships two typefaces.
        'mathtext.fontset': 'custom',
        'mathtext.rm': _FONT,
        'mathtext.it': f'{_FONT}:italic',
        'mathtext.bf': f'{_FONT}:bold',
        'mathtext.sf': _FONT,
        'mathtext.default': 'regular',

        # spines / ticks
        'axes.linewidth': LW_AXIS,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': LW_AXIS,
        'ytick.major.width': LW_AXIS,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'xtick.major.size': 2.2,
        'ytick.major.size': 2.2,
        'xtick.minor.size': 1.2,
        'ytick.minor.size': 1.2,
        'xtick.color': INK,
        'ytick.color': INK,
        'text.color': INK,

        # lines / legend
        'lines.linewidth': 0.9,
        'lines.markersize': 3.2,
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'legend.handletextpad': 0.5,
        'legend.columnspacing': 1.2,
        'legend.borderpad': 0.0,
        'legend.labelspacing': 0.4,

        # output
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 600,
        'savefig.pad_inches': 0.0,
        'axes.unicode_minus': False,   # use ASCII hyphen-minus, renders in all fonts
    })


def panel_letter(fig, x_mm, y_mm, letter):
    """Lower-case bold panel letter, placed outside the axes at its top-left."""
    return text_mm(fig, x_mm, y_mm, letter,
                   fontsize=FS_PANEL, fontweight='bold', color=INK,
                   ha='left', va='top')


def save(fig, outdir, stem):
    """Write <stem>.pdf (vector, Type 42) and <stem>.png (600 dpi).

    No bbox_inches='tight': that would silently change the physical width.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f'{stem}.pdf'
    png = outdir / f'{stem}.png'
    fig.savefig(pdf)                 # dpi irrelevant for vector output
    fig.savefig(png, dpi=600)
    w_mm = fig.get_figwidth() * MM_PER_IN
    h_mm = fig.get_figheight() * MM_PER_IN
    return pdf, png, w_mm, h_mm
