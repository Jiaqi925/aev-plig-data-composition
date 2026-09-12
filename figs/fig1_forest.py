#!/usr/bin/env python
"""
Figure 1 -- paired differences against the baseline (configs 2-11).

Design follows stacey_figures.py: double-column, two panels side by side, a
grey horizontal band marking +/- half-width, Scheme A colours, a three-item
legend along the bottom, and independent x axes on the two panels.

Three deviations from that file, as specified:
  1. font.sans-serif leads with Liberation Sans (no Arial on this system)
  2. no broken axis -- panel a is [-0.30, +0.12], panel b is [-0.085, +0.035]
  3. savefig.bbox="tight" dropped; the canvas is fixed and the output carries
     its true physical size

Unlike stacey_figures.py, no value is hard-coded: every delta, interval,
half-width and significance flag is read from paired_diff.csv.  The literals
that remain are the reconciliation targets asserted in check().

The band runs from -half-width to +half-width, anchored on ZERO rather than on
the point estimate: the argument is whether |delta| escapes the half-width of
its own interval, so the reader looks at whether the marker clears the band.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / 'experiments' / '07_eval' / 'paired_diff.csv'

# ---------------------------------------------------------------- style ----
WIDTH_DOUBLE_MM = S.DOUBLE_COL_MM     # 183 mm, the house double-column width

STYLE = S.STACEY_STYLE                # typography spec, shared with Figure 3
C = S.STACEY_C                        # Scheme A palette, shared with Figure 3

MARKER_SIZE = 16
CAPSIZE = 0
BAND_LW = 6

# --------------------------------------------------------------- labels ----
LABEL = {
    2:  '+ HiQBind',
    3:  '+ Gator filtered 30%',
    4:  '+ Gator filtered 60%',
    5:  '+ Gator filtered 100%',
    6:  '+ Gator unfiltered 30%',
    7:  '+ Gator unfiltered 60%',
    8:  '+ Gator unfiltered 100%',
    9:  'BindingNet+DCS+HiQBind',
    10: 'BindingNet+DCS+HiQBind+Gator',
    11: 'PDBbind only',
}
CONTROL = 11
MAIN = list(range(2, 11))

PANELS = [
    ('a', 'FEP benchmark', 'dFEP_wmPCC', 'dFEP_lo', 'dFEP_hi', 'dFEP_sig',
     r'$\Delta$ wmPCC vs. baseline', (-0.30, 0.12),
     [-0.30, -0.20, -0.10, 0.00, 0.10]),
    ('b', 'OOD Test', 'dOOD_PCC', 'dOOD_lo', 'dOOD_hi', 'dOOD_sig',
     r'$\Delta$ PCC vs. baseline', (-0.085, 0.035),
     [-0.08, -0.06, -0.04, -0.02, 0.00, 0.02]),
]


# ----------------------------------------------------------------- data ----
def load():
    d = pd.read_csv(SRC)
    assert len(d) == 10, f'expected 10 paired rows, got {len(d)}'
    assert sorted(d.config) == list(range(2, 12)), sorted(d.config)
    assert set(d.vs) == {1}, 'all rows must be paired against config 1 (baseline)'
    d = d.sort_values('config').reset_index(drop=True)

    for _l, _t, dcol, lcol, hcol, scol, _x, _lim, _tk in PANELS:
        # half-width is (hi - lo) / 2 -- NOT |delta - lo| or |hi - delta|.
        # BCa intervals are asymmetric, so those three differ; the text uses
        # the first.
        d[f'{dcol}__hw'] = (d[hcol] - d[lcol]) / 2.0
        d[f'{dcol}__abs'] = d[dcol].abs()
        d[f'{dcol}__below'] = d[f'{dcol}__abs'] < d[f'{dcol}__hw']

        # significance twice: the flag shipped in the CSV, and an independent
        # recomputation from whether the interval covers 0.  They must agree.
        from_csv = d[scol].astype(bool).to_numpy()
        recomputed = ~((d[lcol] <= 0) & (d[hcol] >= 0))
        bad = np.nonzero(from_csv != recomputed)[0]
        if len(bad):
            msg = [f'significance mismatch in {scol}:']
            for i in bad:
                msg.append(f'  config {d.config[i]} ({d.name[i]}): '
                           f'CSV {scol}={from_csv[i]} but CI '
                           f'[{d[lcol][i]:.6f}, {d[hcol][i]:.6f}] '
                           f'-> recomputed {recomputed[i]}')
            raise AssertionError('\n'.join(msg))
        d[f'{dcol}__sig'] = from_csv
    return d


# -------------------------------------------------------------- drawing ----
def forest_panel(ax, rows, floor, dcol, lcol, hcol, xlabel, xlim, xticks,
                 show_ylabels):
    """One forest panel: rows top-to-bottom, control row below a separator."""
    n = len(rows)
    y_rows = [n - i for i in range(n)]      # n, n-1, ..., 1
    y_floor = -0.6

    def band(y, lo, hi):
        half = (hi - lo) / 2.0
        ax.plot([-half, half], [y, y], color=C['band'], linewidth=BAND_LW,
                solid_capstyle='butt', zorder=1)

    def interval(y, d, lo, hi, sig):
        col = C['sig'] if sig else C['ns']
        ax.plot([lo, hi], [y, y], color=col, linewidth=1.1, zorder=3,
                solid_capstyle='butt')
        if CAPSIZE:
            for x in (lo, hi):
                ax.plot([x, x], [y - CAPSIZE * .1, y + CAPSIZE * .1],
                        color=col, linewidth=1.1, zorder=3)
        # filled = significant, open = not.  Colour still separates the two,
        # but fill is the channel that survives greyscale and colour blindness:
        # C['ns'] and C['sig'] differ by only 1.20:1 in luminance.
        if sig:
            ax.scatter([d], [y], s=MARKER_SIZE, color=col, zorder=4,
                       edgecolors='white', linewidths=0.4)
        else:
            ax.scatter([d], [y], s=MARKER_SIZE, facecolors='white',
                       edgecolors=col, linewidths=0.9, zorder=4)

    # detection-limit bands first, so intervals draw over them
    for y, (_, r) in zip(y_rows, rows.iterrows()):
        band(y, r[lcol], r[hcol])

    ax.axvline(0, color=C['ref'], linewidth=0.7, linestyle=(0, (4, 3)), zorder=2)

    for y, (_, r) in zip(y_rows, rows.iterrows()):
        interval(y, r[dcol], r[lcol], r[hcol], bool(r[f'{dcol}__sig']))

    # separator, then the control row (config 11)
    ax.axhline(0.35, color='0.75', linewidth=0.5, linestyle='-', zorder=1)
    band(y_floor, floor[lcol], floor[hcol])
    interval(y_floor, floor[dcol], floor[lcol], floor[hcol],
             bool(floor[f'{dcol}__sig']))

    ax.set_yticks(y_rows + [y_floor])
    labels = [LABEL[int(c)] for c in rows.config] + [LABEL[int(floor.config)]]
    ax.set_yticklabels(labels if show_ylabels else [])
    ax.tick_params(axis='y', labelsize=S.FS_BODY, length=0)
    ax.set_ylim(-1.3, n + 0.7)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)

    # keep the earlier brief's tag on the control row
    ax.annotate('control', xy=(1.0, y_floor), xycoords=('axes fraction', 'data'),
                xytext=(-1.5, 0), textcoords='offset points',
                ha='right', va='center', fontsize=S.FS_SMALL, color=C['ref'])
    return y_floor


def build(outdir):
    S.apply()                      # Type 42, mathtext bound to the text family,
    plt.rcParams.update(STYLE)     # missing-glyph guard; then the spec's values
    d = load()

    rows = d[d.config != CONTROL].reset_index(drop=True)
    floor = d[d.config == CONTROL].iloc[0]

    # geometry in millimetres -- fixed canvas, no bbox_inches='tight'
    width_mm, height_mm = WIDTH_DOUBLE_MM, 78.0
    left, gap, right_pad = 43.0, 7.0, 4.0
    panel_w = (width_mm - left - gap - right_pad) / 2.0
    y0, panel_h = 18.0, 51.0
    y_title = y0 + panel_h + 2.5

    fig = plt.figure(figsize=S.figsize_mm(width_mm, height_mm))

    for i, (letter, title, dcol, lcol, hcol, scol, xlab, xlim, xticks) in \
            enumerate(PANELS):
        x0 = left + i * (panel_w + gap)
        ax = S.add_axes_mm(fig, x0, y0, panel_w, panel_h)
        forest_panel(ax, rows, floor, dcol, lcol, hcol, xlab, xlim, xticks,
                     show_ylabels=(i == 0))
        # "a   FEP benchmark", with the panel letter bold at 9 pt
        S.text_mm(fig, x0, y_title, letter, fontsize=S.FS_PANEL,
                  fontweight='bold', ha='left', va='baseline')
        S.text_mm(fig, x0 + 4.2, y_title, title, fontsize=S.FS_LABEL,
                  ha='left', va='baseline')

    handles = [
        Line2D([], [], color=C['ns'], marker='o', markersize=4,
               markerfacecolor='white', markeredgecolor=C['ns'],
               markeredgewidth=0.9, linewidth=1.1,
               label='Not significant (open)'),
        Line2D([], [], color=C['sig'], marker='o', markersize=4,
               markerfacecolor=C['sig'], markeredgecolor='white',
               markeredgewidth=0.4, linewidth=1.1,
               label='Interval excludes zero (filled)'),
        Patch(facecolor=C['band'], edgecolor='none',
              label='Detection limit ($\\pm$ half interval width, from zero)'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 1.5 / height_mm),
               ncol=3, frameon=False, handlelength=1.8, columnspacing=1.6)

    pdf, png, w_mm, h_mm = S.save(fig, outdir, 'fig1_forest')
    plt.close(fig)
    return d, pdf, png, w_mm, h_mm


# ------------------------------------------------- stage 4: reconciliation --
def check(d):
    def row(cfg):
        return d[d.config == cfg].iloc[0]

    def eq(got, want, what, dp=4):
        if round(float(got), dp) != want:
            raise AssertionError(
                f'{what}: computed {float(got):.6f} (rounded {round(float(got), dp)}) '
                f'!= expected {want}')

    cols = {'dFEP': ('dFEP_wmPCC', 'dFEP_lo', 'dFEP_hi'),
            'dOOD': ('dOOD_PCC', 'dOOD_lo', 'dOOD_hi')}
    expect = [
        (8,  'dFEP', 0.0210, -0.0249,  0.0868),
        (5,  'dOOD', 0.0018, -0.0183,  0.0204),
        (2,  'dFEP', -0.0286, -0.0656, 0.0058),
        (2,  'dOOD', -0.0007, -0.0177, 0.0182),
        (9,  'dOOD', -0.0463, -0.0740, -0.0193),
        (10, 'dOOD', -0.0329, -0.0658, -0.0023),
        (11, 'dFEP', -0.2067, -0.2809, -0.1444),
    ]
    for cfg, metric, ed, elo, ehi in expect:
        r = row(cfg)
        dc, lc, hc = cols[metric]
        tag = f'{metric} {r["name"]}'
        eq(r[dc], ed, f'{tag} point estimate')
        eq(r[lc], elo, f'{tag} CI lower')
        eq(r[hc], ehi, f'{tag} CI upper')

    sig_rows = []
    for metric, (dc, lc, hc) in cols.items():
        for _, r in d.iterrows():
            if bool(r[f'{dc}__sig']):
                sig_rows.append((metric, r['name'], float(r[dc])))
    if 2 * len(d) != 20:
        raise AssertionError(f'expected 20 comparisons, got {2 * len(d)}')
    if len(sig_rows) != 4:
        raise AssertionError(f'expected 4 significant, got {len(sig_rows)}: {sig_rows}')
    pos = [s for s in sig_rows if s[2] >= 0]
    if pos:
        raise AssertionError(f'expected all significant effects negative, positives: {pos}')

    main = d[d.config.isin(MAIN)]
    for metric, rng, m in [('dFEP', (0.021, 0.058), 0.039),
                           ('dOOD', (0.016, 0.032), 0.019)]:
        dc = cols[metric][0]
        hw = main[f'{dc}__hw']
        eq(hw.min(), rng[0], f'{metric} min half-width', 3)
        eq(hw.max(), rng[1], f'{metric} max half-width', 3)
        eq(np.median(hw), m, f'{metric} median half-width', 3)

    fep_below = main[main['dFEP_wmPCC__below']]
    if len(fep_below) != 9:
        raise AssertionError(
            f'expected 9/9 dFEP below half-width, got {len(fep_below)}; '
            f'failures: {list(main.loc[~main["dFEP_wmPCC__below"], "name"])}')
    ood_below = main[main['dOOD_PCC__below']]
    if len(ood_below) != 7:
        raise AssertionError(f'expected 7/9 dOOD below half-width, got {len(ood_below)}')
    exc = sorted(main.loc[~main['dOOD_PCC__below'], 'name'])
    want_exc = sorted(['bindingnet+dcs+hiqbind', 'bindingnet+dcs+hiqbind+gator_best'])
    if exc != want_exc:
        raise AssertionError(f'dOOD exceptions {exc} != expected {want_exc}')

    r11 = row(11)
    ratio = abs(r11['dFEP_wmPCC']) / r11['dFEP_wmPCC__hw']
    eq(ratio, 3.03, 'pdbbind_only |dFEP| / half-width', 2)
    return ratio, float(np.median(main['dFEP_wmPCC__hw'])), \
        float(np.median(main['dOOD_PCC__hw']))


def write_values(d, outdir):
    rows = []
    for metric, (dc, lc, hc) in [('FEP_wmPCC', ('dFEP_wmPCC', 'dFEP_lo', 'dFEP_hi')),
                                 ('OOD_PCC', ('dOOD_PCC', 'dOOD_lo', 'dOOD_hi'))]:
        for _, r in d.iterrows():
            rows.append({
                'config': int(r['config']),
                'config_name': r['name'],
                'panel_label': LABEL[int(r['config'])],
                'metric': metric,
                'delta': r[dc],
                'CI_lo': r[lc],
                'CI_hi': r[hc],
                'half_width': r[f'{dc}__hw'],
                'abs_delta': r[f'{dc}__abs'],
                'significant': bool(r[f'{dc}__sig']),
                'below_halfwidth': bool(r[f'{dc}__below']),
            })
    out = Path(outdir) / 'fig1_values.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=str(HERE / 'out'))
    a = ap.parse_args()

    d, pdf, png, w_mm, h_mm = build(a.outdir)
    ratio, med_fep, med_ood = check(d)
    vals = write_values(d, a.outdir)

    print(f'font in use      : {S.resolved_font()}')
    print(f'figure           : {w_mm:.1f} x {h_mm:.1f} mm, broken axis = False')
    print(f'median half-width: dFEP {med_fep:.4f}  dOOD {med_ood:.4f}')
    print(f'control ratio    : {ratio:.4f}')
    print(f'wrote            : {pdf}\n                   {png}\n                   {vals}')
    print('stage-4 reconciliation: all assertions passed')


if __name__ == '__main__':
    main()
