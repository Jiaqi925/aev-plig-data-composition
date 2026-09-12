#!/usr/bin/env python
"""
Figure 3 -- OOD similarity-stratified performance, dumbbell plot.

One panel, single column, eleven rows (configs 1-11) in the same top-to-bottom
order as Figure 1.  All eleven are drawn alike: in the stratified analysis the
configurations are peers, config 11 carries no control role here, and a
separator would imply one.  (Figure 1 does tag it, because there it is the
control for the paired differences.)

Each row carries two point estimates with their BCa 95% CIs:
  high-similarity stratum  filled, Scheme A blue
  low-similarity stratum   open,   Scheme A terracotta
joined by a thin grey connector, so the gap between the two strata is the
thing the eye measures.

Typography and palette come from style.STACEY_STYLE / style.STACEY_C, the
same spec Figure 1 uses.  No value is hard-coded: every PCC, interval and
width is read from strat_ci.csv; the literals that remain are the
reconciliation targets asserted in check().
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / 'experiments' / '10_strat_ci' / 'strat_ci.csv'

STYLE = S.STACEY_STYLE
C = S.STACEY_C

HIGH_COLOUR = C['high']       # #2F6690
LOW_COLOUR = C['low']         # #C4703F
REF_COLOUR = C['ref']         # #909090

# config -> row label, matching Figure 1 (config 1 is the baseline itself)
# No "+" prefix here: this panel plots absolute PCC, not a difference, and a
# leading "+" reads as an increment.  Figure 1 keeps it, because there the rows
# really are differences against the baseline.
LABEL = {
    1:  'Baseline',
    2:  'HiQBind',
    3:  'Gator filtered 30%',
    4:  'Gator filtered 60%',
    5:  'Gator filtered 100%',
    6:  'Gator unfiltered 30%',
    7:  'Gator unfiltered 60%',
    8:  'Gator unfiltered 100%',
    9:  'BindingNet+DCS+HiQBind',
    10: 'BindingNet+DCS+HiQBind+Gator',
    11: 'PDBbind only',
}
XLIM = (-0.12, 0.90)
XTICKS = [0.0, 0.2, 0.4, 0.6, 0.8]


# ----------------------------------------------------------------- data ----
def load():
    d = pd.read_csv(SRC)
    assert len(d) == 22, f'expected 11 configs x 2 strata = 22 rows, got {len(d)}'
    assert sorted(d.config.unique()) == list(range(1, 12)), sorted(d.config.unique())
    assert set(d.layer) == {'high', 'low'}, set(d.layer)
    assert set(d.method) == {'BCa'}, f'expected BCa intervals only, got {set(d.method)}'

    # Recompute the interval width instead of trusting the shipped column, then
    # require the two to agree -- the same double-check Figure 1 runs on its
    # significance flags.
    d['width'] = d.pcc_hi - d.pcc_lo
    bad = np.nonzero(~np.isclose(d['width'], d.pcc_width, atol=1e-12))[0]
    if len(bad):
        msg = ['pcc_width mismatch:']
        for i in bad:
            msg.append(f'  config {d.config[i]} {d.layer[i]}: shipped '
                       f'{d.pcc_width[i]:.9f} but hi-lo = {d["width"][i]:.9f}')
        raise AssertionError('\n'.join(msg))

    d['includes_zero'] = (d.pcc_lo <= 0) & (d.pcc_hi >= 0)
    return d


def wide(d):
    """One row per config, high and low side by side."""
    hi = d[d.layer == 'high'].set_index('config')
    lo = d[d.layer == 'low'].set_index('config')
    return hi.sort_index(), lo.sort_index()


# -------------------------------------------------------------- drawing ----
def build(d, outdir):
    S.apply()
    plt.rcParams.update(STYLE)

    hi, lo = wide(d)
    configs = list(hi.index)               # all eleven, evenly spaced
    n = len(configs)
    y_of = {c: n - i for i, c in enumerate(configs)}

    width_mm, height_mm = S.SINGLE_COL_MM, 77.0
    left, right_pad = 41.0, 4.0
    panel_w = width_mm - left - right_pad
    y0, panel_h = 17.0, 56.0

    fig = plt.figure(figsize=S.figsize_mm(width_mm, height_mm))
    ax = S.add_axes_mm(fig, left, y0, panel_w, panel_h)

    ax.axvline(0, color=REF_COLOUR, linewidth=0.7, linestyle=(0, (4, 3)), zorder=2)

    for cfg in hi.index:
        y = y_of[cfg]
        h, l = hi.loc[cfg], lo.loc[cfg]

        # connector between the two point estimates
        ax.plot([l.pcc, h.pcc], [y, y], color=REF_COLOUR, linewidth=0.7,
                solid_capstyle='butt', zorder=3)

        # BCa 95% intervals
        ax.plot([l.pcc_lo, l.pcc_hi], [y, y], color=LOW_COLOUR, linewidth=1.1,
                solid_capstyle='butt', zorder=4)
        ax.plot([h.pcc_lo, h.pcc_hi], [y, y], color=HIGH_COLOUR, linewidth=1.1,
                solid_capstyle='butt', zorder=5)

        # low stratum open, high stratum filled
        ax.scatter([l.pcc], [y], s=S.MARKER_SIZE, facecolors='white',
                   edgecolors=LOW_COLOUR, linewidths=0.9, zorder=6)
        ax.scatter([h.pcc], [y], s=S.MARKER_SIZE, color=HIGH_COLOUR,
                   edgecolors='white', linewidths=0.4, zorder=7)

    ax.set_yticks([y_of[c] for c in configs])
    ax.set_yticklabels([LABEL[c] for c in configs])
    ax.tick_params(axis='y', labelsize=S.FS_BODY, length=0)
    ax.set_ylim(0.3, n + 0.7)
    ax.set_xlim(*XLIM)
    ax.set_xticks(XTICKS)
    ax.set_xlabel('PCC (OOD Test)')

    handles = [
        Line2D([], [], color=HIGH_COLOUR, marker='o', markersize=4,
               markerfacecolor=HIGH_COLOUR, markeredgecolor='white',
               markeredgewidth=0.4, linewidth=1.1,
               label='High-similarity stratum'),
        Line2D([], [], color=LOW_COLOUR, marker='o', markersize=4,
               markerfacecolor='white', markeredgecolor=LOW_COLOUR,
               markeredgewidth=0.9, linewidth=1.1,
               label='Low-similarity stratum'),
    ]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 1.5 / height_mm),
               ncol=2, frameon=False, handlelength=1.8, columnspacing=1.6)

    pdf, png, w_mm, h_mm = S.save(fig, outdir, 'fig3_strat')
    plt.close(fig)
    return pdf, png, w_mm, h_mm


# ------------------------------------------------- reconciliation ----------
def check(d):
    def eq(got, want, what, dp=4):
        if round(float(got), dp) != want:
            raise AssertionError(
                f'{what}: computed {float(got):.6f} (rounded {round(float(got), dp)}) '
                f'!= expected {want}')

    hi, lo = wide(d)

    eq(hi.pcc.min(), 0.7536, 'high stratum min PCC')
    eq(hi.pcc.max(), 0.8190, 'high stratum max PCC')
    eq(lo.pcc.min(), 0.2114, 'low stratum min PCC')
    eq(lo.pcc.max(), 0.3985, 'low stratum max PCC')

    # every config must separate the two strata
    overlap = [c for c in hi.index if not lo.loc[c].pcc_hi < hi.loc[c].pcc_lo]
    if overlap:
        detail = '; '.join(
            f'config {c}: low_hi={lo.loc[c].pcc_hi:.4f} >= high_lo={hi.loc[c].pcc_lo:.4f}'
            for c in overlap)
        raise AssertionError(f'expected 11/11 non-overlapping strata, {len(overlap)} '
                             f'overlap -- {detail}')
    if len(hi) != 11:
        raise AssertionError(f'expected 11 configs, got {len(hi)}')

    zero = sorted(lo.index[lo.includes_zero])
    if len(zero) != 7:
        raise AssertionError(
            f'expected exactly 7 low-stratum intervals covering 0, got '
            f'{len(zero)}: configs {zero}')

    eq(hi['width'].min(), 0.093, 'high stratum min CI width', 3)
    eq(hi['width'].max(), 0.118, 'high stratum max CI width', 3)
    eq(lo['width'].min(), 0.518, 'low stratum min CI width', 3)
    eq(lo['width'].max(), 0.603, 'low stratum max CI width', 3)

    # the x axis has to actually contain the lowest interval bound
    lo_bound = float(d.pcc_lo.min())
    eq(lo_bound, -0.0737, 'lowest low-stratum CI bound')
    if XLIM[0] > lo_bound or XLIM[1] < float(d.pcc_hi.max()):
        raise AssertionError(
            f'x range {XLIM} does not contain the data '
            f'[{lo_bound:.4f}, {float(d.pcc_hi.max()):.4f}]')
    return zero, lo_bound


def write_values(d, outdir):
    out = Path(outdir) / 'fig3_values.csv'
    d.assign(stratum=d.layer)[
        ['config', 'stratum', 'n', 'pcc', 'pcc_lo', 'pcc_hi', 'width',
         'includes_zero']
    ].rename(columns={'pcc': 'PCC', 'pcc_lo': 'CI_lo', 'pcc_hi': 'CI_hi'}) \
        .sort_values(['config', 'stratum'], ascending=[True, True]) \
        .to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=str(HERE / 'out'))
    a = ap.parse_args()

    d = load()
    pdf, png, w_mm, h_mm = build(d, a.outdir)
    zero, lo_bound = check(d)
    vals = write_values(d, a.outdir)

    hi, lo = wide(d)
    print(f'font in use   : {S.resolved_font()}')
    print(f'figure        : {w_mm:.2f} x {h_mm:.2f} mm, single column, one panel')
    print(f'high stratum  : PCC {hi.pcc.min():.4f}-{hi.pcc.max():.4f}, '
          f'CI width {hi["width"].min():.3f}-{hi["width"].max():.3f}, n={sorted(set(hi.n))}')
    print(f'low stratum   : PCC {lo.pcc.min():.4f}-{lo.pcc.max():.4f}, '
          f'CI width {lo["width"].min():.3f}-{lo["width"].max():.3f}, n={sorted(set(lo.n))}')
    print(f'strata separated in 11/11 configs; lowest CI bound {lo_bound:.4f}')
    print(f'low CI covers 0 in {len(zero)} configs: {zero}')
    print(f'wrote         : {pdf}\n                {png}\n                {vals}')
    print('reconciliation: all assertions passed')


if __name__ == '__main__':
    main()
