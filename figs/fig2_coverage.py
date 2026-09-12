#!/usr/bin/env python
"""
Figure 2 -- OOD target coverage by training source.

Panel a: horizontal bars, OOD target coverage (%) for the four single sources
         and the two unions.
Panel b: three-set Venn over the 94 unique OOD UniProt targets.

The shipped artefacts under 09_uniprot_coverage/ are aggregate-only: they give
|P|, |H|, |G| and the union sizes, which pins |P n H| = 42 but leaves the
interior of the Venn underdetermined.  So the per-target membership is
recomputed here with exactly the logic of scripts/09_uniprot_coverage.py, from
exactly its inputs, and is then checked against every aggregate that script
already published.  Nothing is invented; the recomputation is written out to
out/ood_target_membership.csv so the missing intermediate exists from now on.
"""
import argparse
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E = ROOT / 'experiments'
IDX = ROOT / 'datasets' / 'index'
COV = E / '09_uniprot_coverage'

# keys as they appear in uniprot_coverage.csv (source names are Chinese)
K_PDBBIND = 'PDBbind'
K_HIQBIND = 'HiQBind'
K_GATOR_U = 'Gator(未筛选全量)'
K_GATOR_F = 'Gator(筛选版100%)'
K_UNION_PH = 'PDBbind ∪ HiQBind'
K_UNION_ALL = 'PDBbind ∪ HiQBind ∪ Gator'

# One colour per data source, taken from the shared Scheme B palette so the
# three figures cannot drift apart.  Unions use the reference grey rather than
# the pale band colour, which Figure 1 already spends on its detection limit.
C = {
    'pdbbind':    S.STACEY_C['pdbbind'],
    'hiqbind':    S.STACEY_C['hiqbind'],
    'gator':      S.STACEY_C['gator'],
    'union':      S.STACEY_C['ref'],
    'union_edge': S.darken(S.STACEY_C['ref'], 0.35),
}

# panel a rows, top to bottom
BARS = [
    (K_PDBBIND,   'PDBbind',                          'single', C['pdbbind']),
    (K_HIQBIND,   'HiQBind',                          'single', C['hiqbind']),
    (K_GATOR_U,   'GatorAffinity-DB\n(unfiltered)',   'single', C['gator']),
    (K_GATOR_F,   'GatorAffinity-DB\n(filtered)',     'single', C['gator']),
    (K_UNION_PH,  'PDBbind + HiQBind',              'union',  C['union']),
    (K_UNION_ALL, 'PDBbind + HiQBind + Gator',        'union',  C['union']),
]


# --------------------------------------------------------------------------
# recomputation -- identical logic to scripts/09_uniprot_coverage.py
# --------------------------------------------------------------------------
def load_sifts():
    """PDB ID (4-char, lower) -> set(UniProt accession)."""
    m = {}
    with gzip.open(IDX / '_mappings' / 'pdb_chain_uniprot.tsv.gz', 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('PDB\t'):
                continue
            p = line.split('\t')
            if len(p) < 3:
                continue
            m.setdefault(p[0].strip().lower(), set()).add(p[2].strip())
    return m


def norm(s):
    """Drop isoform suffix, upper-case, strip."""
    s = str(s).strip().upper()
    return s.split('-')[0] if s and s != 'NAN' else ''


def expand(series):
    """One cell may carry several accessions, separated by ';' or ','."""
    out = set()
    for v in series.dropna():
        for tok in str(v).replace(',', ';').split(';'):
            u = norm(tok)
            if u:
                out.add(u)
    return out


def recompute():
    sifts = load_sifts()

    ood = pd.read_csv(IDX / '_testsets' / 'index_oodtest.csv')
    ood = ood[ood.split == 'test'].reset_index(drop=True)
    ood_pdb = [str(x).strip().lower()[:4] for x in ood.PDB_code]
    assert len(ood_pdb) == 295, len(ood_pdb)

    pdb_to_uni = {}
    ood_uni = set()
    n_mapped = 0
    for p in ood_pdb:
        u = {norm(x) for x in sifts.get(p, set())}
        u.discard('')
        pdb_to_uni[p] = u
        if u:
            n_mapped += 1
            ood_uni |= u

    pb = pd.read_csv(E / '01_baseline_index' / 'pdbbind_clean.csv')
    pb_uni = set()
    for p in [str(x).strip().lower()[:4] for x in pb.pdb_id]:
        u = {norm(x) for x in sifts.get(p, set())}
        u.discard('')
        pb_uni |= u

    hq = pd.read_csv(E / '02_hiqbind' / 'hiqbind_dedup.csv')
    hq_uni = expand(hq['Protein UniProtID'])

    ga = pd.read_csv(E / '03_gator' / 'gator_dedup.csv')
    gcol = 'UniProt (SwissProt) Primary ID of Target Chain'
    ga_uni = expand(ga[gcol])
    gaf_uni = expand(ga.loc[ga.in_filtered_100, gcol])

    acc = {K_PDBBIND: len(pb_uni), K_HIQBIND: len(hq_uni),
           K_GATOR_U: len(ga_uni), K_GATOR_F: len(gaf_uni),
           K_UNION_PH: len(pb_uni | hq_uni),
           K_UNION_ALL: len(pb_uni | hq_uni | ga_uni)}
    sets = {K_PDBBIND: pb_uni, K_HIQBIND: hq_uni,
            K_GATOR_U: ga_uni, K_GATOR_F: gaf_uni,
            K_UNION_PH: pb_uni | hq_uni,
            K_UNION_ALL: pb_uni | hq_uni | ga_uni}

    # per-target boolean table over the 94 OOD targets
    members = pd.DataFrame({'uniprot': sorted(ood_uni)})
    for key, col in [(K_PDBBIND, 'in_pdbbind'), (K_HIQBIND, 'in_hiqbind'),
                     (K_GATOR_U, 'in_gator_unfiltered'),
                     (K_GATOR_F, 'in_gator_filtered')]:
        members[col] = members.uniprot.isin(sets[key])
    members['in_any_source'] = (members.in_pdbbind | members.in_hiqbind
                                | members.in_gator_unfiltered)

    # OOD complexes whose target is covered, per source and per union
    hits = {}
    for key, uni in sets.items():
        hits[key] = sum(1 for p in ood_pdb if pdb_to_uni[p] & uni)

    return dict(n_pdb=len(ood_pdb), n_mapped=n_mapped, ood_uni=ood_uni,
                sets=sets, acc=acc, members=members, hits=hits)


def venn_regions(m):
    """The seven Venn regions plus the uncovered count, from the boolean table."""
    P, H, G = m.in_pdbbind, m.in_hiqbind, m.in_gator_unfiltered
    return {
        '100': int(( P & ~H & ~G).sum()),   # PDBbind only
        '010': int((~P &  H & ~G).sum()),   # HiQBind only
        '110': int(( P &  H & ~G).sum()),
        '001': int((~P & ~H &  G).sum()),   # Gator only -- must be 0
        '101': int(( P & ~H &  G).sum()),
        '011': int((~P &  H &  G).sum()),
        '111': int(( P &  H &  G).sum()),
        'none': int((~P & ~H & ~G).sum()),
    }


# --------------------------------------------------------------------------
# stage 4 -- reconciliation
# --------------------------------------------------------------------------
def check(r, reg):
    def eq(got, want, what):
        if int(got) != int(want):
            raise AssertionError(f'{what}: computed {got} != expected {want}')

    eq(r['n_pdb'], 295, 'OOD complexes')
    eq(r['n_mapped'], 285, 'OOD complexes mapped to UniProt')
    eq(len(r['ood_uni']), 94, 'unique OOD targets')

    inter = {k: len(r['ood_uni'] & v) for k, v in r['sets'].items()}
    for key, n_acc, n_tgt in [
            (K_PDBBIND,   4670, 60), (K_HIQBIND,   2592, 43),
            (K_GATOR_U,   1313, 15), (K_GATOR_F,   1219, 15),
            (K_UNION_PH,  5469, 61), (K_UNION_ALL, 6226, 61)]:
        eq(r['acc'][key], n_acc, f'{key} accessions')
        eq(inter[key], n_tgt, f'{key} OOD targets covered')

    pct = inter[K_PDBBIND] / 94 * 100
    if round(pct, 1) != 63.8:
        raise AssertionError(f'PDBbind target coverage {pct:.3f}% != 63.8%')
    for key, want in [(K_GATOR_U, 16.0), (K_GATOR_F, 16.0), (K_UNION_ALL, 64.9),
                      (K_HIQBIND, 45.7)]:
        got = inter[key] / 94 * 100
        if round(got, 1) != want:
            raise AssertionError(f'{key} target coverage {got:.3f}% != {want}%')

    # the filtered Gator hits the very same 15 targets as the unfiltered one
    gu = r['ood_uni'] & r['sets'][K_GATOR_U]
    gf = r['ood_uni'] & r['sets'][K_GATOR_F]
    if gu != gf:
        raise AssertionError(
            'filtered and unfiltered Gator cover different OOD targets: '
            f'only-unfiltered={sorted(gu - gf)} only-filtered={sorted(gf - gu)}')

    eq(r['hits'][K_PDBBIND], 228, 'OOD complexes covered by PDBbind')

    # the recomputation must reproduce the shipped aggregates exactly
    shipped = pd.read_csv(COV / 'uniprot_coverage.csv').set_index('source')
    for key in r['sets']:
        eq(r['acc'][key], shipped.loc[key, 'n_uniprot_source'],
           f'{key} accessions vs uniprot_coverage.csv')
        eq(inter[key], shipped.loc[key, 'n_intersect'],
           f'{key} intersect vs uniprot_coverage.csv')
    shipped_pdb = pd.read_csv(COV / 'uniprot_coverage_by_pdb.csv').set_index('source')
    for key in shipped_pdb.index:
        eq(r['hits'][key], shipped_pdb.loc[key, 'n_hit'],
           f'{key} complexes vs uniprot_coverage_by_pdb.csv')

    # Venn interior
    total = sum(v for k, v in reg.items() if k != 'none') + reg['none']
    eq(total, 94, 'Venn regions summing to the OOD target count')
    eq(reg['100'] + reg['110'] + reg['101'] + reg['111'], 60, 'PDBbind regions')
    eq(reg['010'] + reg['110'] + reg['011'] + reg['111'], 43, 'HiQBind regions')
    eq(reg['001'] + reg['101'] + reg['011'] + reg['111'], 15, 'Gator regions')
    eq(reg['none'], 94 - 61, 'targets covered by no source')

    if reg['001'] != 0:
        raise AssertionError(
            'STOP -- the Gator-only Venn region is %d, not 0. The paper\'s core '
            'claim (GatorAffinity-DB contributes no OOD target that PDBbind or '
            'HiQBind does not already cover) does NOT hold on this data. '
            'Targets in Gator alone: %s'
            % (reg['001'], sorted(
                r['members'].loc[r['members'].in_gator_unfiltered
                                 & ~r['members'].in_pdbbind
                                 & ~r['members'].in_hiqbind, 'uniprot'])))
    return inter


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------
def build(r, reg, inter, outdir):
    S.apply()
    from matplotlib_venn import venn3, venn3_circles

    width_mm, height_mm = S.DOUBLE_COL_MM, 88.0
    panel_top, y_title = 78.0, 82.0     # 4 mm from title baseline to panel top
    fig = plt.figure(figsize=S.figsize_mm(width_mm, height_mm))

    # ---------------- panel a ----------------
    ax = S.add_axes_mm(fig, 38.0, 18.0, 52.0, panel_top - 18.0)
    ys = np.arange(len(BARS))
    for y, (key, label, kind, colour) in zip(ys, BARS):
        pctv = inter[key] / 94 * 100
        ax.barh(y, pctv, height=0.62, zorder=3, color=colour,
                edgecolor=C['union_edge'] if kind == 'union' else S.INK,
                linewidth=0.6 if kind == 'union' else 0.5)
        ax.annotate(f'{r["acc"][key]:,}',
                    xy=(pctv, y), xytext=(2.5, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=S.FS_SMALL,
                    color=S.INK, annotation_clip=False)

    ax.set_yticks(ys)
    ax.set_yticklabels([b[1] for b in BARS], fontsize=S.FS_BODY)
    ax.set_ylim(len(BARS) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('OOD target coverage (%)', fontsize=S.FS_LABEL)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    for x in (20, 40, 60, 80, 100):
        ax.plot([x, x], [-0.5, len(BARS) - 0.5], color=S.GREY_XLT, lw=0.4, zorder=0)

    # "single source" is shown as the three source colours side by side, so
    # the two entries are told apart by colour alone -- no hatching.
    single_handle = tuple(Patch(facecolor=c, edgecolor=S.INK, lw=0.4)
                          for c in (C['pdbbind'], C['hiqbind'], C['gator']))
    union_handle = Patch(facecolor=C['union'], edgecolor=C['union_edge'], lw=0.6)
    fig.legend(handles=[single_handle, union_handle],
               labels=['single source', 'union'],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
               loc='lower left', bbox_to_anchor=(38.0 / width_mm, 1.5 / height_mm),
               ncol=2, fontsize=S.FS_SMALL, frameon=False, handlelength=2.4,
               handleheight=1.0, columnspacing=1.6)

    # ---------------- panel b ----------------
    axv = S.add_axes_mm(fig, 97.0, 8.0, 82.0, panel_top - 8.0)
    axv.set_axis_off()

    subsets = (reg['100'], reg['010'], reg['110'],
               reg['001'], reg['101'], reg['011'], reg['111'])
    colours = (C['pdbbind'], C['hiqbind'], C['gator'])
    REGION_TEXT = '#333333'
    v = venn3(subsets=subsets, ax=axv,
              set_labels=('PDBbind', 'HiQBind', 'GatorAffinity-DB\n(unfiltered)'),
              set_colors=colours, alpha=0.12,
              subset_label_formatter=lambda x: str(int(x)))

    # With three heavily nested circles the translucent fills stack into one
    # muddy neutral, so the outlines carry the distinction: full source colour,
    # heavier stroke, and three dash patterns so hue is not the only channel.
    c = venn3_circles(subsets=subsets, ax=axv, linewidth=1.4)
    for circ, col, ls in zip(c, colours,
                             ['solid', (0, (4, 1.6)), (0, (1, 1.4))]):
        circ.set_edgecolor(col)
        circ.set_linestyle(ls)

    for lab in v.set_labels:
        if lab is not None:
            lab.set_fontsize(S.FS_BODY)

    (px, py), pr = c[0].center, c[0].radius
    (hx, hy), hr = c[1].center, c[1].radius
    (gx, gy), gr = c[2].center, c[2].radius

    # Gator sits entirely inside PDBbind (|P n G| = |G| = 15), so the
    # Gator-only and (HiQBind n Gator)-only regions have no drawable area --
    # matplotlib-venn returns no label for them at all.  The HiQBind-only
    # region is a sliver too thin to hold a digit, so its label is blanked.
    # The four regions with real area keep their counts.
    for sid in ('100', '110', '101', '111'):
        lab = v.get_label_by_id(sid)
        if lab is not None:
            lab.set_fontsize(S.FS_BODY)
            lab.set_text(str(reg[sid]))
            lab.set_color(REGION_TEXT)
    lab010 = v.get_label_by_id('010')
    if lab010 is not None:
        lab010.set_text('')

    lab100 = v.get_label_by_id('100')
    if lab100 is not None:                       # centre it in the P-only crescent
        lab100.set_position((px - 0.72 * pr, py + 0.29 * pr))

    # Universe frame: the 94 OOD targets are the rectangle, the three sources
    # are the circles, so "inside the frame but outside every circle" is
    # literally the 33 targets no source covers.  Bounds are derived from the
    # fitted circles and leave room for the set labels matplotlib-venn puts
    # outside them.
    rect_l = px - pr - 0.10
    rect_r = hx + 1.34 * hr + 0.09          # clears the HiQBind-only leader
    rect_t = py + pr + 0.21                 # clears the PDBbind / HiQBind labels
    rect_b = gy - gr - 0.24                 # clears the GatorAffinity-DB label
    axv.add_patch(Rectangle((rect_l, rect_b), rect_r - rect_l, rect_t - rect_b,
                            fill=False, edgecolor=S.STACEY_C['ref'],
                            linewidth=1.0, zorder=0))
    pad = 0.045
    axv.text(rect_l + pad, rect_t - pad, f'{len(r["ood_uni"])} OOD targets',
             ha='left', va='top', fontsize=S.FS_BODY, color=S.INK)
    axv.text(rect_r - pad, rect_b + pad, f'{reg["none"]} in no source',
             ha='right', va='bottom', fontsize=S.FS_BODY, color=S.INK)

    callout_y = rect_b - 0.09
    axv.set_xlim(rect_l - 0.05, rect_r + 0.05)
    axv.set_ylim(callout_y - 0.07, rect_t + 0.05)

    # The figure's point, stated once below the Gator circle.  The leader is
    # drawn by hand rather than by annotate(): annotate routes from the text
    # box's centre, which sends the line straight through the Gator set label.
    # Both endpoints are derived from the fitted circle.
    anchor = (gx + gr * np.cos(np.deg2rad(205)),
              gy + gr * np.sin(np.deg2rad(205)))
    axv.plot([gx - 1.40 * gr, anchor[0]], [callout_y + 0.05, anchor[1]],
             color=S.GREY, lw=0.6, zorder=5, solid_capstyle='butt')
    axv.text(gx, callout_y,
             'GatorAffinity-DB adds no target outside PDBbind',
             ha='center', va='center', fontsize=S.FS_BODY, color=S.INK)

    # HiQBind-only is a real region but a sliver far too thin to hold a digit,
    # so its count is repeated outside on a leader.  Same hand-routed approach:
    # anchor at the midpoint between the PDBbind and HiQBind right-hand edges,
    # text out to the right, well clear of the HiQBind set label above.
    sliver_x = ((px + pr) + (hx + hr)) / 2.0
    tx, ty = hx + 1.34 * hr, hy + 0.34 * hr
    axv.plot([tx - 0.035, sliver_x], [ty - 0.012, hy], color=S.GREY, lw=0.6,
             zorder=5, solid_capstyle='butt')
    axv.text(tx, ty, str(reg['010']), ha='left', va='center',
             fontsize=S.FS_BODY, color=REGION_TEXT)

    # panel headings in Figure 1's format: bold letter, space, title,
    # left-aligned on each panel's own left edge and sharing one baseline
    # Panel a's heading lines up with the leftmost extent of the panel, tick
    # labels included, so it matches panel b (whose axes carry no y labels).
    # Measured from the rendered labels rather than guessed.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    a_left = min(t.get_window_extent(renderer=rend).x0
                 for t in ax.get_yticklabels()) / fig.dpi * 25.4

    for x0, letter, title in ((a_left, 'a', 'Target coverage by source'),
                              (97.0, 'b', 'Distribution of the 94 OOD targets')):
        S.text_mm(fig, x0, y_title, letter, fontsize=S.FS_PANEL,
                  fontweight='bold', ha='left', va='baseline')
        S.text_mm(fig, x0 + 4.2, y_title, title, fontsize=S.FS_LABEL,
                  ha='left', va='baseline')

    pdf, png, w_mm, h_mm = S.save(fig, outdir, 'fig2_coverage')
    plt.close(fig)
    return pdf, png, w_mm, h_mm


def write_values(r, inter, outdir):
    rows = []
    for key, label, _kind, _colour in BARS:
        rows.append({
            'source': key,
            'panel_label': label.replace('\n', ' '),
            'accessions': r['acc'][key],
            'targets_covered': inter[key],
            'target_pct': inter[key] / 94 * 100,
            'complexes_covered': r['hits'][key],
            'complex_pct': r['hits'][key] / 295 * 100,
        })
    out = Path(outdir) / 'fig2_values.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default=str(HERE / 'out'))
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    r = recompute()
    reg = venn_regions(r['members'])
    inter = check(r, reg)

    mem = outdir / 'ood_target_membership.csv'
    r['members'].to_csv(mem, index=False)

    pdf, png, w_mm, h_mm = build(r, reg, inter, outdir)
    vals = write_values(r, inter, outdir)

    print(f'font in use   : {S.resolved_font()}')
    print(f'figure        : {w_mm:.1f} x {h_mm:.1f} mm')
    print(f'OOD           : {r["n_mapped"]}/{r["n_pdb"]} complexes mapped '
          f'-> {len(r["ood_uni"])} unique targets')
    print('Venn regions  : ' + '  '.join(f'{k}={v}' for k, v in reg.items()))
    print(f'Gator-only    : {reg["001"]}  (core claim holds)')
    print(f'wrote         : {pdf}\n                {png}\n'
          f'                {vals}\n                {mem}')
    print('stage-4 reconciliation: all assertions passed')


if __name__ == '__main__':
    main()
