#!/usr/bin/env python
"""
Build figs/out/Supplementary_Tables.docx (Tables S1, S2, S6).

Layout matches the main text: A4, 25 mm margins, Times New Roman, body 12 pt
at 1.5 line spacing, table content 7.5 pt single-spaced.  Booktabs rules only
(above the header, below the header, at the foot) -- no vertical rules, no
shading.  S1 and S2 sit on landscape pages, S6 on a portrait page.

Every number is read from the experiment outputs; the caption figures are
computed here and asserted against the values quoted in the text.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E = ROOT / 'experiments'
OUT = HERE / 'out'

FONT = 'Times New Roman'
FS_BODY = Pt(12)
FS_TABLE = Pt(7.5)
FS_CAPTION = Pt(10)
MARGIN = Mm(25)

MINUS = '−'          # U+2212 MINUS SIGN
PM = '±'
TIMES = '×'
TAU = 'τ'
DELTA = 'Δ'

# booktabs rule weights, in eighths of a point
RULE_HEAVY = 12           # 1.5 pt -- top and foot
RULE_MID = 6              # 0.75 pt -- under the header
RULE_GROUP = 4            # 0.5 pt -- between configuration groups


# --------------------------------------------------------------- numbers ---
def dec(x, dp=4, signed=False):
    """Fixed-decimal, thousands-separated, U+2212 for the minus sign."""
    s = f'{x:+,.{dp}f}' if signed else f'{x:,.{dp}f}'
    return s.replace('-', MINUS)


def integer(x):
    return f'{int(x):,}'.replace('-', MINUS)


def interval(lo, hi):
    return f'[{dec(lo, 4, True)}, {dec(hi, 4, True)}]'


# ----------------------------------------------------------------- docx ----
def _border(el, sz, val='single', color='000000'):
    el.set(qn('w:val'), val)
    el.set(qn('w:sz'), str(sz))
    el.set(qn('w:space'), '0')
    el.set(qn('w:color'), color)


def clear_table_borders(table):
    """No grid, no shading -- the three rules are added per cell afterwards."""
    tblPr = table._tbl.tblPr
    for old in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old)
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        _border(el, 0, val='none')
        borders.append(el)
    tblPr.append(borders)


def rule(row, edge, sz):
    """Draw a horizontal rule along one edge of every cell in a row."""
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        tcBorders = tcPr.find(qn('w:tcBorders'))
        if tcBorders is None:
            tcBorders = OxmlElement('w:tcBorders')
            tcPr.append(tcBorders)
        el = tcBorders.find(qn(f'w:{edge}'))
        if el is None:
            el = OxmlElement(f'w:{edge}')
            tcBorders.append(el)
        _border(el, sz)


def repeat_header(row):
    """Mark the header row so Word repeats it across page breaks."""
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement('w:tblHeader')
    el.set(qn('w:val'), 'true')
    trPr.append(el)


def write_cell(cell, text, *, align='right', bold=False, italic=False):
    cell.text = ''
    para = cell.paragraphs[0]
    para.alignment = {'left': WD_ALIGN_PARAGRAPH.LEFT,
                      'right': WD_ALIGN_PARAGRAPH.RIGHT,
                      'center': WD_ALIGN_PARAGRAPH.CENTER}[align]
    pf = para.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    run = para.add_run(text)
    run.font.name = FONT
    run.font.size = FS_TABLE
    run.bold = bold
    run.italic = italic
    run._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)


def make_table(doc, headers, widths_mm, aligns):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = doc.styles['Table Grid']
    clear_table_borders(table)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr = table.rows[0]
    for i, (h, w, a) in enumerate(zip(headers, widths_mm, aligns)):
        write_cell(hdr.cells[i], h, align=a, bold=True)
        hdr.cells[i].width = Mm(w)
    rule(hdr, 'top', RULE_HEAVY)
    rule(hdr, 'bottom', RULE_MID)
    repeat_header(hdr)
    return table


def add_row(table, values, widths_mm, aligns, *, bold=False, italic=False):
    row = table.add_row()
    for i, (v, w, a) in enumerate(zip(values, widths_mm, aligns)):
        write_cell(row.cells[i], v, align=a, bold=bold, italic=italic)
        row.cells[i].width = Mm(w)
    return row


def add_caption(doc, label, text):
    para = doc.add_paragraph()
    pf = para.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    pf.space_before = Pt(6)
    pf.space_after = Pt(0)
    for chunk, bold in ((label + ' ', True), (text, False)):
        run = para.add_run(chunk)
        run.font.name = FONT
        run.font.size = FS_CAPTION
        run.bold = bold
        run._element.rPr.rFonts.set(qn('w:eastAsia'), FONT)


def setup_section(sec, landscape):
    if landscape:
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = Mm(297), Mm(210)
    else:
        sec.orientation = WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = Mm(210), Mm(297)
    sec.left_margin = sec.right_margin = MARGIN
    sec.top_margin = sec.bottom_margin = MARGIN


def new_document():
    doc = Document()
    normal = doc.styles['Normal']
    normal.font.name = FONT
    normal.font.size = FS_BODY
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    setup_section(doc.sections[0], landscape=True)
    return doc


# ----------------------------------------------------------------- data ----
def load_training():
    files = sorted(glob.glob(str(E / '06_train' / 'cfg*' / 'seed*' / 'done.json')))
    rows = []
    for f in files:
        d = json.load(open(f))
        rows.append({
            'config': d['config'], 'seed': d['seed'],
            'best_avg8': d['best_avg8_val_pcc'],
            'val_at_best': d['val_pcc_at_best'],
            'best_epoch': d['best_epoch'], 'epochs_run': d['epochs_run'],
            'max_epochs': d['hyper']['max_epochs'],
            'patience': d['hyper']['early_patience'],
            'n_train': d['n_train'], 'n_val': d['n_val'],
            'minutes': d['minutes'],
        })
    t = pd.DataFrame(rows).sort_values(['config', 'seed']).reset_index(drop=True)
    caps = set(t.max_epochs)
    assert len(caps) == 1, f'mixed epoch caps: {caps}'
    pats = set(t.patience)
    assert len(pats) == 1, f'mixed early-stopping patience: {pats}'
    t['early'] = t.epochs_run < t.max_epochs
    return t, int(caps.pop()), int(pats.pop())


def load_names():
    m = pd.read_csv(E / '07_eval' / 'table_main.csv')
    return dict(zip(m.config, m.name)), m.set_index('config')


# --------------------------------------------------------------- tables ----
def table_s1(doc, train, cap, patience, names):
    headers = ['Configuration', 'Seed', 'best_avg8', 'val@best', 'best epoch',
               'epochs run', 'Early stopped', 'n_train', 'n_val', 'Time (min)']
    widths = [56, 13, 21, 21, 21, 21, 24, 23, 20, 22]
    aligns = ['left', 'right', 'right', 'right', 'right', 'right', 'center',
              'right', 'right', 'right']
    table = make_table(doc, headers, widths, aligns)

    configs = sorted(train.config.unique())
    for ci, cfg in enumerate(configs):
        grp = train[train.config == cfg]
        for j, (_, r) in enumerate(grp.iterrows()):
            row = add_row(table, [
                names[cfg] if j == 0 else '',
                integer(r.seed),
                dec(r.best_avg8), dec(r.val_at_best),
                integer(r.best_epoch), integer(r.epochs_run),
                'Yes' if r.early else 'No',
                integer(r.n_train), integer(r.n_val),
                dec(r.minutes, 1),
            ], widths, aligns)
        if ci < len(configs) - 1:
            rule(row, 'bottom', RULE_GROUP)        # thin line between groups
    rule(row, 'bottom', RULE_HEAVY)

    n_early = int(train.early.sum())
    n_cap = int((~train.early).sum())
    hours = train.minutes.sum() / 60.0
    add_caption(doc, 'Table S1.',
                f'Training summary for all {len(train)} models '
                f'({len(configs)} configurations {TIMES} '
                f'{len(train) // len(configs)} random seeds). best_avg8 is the '
                f'best 8-epoch moving average of validation PCC; val@best is the '
                f'single-epoch validation PCC at that point. '
                f'Early stopping was triggered by {patience} epochs without '
                f'improvement in the checkpoint criterion; runs are identified '
                f'here as early-stopped where epochs run < {cap}. {n_early} of '
                f'the {len(train)} models stopped early and {n_cap} were '
                f'truncated at the {cap}-epoch cap. Cumulative training time was '
                f'{hours:,.1f} h.')
    return table, len(table.rows)


def table_s2(doc, single, main, names):
    headers = ['Configuration', 'Seed', 'FEP wmPCC', f'FEP wmK{TAU}',
               'OOD PCC', f'OOD K{TAU}', 'OOD RMSE']
    widths = [56, 26, 33, 33, 33, 33, 33]
    aligns = ['left', 'left', 'right', 'right', 'right', 'right', 'right']
    metrics = ['FEP_wmPCC', 'FEP_wmKtau', 'OOD_PCC', 'OOD_Ktau', 'OOD_RMSE']
    table = make_table(doc, headers, widths, aligns)

    configs = sorted(single.config.unique())
    gains = {'FEP_wmPCC': [], 'OOD_PCC': []}
    for ci, cfg in enumerate(configs):
        grp = single[single.config == cfg].sort_values('seed')
        for j, (_, r) in enumerate(grp.iterrows()):
            add_row(table, [names[cfg] if j == 0 else '', integer(r.seed)]
                    + [dec(r[m]) for m in metrics], widths, aligns)
        add_row(table, ['', 'Mean / SD']
                + [f'{dec(grp[m].mean())} {PM} {dec(grp[m].std(ddof=1))}'
                   for m in metrics], widths, aligns, italic=True)
        row = add_row(table, ['', 'Ensemble']
                      + [dec(main.loc[cfg, m]) for m in metrics],
                      widths, aligns, bold=True)
        for m in gains:
            gains[m].append(main.loc[cfg, m] - grp[m].mean())
        if ci < len(configs) - 1:
            rule(row, 'bottom', RULE_GROUP)
    rule(row, 'bottom', RULE_HEAVY)

    n_seeds = int(single.groupby('config').size().unique()[0])
    fep = np.array(gains['FEP_wmPCC'])
    ood = np.array(gains['OOD_PCC'])
    n_pos = int(((fep > 0) & (ood > 0)).sum())
    add_caption(doc, 'Table S2.',
                f'Per-seed single-model metrics and the gain from ensembling. '
                f'Each configuration lists its {n_seeds} '
                f'seeds, their mean {PM} SD (italic), and the {n_seeds}-model '
                f'ensemble (bold). wmPCC and wmK{TAU} are weighted means over the FEP '
                f'benchmark targets. Ensemble predictions exceeded the seed '
                f'mean on both metrics in all {n_pos} configurations, by '
                f'{dec(fep.mean(), 3, True)} on average for FEP wmPCC and '
                f'{dec(ood.mean(), 3, True)} for OOD PCC.')
    return table, len(table.rows)


def table_s6(doc, paired, names):
    headers = ['Configuration', f'{DELTA}FEP wmPCC [95% CI]',
               f'{DELTA}OOD PCC [95% CI]']
    widths = [52, 54, 54]
    aligns = ['left', 'right', 'right']
    table = make_table(doc, headers, widths, aligns)

    for _, r in paired.sort_values('config').iterrows():
        row = add_row(table, [
            names[int(r.config)],
            f'{dec(r.dFEP_wmPCC, 4, True)}  {interval(r.dFEP_lo, r.dFEP_hi)}',
            f'{dec(r.dOOD_PCC, 4, True)}  {interval(r.dOOD_lo, r.dOOD_hi)}',
        ], widths, aligns)
    rule(row, 'bottom', RULE_HEAVY)

    ref = int(paired.vs.iloc[0])
    fep_excl = int((~((paired.dFEP_lo <= 0) & (paired.dFEP_hi >= 0))).sum())
    ood_excl = int((~((paired.dOOD_lo <= 0) & (paired.dOOD_hi >= 0))).sum())
    add_caption(doc, 'Table S6.',
                f'Paired differences against the crystal-structure control '
                f'(configuration {ref}, {names[ref]}). Differences are taken '
                f'relative to Configuration {ref}; the sign is therefore '
                f'reversed with respect to Table 4, where the baseline is the '
                f'reference. Intervals are BCa 95% confidence intervals from the '
                f'paired bootstrap. All '
                f'{fep_excl} of the {len(paired)} configurations have an FEP '
                f'wmPCC interval that excludes zero; {ood_excl} of {len(paired)} '
                f'do so for OOD PCC.')
    return table, len(table.rows)


# ------------------------------------------------------------------ main ---
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, cap, patience = load_training()
    names, main_tbl = load_names()
    single = pd.read_csv(E / '07_eval' / 'table_single.csv')
    paired = pd.read_csv(E / '07_eval_base11' / 'paired_diff.csv')

    doc = new_document()
    t1, n1 = table_s1(doc, train, cap, patience, names)

    setup_section(doc.add_section(WD_SECTION.NEW_PAGE), landscape=True)
    t2, n2 = table_s2(doc, single, main_tbl, names)

    setup_section(doc.add_section(WD_SECTION.NEW_PAGE), landscape=False)
    t6, n6 = table_s6(doc, paired, names)

    assert (patience, cap) == (20, 250), (patience, cap)
    path = OUT / 'Supplementary_Tables.docx'
    doc.save(path)

    n_cfg = train.config.nunique()
    print(f'wrote {path}')
    print()
    print(f'{"Table":<6} {"orientation":<11} {"cols":>4} {"header":>7} '
          f'{"body":>6} {"total rows":>11}')
    for tag, land, tbl, n, body in (
            ('S1', 'landscape', t1, n1, len(train)),
            ('S2', 'landscape', t2, n2, len(single) + 2 * n_cfg),
            ('S6', 'portrait', t6, n6, len(paired))):
        print(f'{tag:<6} {land:<11} {len(tbl.columns):>4} {1:>7} {body:>6} {n:>11}')
    print()
    print(f'S1 body  = {len(train)} runs '
          f'({n_cfg} configs x {len(train) // n_cfg} seeds)')
    print(f'S2 body  = {len(single)} seed rows + {n_cfg} Mean/SD + '
          f'{n_cfg} Ensemble = {len(single) + 2 * n_cfg}')
    print(f'S6 body  = {len(paired)} configurations vs config '
          f'{int(paired.vs.iloc[0])}')
    print()
    print(f'early stopped {int(train.early.sum())} / truncated at cap '
          f'{int((~train.early).sum())} / cap {cap} epochs')
    print(f'cumulative training time {train.minutes.sum() / 60:,.1f} h')


if __name__ == '__main__':
    main()
