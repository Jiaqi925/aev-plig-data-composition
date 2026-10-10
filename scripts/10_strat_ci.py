#!/usr/bin/env python
"""
Supplementary task 2: rerun the existing BCa and paired difference code on the
low / high layers of the OOD similarity stratification.

Motivation: the existing strat.csv only has point estimates, so the report
could only say "should not be over-interpreted". This script gives each layer
a 95% BCa confidence interval and runs a within-layer paired difference test of
each config vs config 1, so that the "low-layer ranking" observation has a
statistical basis.

Fully reuses bca_ci / paired_diff_ci / pcc / ktau from lib_eval; the stratified
layer definition matches strat_layers in 07_evaluate.py (max_seqid < 0.30 is
low).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
sys.path.insert(0, f'{ROOT}/scripts')
import lib_eval as LE  # noqa: E402

TRAIN = f'{E}/06_train'
EVAL = f'{E}/07_eval'
OUT = f'{E}/10_strat_ci'
IDX = f'{ROOT}/datasets/index'
SEEDS = [42, 43, 44, 45, 46]
N_BOOT = 10000
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)


def load_ens(cid):
    """Load the 5-seed ensemble OOD predictions of one config (same convention
    as the ensemble in 07_evaluate: per-sample mean)"""
    ps = []
    for s in SEEDS:
        f = f'{TRAIN}/cfg{cid:02d}/seed{s}/pred_ood295.npy'
        if not os.path.exists(f):
            return None
        ps.append(np.load(f).astype(float))
    return np.mean(np.vstack(ps), axis=0)


def main():
    # ---------- Ground truth ----------
    ood = pd.read_csv(f'{IDX}/_testsets/index_oodtest.csv')
    ood = ood[ood.split == 'test'].reset_index(drop=True)
    y = ood['-logKd/Ki'].values.astype(float)

    # ---------- Layer labels: reuse eval_maxid.csv from 07_evaluate; for each
    # config take the max similarity over the sources it contains ----------
    maxid = pd.read_csv(f'{E}/00_meta/eval_maxid.csv')
    cmeta = json.load(open(f'{E}/05_configs/configs.json'))
    SRC_OF_POOL = {'pdbbind': 'pdbbind', 'bindingnet': 'pdbbind',
                   'bindingdb_dcs': 'pdbbind', 'hiqbind': 'hiqbind',
                   'gatoraffinity': 'gator'}
    ood_keys = ['ood|' + str(x).strip().lower()[:4] for x in ood.PDB_code]

    def layers_for(cid):
        parts = cmeta.get(str(cid), {}).get('parts', [])
        base = [p.split('_')[0] if p.startswith('gator') else p for p in parts]
        base = ['gatoraffinity' if p == 'gator' else p for p in base]
        pools = sorted({SRC_OF_POOL[p] for p in base if p in SRC_OF_POOL})
        cols = [c for c in pools if c in maxid.columns]
        v = maxid[cols].max(axis=1).fillna(0.0).values
        lay = {q: ('high' if x >= 0.30 else 'low') for q, x in zip(maxid['q'], v)}
        return np.array([lay.get(k, 'unknown') for k in ood_keys])

    log('Stratification basis: experiments/00_meta/eval_maxid.csv (mmseqs, '
        'evaluation targets vs each source target pool), threshold seqid '
        '0.30, per config the max over the sources it contains')

    # ---------- Stratified BCa per config ----------
    log(f'\n[1/2] stratified BCa per config (n_boot={N_BOOT}) …')
    rows = []
    ens = {}
    lays = {}
    for cid in range(1, 12):
        p = load_ens(cid)
        if p is None:
            log(f'  config {cid}: predictions missing, skipped')
            continue
        ens[cid] = p
        layer = layers_for(cid)
        lays[cid] = layer
        for L in ['high', 'low']:
            m = (layer == L) & np.isfinite(p) & np.isfinite(y)
            n = int(m.sum())
            if n < 5:
                continue
            units = list(zip(y[m], p[m]))
            r_pcc = LE.bca_ci(units,
                              lambda U: LE.pcc([u[0] for u in U], [u[1] for u in U]),
                              n_boot=N_BOOT, seed=cid * 10)
            r_ktau = LE.bca_ci(units,
                               lambda U: LE.ktau([u[0] for u in U], [u[1] for u in U]),
                               n_boot=N_BOOT, seed=cid * 10 + 1)
            r_rmse = LE.bca_ci(units,
                               lambda U: LE.rmse([u[0] for u in U], [u[1] for u in U]),
                               n_boot=N_BOOT, seed=cid * 10 + 2)
            rows.append({
                'config': cid, 'layer': L, 'n': n,
                'pcc': r_pcc['point'], 'pcc_lo': r_pcc['lo'], 'pcc_hi': r_pcc['hi'],
                'pcc_width': r_pcc['hi'] - r_pcc['lo'],
                'ktau': r_ktau['point'], 'ktau_lo': r_ktau['lo'], 'ktau_hi': r_ktau['hi'],
                'rmse': r_rmse['point'], 'rmse_lo': r_rmse['lo'], 'rmse_hi': r_rmse['hi'],
                'method': r_pcc['method']})
            log(f'  cfg{cid:02d} {L:<4s} n={n:>3d}  PCC {r_pcc["point"]:.4f} '
                f'[{r_pcc["lo"]:.4f}, {r_pcc["hi"]:.4f}] '
                f'width {r_pcc["hi"]-r_pcc["lo"]:.4f}')

    pd.DataFrame(rows).to_csv(f'{OUT}/strat_ci.csv', index=False)

    # ---------- Within-layer paired difference vs config 1 ----------
    log('\n[2/2] within-layer paired difference (each config − config 1) …')
    pd_rows = []
    if 1 in ens:
        p1 = ens[1]
        for cid in sorted(ens):
            if cid == 1:
                continue
            p2 = ens[cid]
            # Pairing must be on the same set of units, so the layers of the
            # reference config (1) are used, avoiding different layer members
            # between the two configs
            layer = lays[1]
            for L in ['high', 'low']:
                m = (layer == L) & np.isfinite(p1) & np.isfinite(p2) & np.isfinite(y)
                n = int(m.sum())
                if n < 5:
                    continue
                ua = list(zip(y[m], p2[m]))
                ub = list(zip(y[m], p1[m]))
                d = LE.paired_diff_ci(
                    ua, ub,
                    lambda U: LE.pcc([u[0] for u in U], [u[1] for u in U]),
                    n_boot=N_BOOT, seed=cid * 100 + (0 if L == 'high' else 5))
                sig = (np.isfinite(d['lo']) and np.isfinite(d['hi'])
                       and (d['lo'] > 0 or d['hi'] < 0))
                pd_rows.append({'config': cid, 'vs': 1, 'layer': L, 'n': n,
                                'dPCC': d['point'], 'dPCC_lo': d['lo'],
                                'dPCC_hi': d['hi'], 'sig': bool(sig),
                                'method': d['method']})
                log(f'  cfg{cid:02d} {L:<4s} ΔPCC {d["point"]:+.4f} '
                    f'[{d["lo"]:+.4f}, {d["hi"]:+.4f}] '
                    f'{"significant" if sig else "not significant"}')
        pd.DataFrame(pd_rows).to_csv(f'{OUT}/strat_paired_diff.csv', index=False)

    # ---------- CI of the high − low drop (within one config the two layers
    #            are independent and a normal approximation of the difference
    #            is inappropriate, so instead report each CI separately and
    #            state whether the intervals overlap) ----------
    log('\n[extra] do the high and low CIs overlap (overlap ⇒ the drop cannot '
        'be asserted statistically):')
    ov_rows = []
    df = pd.DataFrame(rows)
    for cid in sorted(ens):
        h = df[(df.config == cid) & (df.layer == 'high')]
        l = df[(df.config == cid) & (df.layer == 'low')]
        if len(h) == 0 or len(l) == 0:
            continue
        h, l = h.iloc[0], l.iloc[0]
        overlap = not (l.pcc_hi < h.pcc_lo or h.pcc_hi < l.pcc_lo)
        ov_rows.append({'config': cid, 'high_pcc': h.pcc, 'high_lo': h.pcc_lo,
                        'high_hi': h.pcc_hi, 'low_pcc': l.pcc, 'low_lo': l.pcc_lo,
                        'low_hi': l.pcc_hi, 'delta': h.pcc - l.pcc,
                        'ci_overlap': bool(overlap)})
        log(f'  cfg{cid:02d} high [{h.pcc_lo:.3f},{h.pcc_hi:.3f}] vs '
            f'low [{l.pcc_lo:.3f},{l.pcc_hi:.3f}] '
            f'drop {h.pcc-l.pcc:.4f} {"overlap" if overlap else "no overlap"}')
    pd.DataFrame(ov_rows).to_csv(f'{OUT}/strat_overlap.csv', index=False)

    json.dump({'n_boot': N_BOOT, 'seeds': SEEDS,
               'layer_rule': ('eval_maxid.csv (mmseqs, evaluation targets vs '
                              'the target pool of each source), seqid >= 0.30 '
                              '-> high, else low; per config, the maximum over '
                              'the sources it contains'),
               'paired_diff_layer_basis': ('config 1 (keeps the pairing unit '
                                           'consistent)'),
               'n_by_config': {str(c): {'high': int((lays[c] == 'high').sum()),
                                        'low': int((lays[c] == 'low').sum())}
                               for c in sorted(lays)},
               'strat_ci': rows, 'paired_diff': pd_rows, 'overlap': ov_rows},
              open(f'{OUT}/strat_ci.json', 'w'), indent=1, default=float)
    log(f'\nOutputs → {OUT}/{{strat_ci.csv, strat_paired_diff.csv, '
        f'strat_overlap.csv, strat_ci.json}}')


if __name__ == '__main__':
    main()
