#!/usr/bin/env python
"""
Stage 6: evaluation and statistical inference.

Inputs: pred_ood295.npy / pred_fep42.npy under each (config, seed) directory
        of 06_train (the training script has already run inference with the
        best checkpoint of each run and inverse-transformed outputs to pK).

Outputs in experiments/07_eval/
  results.json         structured results for all metrics and CIs
  table_main.csv       main table: per-config FEP wmPCC/wmKτ + OOD
                       PCC/τ_b/RMSE/MAE (with 95% BCa confidence intervals)
  table_single.csv     single-model (per-seed) metrics, to show the
                       ensembling gain
  per_series/cfgXX.csv per-series r / τ / n over the 42 series, per config
  paired_diff.csv      paired difference CIs of each config vs config 1
                       (significance is judged from the paired difference,
                       not from overlap of the individual intervals)
  scaling.csv          scaling-law log-linear fit (k, b, R²)
  strat.csv            sequence-similarity stratified analysis (exploratory)
  gator_best.json      config with the highest FEP wmPCC among configs 3-8
                       → which Gator variant config 10 uses
  report.md            human-readable summary

Notes (also recorded in report.md):
 · Stratification computes the maximum sequence identity against the
   source-level target pool: the 30%/60% Gator size subsets are random
   samples whose target coverage is almost identical to the 100% version,
   and the stratified analysis is exploratory by design, so it is not
   broken down per subset.
 · BindingNet / BindingDB-DCS have no separate sequence pool; their targets
   are PDB entries, so the PDBbind sequence pool is used as a proxy.
"""
import os, sys, json, argparse, subprocess, shutil
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_data import GraphStore, load_testset, config_meta
import lib_eval as LE

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
META = f'{E}/00_meta'
TRAIN = f'{E}/06_train'
OUT = f'{E}/07_eval'
TMP = f'{ROOT}/tmp_07'
log = lambda *a: print(*a, flush=True)

SRC_OF_CONFIG_POOL = {          # config component → pool for stratification
    'pdbbind': 'pdbbind', 'bindingnet': 'pdbbind', 'bindingdb_dcs': 'pdbbind',
    'hiqbind': 'hiqbind', 'gatoraffinity': 'gator'}


# ─────────────── prediction collection and ensembling ───────────────

def collect(cid, seeds):
    """Return {seed: {'ood295': arr, 'fep42': arr}}.

    Only seeds whose done.json exists are included.
    """
    out = {}
    for s in seeds:
        d = f'{TRAIN}/cfg{cid:02d}/seed{s}'
        if not os.path.exists(f'{d}/done.json'):
            continue
        try:
            out[s] = {t: np.load(f'{d}/pred_{t}.npy') for t in ['ood295', 'fep42']}
        except FileNotFoundError:
            log(f'  ! cfg{cid:02d} seed{s} missing predictions, skipping')
    return out


def ensemble(preds, key):
    """Simple arithmetic mean; NaNs are ignored per sample."""
    a = np.vstack([preds[s][key] for s in sorted(preds)])
    with np.errstate(invalid='ignore'):
        m = np.nanmean(a, axis=0)
    return m, a


# ─────────────── sequence-similarity stratification ───────────────

def write_fa(path, recs):
    with open(path, 'w') as f:
        for k, s in recs:
            if s:
                f.write(f'>{k}\n{s}\n')


AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O'}


def longest_chain_seq(pdb_path):
    """Same convention as 02b_extract_smiles_seq.py.

    Parse each chain from its CA atoms and return the longest chain.
    """
    try:
        chains, seen = {}, set()
        with open(pdb_path, 'r', errors='ignore') as f:
            for ln in f:
                if ln.startswith('ENDMDL'):
                    break
                if ln.startswith('ATOM') and ln[12:16].strip() == 'CA':
                    res = ln[17:20].strip().upper()
                    if res not in AA3:
                        continue
                    rid = (ln[21], ln[22:27])
                    if rid in seen:
                        continue
                    seen.add(rid)
                    chains.setdefault(ln[21], []).append(AA3[res])
        return ''.join(max(chains.values(), key=len)) if chains else ''
    except Exception:
        return ''


def ood_seqs():
    """Target sequences of OOD 295.

    Do not look these up in pdbbind_smiles_seq.csv: that table is generated
    from pdbbind_clean.csv, from which the 295 OOD entries have already been
    removed, so the lookup fails (silently yielding empty strings and
    dropping the entire OOD part of the stratification). Instead, parse the
    refined-set PDB structures directly and cache the result in
    00_meta/ood295_seq.fasta.
    """
    cache = f'{META}/ood295_seq.fasta'
    ood = pd.read_csv(f'{META}/testset_ood295.csv')
    if os.path.exists(cache):
        out, cur, buf = [], None, []
        for ln in open(cache):
            if ln.startswith('>'):
                if cur:
                    out.append((cur, ''.join(buf)))
                cur, buf = ln[1:].strip(), []
            else:
                buf.append(ln.strip())
        if cur:
            out.append((cur, ''.join(buf)))
        return out
    recs = []
    for p in ood.pdb_id.unique():
        s = longest_chain_seq(f'{ROOT}/datasets/structure/pdbbind/refined-set/'
                             f'{p}/{p}_protein.pdb')
        recs.append((f'ood|{str(p).lower()}', s))
    write_fa(cache, recs)
    n = sum(1 for _, s in recs if s)
    log(f'  OOD target sequences parsed {n}/{len(recs)} → {cache}')
    return recs


def eval_maxid(force=False):
    """Max sequence identity of each eval target vs each source target pool.

    Written to 00_meta/eval_maxid.csv.
    """
    cache = f'{META}/eval_maxid.csv'
    if os.path.exists(cache) and not force:
        return pd.read_csv(cache)
    if shutil.which('mmseqs') is None:
        log('  ! mmseqs not found, skipping stratification')
        return None
    os.makedirs(TMP, exist_ok=True)

    # query = OOD 295 target sequences + FEP 42 series target sequences
    q = list(ood_seqs())
    fep_fa = f'{META}/pool_fep42.fasta'
    if os.path.exists(fep_fa):
        cur, buf = None, []
        for ln in open(fep_fa):
            if ln.startswith('>'):
                if cur:
                    q.append((f'fep|{cur}', ''.join(buf)))
                # Headers look like ">FEP|ptp1b"; keep only the series name
                # after "|", otherwise the key becomes fep|FEP|ptp1b and no
                # longer matches fep|ptp1b on the evaluation side.
                cur, buf = ln[1:].strip().split()[0].split('|')[-1], []
            else:
                buf.append(ln.strip())
        if cur:
            q.append((f'fep|{cur}', ''.join(buf)))
    qf = f'{TMP}/query.fasta'
    write_fa(qf, q)
    log(f'  stratification query: {sum(1 for _, s in q if s)} sequences'
        f' (OOD {sum(1 for k, s in q if k.startswith("ood|") and s)} + '
        f'FEP {sum(1 for k, s in q if k.startswith("fep|") and s)})')

    rows = []
    for src, fa in [('pdbbind', f'{META}/pool_pdbbind.fasta'),
                    ('hiqbind', f'{META}/pool_hiqbind.fasta'),
                    ('gator', f'{E}/03_gator/pool_gator.fasta')]:
        if not os.path.exists(fa):
            log(f'  ! missing {fa}, skipping {src}')
            continue
        o = f'{TMP}/hits_{src}.tsv'
        cmd = ['mmseqs', 'easy-search', qf, fa, o, f'{TMP}/t_{src}',
               '--min-seq-id', '0.0', '-c', '0.5', '--cov-mode', '0', '-e', '1e-3',
               '-s', '7.5', '--max-seqs', '4000', '--threads', '32',
               '--format-output', 'query,target,fident']
        log(f'  [mmseqs] eval targets vs {src} …')
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log(f'  ! mmseqs failed: {r.stderr[-400:]}')
            continue
        h = pd.read_csv(o, sep='\t', names=['q', 't', 'fident'])
        g = h.groupby('q').fident.max().reset_index()
        g['src'] = src
        rows.append(g)
    if not rows:
        return None
    d = pd.concat(rows, ignore_index=True)
    d = d.pivot(index='q', columns='src', values='fident').reset_index()
    # No hit at all against a pool means the similarity is below the
    # e-value/coverage threshold, so fill with 0 instead of dropping. If
    # dropped, these queries would land in neither high nor low in the
    # downstream lay.get(i,'unknown'), wasting samples that are exactly the
    # ones the low-similarity stratum most needs to keep.
    d = pd.DataFrame({'q': [k for k, s in q if s]}).merge(d, on='q', how='left')
    d[[c for c in d.columns if c != 'q']] = d[
        [c for c in d.columns if c != 'q']].fillna(0.0)
    d.to_csv(cache, index=False)
    log(f'  stratification cache → {cache}')
    return d


def strat_layers(maxid, parts, keys):
    """Return {query: 'high'|'low'} for the given config components.

    Threshold: 30% maximum sequence identity.
    """
    pools = sorted({SRC_OF_CONFIG_POOL[p] for p in parts if p in SRC_OF_CONFIG_POOL})
    cols = [c for c in pools if c in maxid.columns]
    if not cols:
        return {}
    v = maxid[cols].max(axis=1).fillna(0.0).values
    return {q: ('high' if x >= 0.30 else 'low') for q, x in zip(maxid['q'], v)}


# ───────────────────────────── main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', type=int, default=list(range(1, 11)))
    ap.add_argument('--seeds', nargs='*', type=int, default=list(range(42, 47)))
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--baseline', type=int, default=1,
                    help='baseline config for paired differences '
                         '(config 1 = baseline)')
    ap.add_argument('--skip-strat', action='store_true')
    ap.add_argument('--force-strat', action='store_true',
                    help='recompute the 00_meta/eval_maxid.csv cache '
                         '(reruns mmseqs)')
    ap.add_argument('--single-model', action='store_true',
                    help='also output per-seed single-model metrics (slow: '
                         'running BCa for every seed would take very long, '
                         'so only point estimates are computed here)')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f'{OUT}/per_series', exist_ok=True)

    store = GraphStore(['ood295', 'fep42'])
    ood = load_testset('ood295', store)
    fep = load_testset('fep42', store)
    y_ood = ood.y.values.astype(float)
    y_fep = fep.y.values.astype(float)
    ser = fep.series.astype(str).values
    log(f'Eval sets: OOD {len(ood)} entries; FEP {len(fep)} ligands / '
        f'{len(set(ser))} series')

    cmeta = config_meta() if os.path.exists(f'{E}/05_configs/configs.json') else {}
    R, main_rows, single_rows = {}, [], []
    ens = {}   # cid → (ood_pred, fep_pred)

    for cid in a.configs:
        pr = collect(cid, a.seeds)
        if not pr:
            log(f'[config {cid:2d}] no completed seeds, skipping')
            continue
        po, ao = ensemble(pr, 'ood295')
        pf, af = ensemble(pr, 'fep42')
        ens[cid] = (po, pf)
        mo = np.isfinite(po) & np.isfinite(y_ood)
        mf = np.isfinite(pf) & np.isfinite(y_fep)
        name = cmeta.get(str(cid), {}).get('name', f'config{cid}')
        ntr = cmeta.get(str(cid), {}).get('n', np.nan)
        log(f'[config {cid:2d}] {name}: seeds {sorted(pr)}; '
            f'OOD valid {int(mo.sum())}/{len(po)}, '
            f'FEP valid {int(mf.sum())}/{len(pf)}')

        fm = LE.fep_metrics(ser[mf], y_fep[mf], pf[mf], n_boot=a.n_boot, seed=cid * 10)
        om = LE.ood_metrics(y_ood[mo], po[mo], n_boot=a.n_boot, seed=cid * 10 + 5)
        R[cid] = {'name': name, 'n_train': ntr, 'seeds': sorted(pr),
                  'n_ood_used': int(mo.sum()), 'n_fep_used': int(mf.sum()),
                  'fep': fm, 'ood': om}
        pd.DataFrame(fm['per_series']).to_csv(f'{OUT}/per_series/cfg{cid:02d}.csv', index=False)

        f = lambda c: f"{c['point']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]"
        main_rows.append({
            'config': cid, 'name': name, 'n_train': ntr, 'n_seeds': len(pr),
            'FEP_wmPCC': fm['wmPCC']['point'], 'FEP_wmPCC_lo': fm['wmPCC']['lo'],
            'FEP_wmPCC_hi': fm['wmPCC']['hi'],
            'FEP_wmKtau': fm['wmKtau']['point'], 'FEP_wmKtau_lo': fm['wmKtau']['lo'],
            'FEP_wmKtau_hi': fm['wmKtau']['hi'],
            'OOD_PCC': om['PCC']['point'], 'OOD_PCC_lo': om['PCC']['lo'],
            'OOD_PCC_hi': om['PCC']['hi'],
            'OOD_Ktau': om['Ktau']['point'], 'OOD_Ktau_lo': om['Ktau']['lo'],
            'OOD_Ktau_hi': om['Ktau']['hi'],
            'OOD_RMSE': om['RMSE']['point'], 'OOD_MAE': om['MAE']['point']})
        log(f'    FEP wmPCC {f(fm["wmPCC"])}  wmKτ {f(fm["wmKtau"])}')
        log(f'    OOD PCC   {f(om["PCC"])}  τ_b {f(om["Ktau"])}  '
            f'RMSE {om["RMSE"]["point"]:.3f}')

        if a.single_model:
            for i, s in enumerate(sorted(pr)):
                o1, f1 = ao[i], af[i]
                m1, m2 = np.isfinite(o1), np.isfinite(f1)
                u = LE.series_stats(ser[m2], y_fep[m2], f1[m2])
                single_rows.append({'config': cid, 'seed': s,
                                    'FEP_wmPCC': LE.weighted_mean(u, 1),
                                    'FEP_wmKtau': LE.weighted_mean(u, 2),
                                    'OOD_PCC': LE.pcc(y_ood[m1], o1[m1]),
                                    'OOD_Ktau': LE.ktau(y_ood[m1], o1[m1]),
                                    'OOD_RMSE': LE.rmse(y_ood[m1], o1[m1])})

    if not R:
        log('No config has any results, exiting.'); return 1
    pd.DataFrame(main_rows).to_csv(f'{OUT}/table_main.csv', index=False)
    if single_rows:
        pd.DataFrame(single_rows).to_csv(f'{OUT}/table_single.csv', index=False)

    # ── gator_best: highest FEP wmPCC among configs 3-8 (defines config 10) ──
    cand = {c: R[c]['fep']['wmPCC']['point'] for c in range(3, 9) if c in R}
    if cand:
        bc = max(cand, key=lambda c: cand[c])
        spec = {3: 'filtered_30', 4: 'filtered_60', 5: 'filtered_100',
                6: 'unfiltered_30', 7: 'unfiltered_60', 8: 'unfiltered_100'}[bc]
        json.dump({'best_config': bc, 'spec': spec, 'wmPCC': cand[bc],
                   'candidates': cand}, open(f'{META}/gator_best.json', 'w'), indent=1)
        log(f'\ngator_best = config {bc} ({spec}), '
            f'FEP wmPCC {cand[bc]:.4f}  → config 10 uses: '
            f'python 05_build_configs.py --config10 {spec}')

    # ── paired difference CIs (baseline = config a.baseline, default 1) ──
    pd_rows = []
    B = a.baseline
    if B in ens:
        po1, pf1 = ens[B]
        for cid in sorted(ens):
            if cid == B:
                continue
            po2, pf2 = ens[cid]
            m = np.isfinite(po1) & np.isfinite(po2) & np.isfinite(y_ood)
            ua = list(zip(y_ood[m], po2[m])); ub = list(zip(y_ood[m], po1[m]))
            d_ood = LE.paired_diff_ci(
                ua, ub, lambda U: LE.pcc([u[0] for u in U], [u[1] for u in U]),
                n_boot=a.n_boot, seed=cid * 100)
            m = np.isfinite(pf1) & np.isfinite(pf2) & np.isfinite(y_fep)
            sa = LE.series_stats(ser[m], y_fep[m], pf2[m])
            sb = LE.series_stats(ser[m], y_fep[m], pf1[m])
            d_fep = LE.paired_diff_ci(sa, sb, lambda U: LE.weighted_mean(U, 1),
                                      n_boot=a.n_boot, seed=cid * 100 + 1)
            sig = lambda c: (np.isfinite(c['lo']) and np.isfinite(c['hi'])
                             and (c['lo'] > 0 or c['hi'] < 0))
            pd_rows.append({'config': cid, 'name': R[cid]['name'], 'vs': B,
                            'dFEP_wmPCC': d_fep['point'], 'dFEP_lo': d_fep['lo'],
                            'dFEP_hi': d_fep['hi'], 'dFEP_sig': bool(sig(d_fep)),
                            'dOOD_PCC': d_ood['point'], 'dOOD_lo': d_ood['lo'],
                            'dOOD_hi': d_ood['hi'], 'dOOD_sig': bool(sig(d_ood))})
            log(f'  Δ config{cid} − config{B}: '
                f'FEP wmPCC {d_fep["point"]:+.4f} '
                f'[{d_fep["lo"]:+.4f}, {d_fep["hi"]:+.4f}] '
                f'{"significant" if sig(d_fep) else "not significant"}; '
                f'OOD PCC {d_ood["point"]:+.4f} '
                f'[{d_ood["lo"]:+.4f}, {d_ood["hi"]:+.4f}] '
                f'{"significant" if sig(d_ood) else "not significant"}')
        pd.DataFrame(pd_rows).to_csv(f'{OUT}/paired_diff.csv', index=False)
    else:
        log(f'  ! baseline config {B} has no results, '
            f'skipping paired differences')

    # ── Scaling Law ──
    sc_rows = []
    for tag, cs in [('filtered', [3, 4, 5]), ('unfiltered', [6, 7, 8])]:
        cs = [c for c in cs if c in R]
        if len(cs) < 3:
            continue
        sizes = [R[c]['n_train'] for c in cs]
        for metric, get in [('FEP_wmPCC', lambda c: R[c]['fep']['wmPCC']['point']),
                            ('OOD_PCC', lambda c: R[c]['ood']['PCC']['point'])]:
            fit = LE.scaling_fit(sizes, [get(c) for c in cs])
            sc_rows.append({'version': tag, 'metric': metric, 'configs': str(cs),
                            'sizes': str(sizes), **fit})
            log(f'  Scaling[{tag}/{metric}] k={fit["k"]:.4f} b={fit["b"]:.4f} '
                f'R²={fit["r2"]:.4f}')
    if sc_rows:
        pd.DataFrame(sc_rows).to_csv(f'{OUT}/scaling.csv', index=False)

    # ── sequence-similarity stratification (exploratory) ──
    st_rows = []
    if not a.skip_strat:
        mi = eval_maxid(force=a.force_strat)
        if mi is not None:
            for cid in sorted(ens):
                parts = cmeta.get(str(cid), {}).get('parts', [])
                base = [p.split('_')[0] if p.startswith('gator') else p for p in parts]
                base = ['gatoraffinity' if p == 'gator' else p for p in base]
                lay = strat_layers(mi, base, None)
                po, pf = ens[cid]
                for lab, ids, y, p, kind in [
                        ('OOD', ['ood|' + str(x).lower() for x in ood.pdb_id],
                         y_ood, po, 'ood'),
                        ('FEP', ['fep|' + str(x) for x in ser], y_fep, pf, 'fep')]:
                    tags = np.array([lay.get(i, 'unknown') for i in ids])
                    for L in ['high', 'low']:
                        m = (tags == L) & np.isfinite(p) & np.isfinite(y)
                        if m.sum() < 5:
                            continue
                        if kind == 'fep':
                            u = LE.series_stats(np.array(ids)[m], y[m], p[m])
                            v1, v2 = LE.weighted_mean(u, 1), LE.weighted_mean(u, 2)
                        else:
                            v1, v2 = LE.pcc(y[m], p[m]), LE.ktau(y[m], p[m])
                        st_rows.append({'config': cid, 'set': lab, 'layer': L,
                                        'n': int(m.sum()), 'pcc': v1, 'ktau': v2})
            if st_rows:
                pd.DataFrame(st_rows).to_csv(f'{OUT}/strat.csv', index=False)
                log(f'  stratification results → {OUT}/strat.csv')

    json.dump({'configs': R, 'paired_diff': pd_rows, 'scaling': sc_rows,
               'strat': st_rows, 'n_boot': a.n_boot,
               'n_ood': int(len(ood)), 'n_fep': int(len(fep)),
               'n_fep_series': int(len(set(ser)))},
              open(f'{OUT}/results.json', 'w'), indent=1, default=float)

    # ── report.md ──
    with open(f'{OUT}/report.md', 'w') as fo:
        fo.write('# Stage 6 evaluation results\n\n')
        fo.write(f'- BCa bootstrap n={a.n_boot} (FEP resampled at series '
                 f'level, OOD resampled at sample level)\n')
        fo.write(f'- FEP: {len(set(ser))} series / {len(fep)} ligands; '
                 f'OOD: {len(ood)} entries\n')
        fo.write('- Reference values (Valsson et al. 2024): '
                 'FEP wmPCC 0.59, wmKτ 0.42\n\n')
        fo.write('## Main table\n\n')
        fo.write(pd.DataFrame(main_rows).to_markdown(index=False, floatfmt='.4f'))
        if pd_rows:
            fo.write(f'\n\n## Paired differences vs config {B} '
                     f'(significant if the 95% BCa confidence interval '
                     f'excludes 0)\n\n')
            fo.write(pd.DataFrame(pd_rows).to_markdown(index=False, floatfmt='.4f'))
        if sc_rows:
            fo.write('\n\n## Scaling law (value = k·log10(N) + b)\n\n')
            fo.write(pd.DataFrame(sc_rows).to_markdown(index=False, floatfmt='.4f'))
        if st_rows:
            fo.write('\n\n## Sequence-similarity stratification '
                     '(exploratory, threshold 30%)\n\n')
            fo.write(pd.DataFrame(st_rows).to_markdown(index=False, floatfmt='.4f'))
            sd = pd.DataFrame(st_rows)
            for s in sd['set'].unique():
                miss = {'high', 'low'} - set(sd.loc[sd['set'] == s, 'layer'])
                if miss:
                    fo.write(f'\n\n> {s} is missing the '
                             f'"{"/".join(sorted(miss))}" stratum: '
                             f'fewer than 5 valid samples in that stratum, '
                             f'i.e. the maximum sequence identity of every '
                             f'{s} target to the training source target '
                             f'pools falls on the same side of the '
                             f'threshold, so this evaluation set cannot be '
                             f'stratified by similarity.\n')
        fo.write('\n\n### Methodological notes\n'
                 '- Stratification uses source-level target pools (the Gator '
                 'size subsets are random samples whose target coverage is '
                 'essentially identical to the 100% version).\n'
                 '- BindingNet / BindingDB-DCS have no separate sequence '
                 'pool; their targets are PDB entries, so the PDBbind '
                 'sequence pool is used as a proxy.\n')
    log(f'\nOutputs → {OUT}/{{results.json, table_main.csv, paired_diff.csv, '
        f'scaling.csv, strat.csv, per_series/, report.md}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
