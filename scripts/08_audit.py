#!/usr/bin/env python
"""
Stage 4.5: comprehensive compliance audit of data processing — recomputed
independently, without trusting the numbers self-reported by each stage.

Checks, item by item:
  A Raw dataset sizes vs expected values
  B Leakage control: first 4 characters of every source PDB ID
    ∩ (OOD 295 ∪ FEP) must be empty
  C HiQBind deduplication: count closure + spot check of the
    three-criterion log
  D Gator evaluation-set separation: removed/kept rows are consistent with
    the criteria
  E Gator size sampling: 30% ⊂ 60% ⊂ 100%, correct proportions,
    reproducible with seed 42
  F Version relationship: filtered = unfiltered ∩ (confidence ≥ 0.7)
  G pK convention: per-source distributions and conversion rules
  H Evaluation sets: 295 OOD entries, 42 FEP series with n ≥ 10 per series
  I FEP pK conversion re-check (ΔG = −RT·ln10·pK,
    RT·ln10 = 1.358985 kcal/mol @297K)
  J Graph features: 367-dim nodes, 4-dim edges, counts aligned with the
    index tables
  K Config assembly: each config's count = sum of its components, with no
    leaked IDs
  L Config-level leakage red line: every config_XX.csv is re-checked
    independently — neither the PDB ID nor the unique_id of any training
    sample may overlap OOD 295 / FEP 42 (a hard constraint explicitly
    required by the user)

Exit code 0 = all passed (including SKIP), 1 = at least one FAIL.
"""
import os, sys, json, math
import numpy as np
import pandas as pd

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
META = f'{E}/00_meta'
G = f'{E}/04_graphs'
RT_LN10 = 1.987204259e-3 * 297.0 * math.log(10.0)

RES = []
def chk(tag, ok, msg=''):
    s = 'SKIP' if ok is None else ('PASS' if bool(ok) else 'FAIL')
    RES.append((s, tag, msg))
    print(f'[{s}] {tag}' + (f' — {msg}' if msg else ''), flush=True)

def head(t):
    print(f'\n══ {t} ' + '═' * max(0, 60 - len(t)), flush=True)


def main():
    # ─────────────────── A raw dataset sizes ───────────────────
    head('A raw dataset sizes vs expected')
    r1 = json.load(open(f'{E}/01_baseline_index/report.json'))
    r2a = json.load(open(f'{E}/02_hiqbind/report_02a.json'))
    r3a = json.load(open(f'{E}/03_gator/report_03a.json'))
    plan_raw = {'PDBbind': 19443, 'BindingNet': 69816, 'BindingDB-DCS': 8822,
                'HiQBind': 32275, 'GatorAffinity(structures)': 65031}
    got = {'PDBbind': r1['pdbbind']['raw'], 'BindingNet': r1['bindingnet']['raw'],
           'BindingDB-DCS': r1['bindingdb_dcs']['raw'], 'HiQBind': r2a['raw'],
           'GatorAffinity(structures)': r3a['structure_dirs']}
    for k, v in plan_raw.items():
        d = got[k] - v
        chk(f'A.{k}', d == 0 or abs(d) <= 30,
            f'expected {v} / actual {got[k]} (diff {d:+d})'
            + ('; expected counts are approximate, the difference is '
               'recorded in this report' if d else ''))

    # ─────────────────── B leakage control ───────────────────
    head('B leakage control')
    leak = json.load(open(f'{META}/leakage_ids.json'))
    ood_ids, fep_ids = set(leak['ood_pdb_ids']), set(leak['fep_pdb_ids'])
    excl = set(leak['excluded_pdb_ids'])
    chk('B.0 excluded set = OOD ∪ FEP', excl == (ood_ids | fep_ids),
        f'OOD {len(ood_ids)} + FEP {len(fep_ids)} → '
        f'union {len(ood_ids | fep_ids)}, '
        f'excluded set {len(excl)}')
    srcs = {'pdbbind': f'{E}/01_baseline_index/pdbbind_clean.csv',
            'bindingnet': f'{E}/01_baseline_index/bindingnet_clean.csv',
            'bindingdb_dcs': f'{E}/01_baseline_index/bindingdb_dcs_clean.csv',
            'hiqbind': f'{E}/02_hiqbind/hiqbind_dedup.csv',
            'gatoraffinity': f'{E}/03_gator/gator_dedup.csv'}
    tab = {}
    for s, p in srcs.items():
        d = pd.read_csv(p, low_memory=False)
        tab[s] = d
        if 'pdb_id' in d.columns:
            pid = d.pdb_id.fillna('').astype(str).str.lower().str[:4]
            hit = sorted(set(pid[pid.isin(excl) & (pid != '')]))
        elif 'PDB ID(s) for Ligand-Target Complex' in d.columns:
            # Gator has no real PDB IDs, but BindingDB metadata occasionally
            # has this field (multiple IDs, comma-separated). Step 1 of 03b
            # already used it as a safety-net filter; re-checked
            # independently here.
            col = d['PDB ID(s) for Ligand-Target Complex'].fillna('').astype(str)
            hit = sorted({t.strip().lower()[:4] for v in col for t in v.split(',')
                          if t.strip()} & excl)
        else:
            chk(f'B.{s}', None, 'no PDB ID column')
            continue
        chk(f'B.{s} PDB IDs free of leakage', len(hit) == 0,
            f'{len(d)} rows, hits {len(hit)}' + (f' {hit[:8]}' if hit else ''))

    # ─────────────────── C HiQBind deduplication ───────────────────
    head('C HiQBind deduplication')
    r2c = json.load(open(f'{E}/02_hiqbind/report_02c.json'))
    chk('C.1 count closure', r2c['hiqbind_in'] - r2c['duplicates_removed'] == r2c['net_new']
        == len(tab['hiqbind']),
        f"{r2c['hiqbind_in']} − {r2c['duplicates_removed']} = "
        f"{r2c['net_new']}, "
        f"file {len(tab['hiqbind'])} rows")
    chk('C.2 thresholds match expected values',
        r2c['thresholds'] == {'tanimoto': 0.9, 'abs_dpK': 1.0, 'tm_score': 0.8,
                             'tm_norm': 'max(chain1,chain2)'},
        str(r2c['thresholds']))
    dp = pd.read_csv(f'{E}/02_hiqbind/hiqbind_dup_pairs.csv')
    bad = dp[(dp.tanimoto <= 0.9) | (dp.abs_dpK > 1.0) | (dp.tm_score <= 0.8)]
    chk('C.3 pairs flagged as duplicates meet all three criteria',
        len(bad) == 0,
        f'{len(dp)} pairs, violations {len(bad)}')
    rm = set(dp.hq_unique_id)
    if rm:
        chk('C.4 duplicate samples removed from output',
            len(rm & set(tab['hiqbind'].unique_id)) == 0,
            f'duplicates {len(rm)} uids, remaining '
            f'{len(rm & set(tab["hiqbind"].unique_id))}')

    # ─────────────────── D Gator evaluation-set separation ───────────────────
    head('D Gator evaluation-set separation and cross-source deduplication')
    r3b = json.load(open(f'{E}/03_gator/report_03b.json'))
    g = tab['gatoraffinity']
    fl = pd.read_csv(f'{E}/03_gator/gator_testset_flags.csv')
    chk('D.1 count closure',
        r3b['gator_in'] - r3b['step1_pdbid_leak'] - r3b['step2']['removed_AND']
        - r3b['step3']['duplicates_removed'] == r3b['gator_final_unfiltered'] == len(g),
        f"{r3b['gator_in']} − {r3b['step1_pdbid_leak']} − {r3b['step2']['removed_AND']}"
        f" − {r3b['step3']['duplicates_removed']} = "
        f"{r3b['gator_final_unfiltered']}, "
        f'file {len(g)} rows')
    if {'seq_hit', 'lig_hit'} <= set(fl.columns):
        both = fl[(fl.seq_hit) & (fl.lig_hit)]
        chk('D.2 rows hitting both sequence and ligand removed',
            len(set(both.unique_id) & set(g.unique_id)) == 0,
            f'AND hits {len(both)} rows, remaining '
            f'{len(set(both.unique_id) & set(g.unique_id))}')
        chk('D.3 counts for readings A/B recorded faithfully',
            r3b['step2']['would_remove_OR'] > r3b['step2']['removed_AND'],
            f"AND removed {r3b['step2']['removed_AND']}, OR would remove "
            f"{r3b['step2']['would_remove_OR']} (diff "
            f"{r3b['step2']['would_remove_OR'] - r3b['step2']['removed_AND']})")
    gp = pd.read_csv(f'{E}/03_gator/gator_dup_pairs.csv')
    if len(gp):
        bad = gp[(gp.tanimoto <= 0.9) | (gp.abs_dpK > 1.0) | (gp.tm_score <= 0.8)]
        chk('D.4 cross-source duplicate pairs meet all three criteria',
            len(bad) == 0,
            f'{len(gp)} pairs, violations {len(bad)}')

    # ─────────────────── E/F Gator sampling and versions ───────────────────
    head('E size sampling + F version relationship')
    for ver in ['unfiltered', 'filtered']:
        c = {p: set(g.loc[g[f'in_{ver}_{p}'], 'unique_id']) for p in [30, 60, 100]}
        chk(f'E.{ver} nested 30 ⊂ 60 ⊂ 100',
            c[30] <= c[60] <= c[100],
            f'{len(c[30])} ⊂ {len(c[60])} ⊂ {len(c[100])}')
        base = len(c[100])
        ok = all(abs(len(c[p]) - round(base * p / 100)) <= 1 for p in [30, 60])
        chk(f'E.{ver} proportions correct', ok,
            f'30%: {len(c[30])} vs {round(base * .3)}; 60%: {len(c[60])} vs '
            f'{round(base * .6)}')
        # Reproduce the sampling with seed 42
        idx = g.index.values if ver == 'unfiltered' else g.index[g.pass_conf].values
        order = np.random.RandomState(42).permutation(idx)
        rep = set(g.loc[order[:round(len(order) * 0.3)], 'unique_id'])
        chk(f'E.{ver} reproducible with seed 42', rep == c[30],
            f'recomputed {len(rep)} rows, diff vs output {len(rep ^ c[30])}')
    chk('F filtered = unfiltered ∩ conf≥0.7',
        set(g.loc[g[f'in_filtered_100'], 'unique_id']) ==
        set(g.loc[g.pass_conf, 'unique_id']),
        f"unfiltered {int(g['in_unfiltered_100'].sum())}, "
        f"filtered {int(g['in_filtered_100'].sum())}, "
        f"conf passed {int(g.pass_conf.sum())}")
    if 'conf' in g.columns:
        v = g.loc[g.pass_conf, 'conf']
        chk('F.2 filtered confidence all ≥ 0.7', float(v.min()) >= 0.7 - 1e-9,
            f'min={float(v.min()):.4f}')

    # ─────────────────── G pK convention ───────────────────
    head('G pK convention (conversion rules)')
    # No pK truncation range was specified, so nothing is filtered; only the
    # distribution and the number of extreme values are reported. Records
    # with pK < 0 come from experimental measurements of very weak binding in
    # the original databases (e.g. Kd on the order of M) and are kept as-is.
    for s, d in tab.items():
        v = pd.to_numeric(d.pK, errors='coerce')
        neg = int((v < 0).sum()); hi = int((v > 14).sum())
        chk(f'G.{s} pK all parseable', bool(v.notna().all()),
            f'n={len(v)} min={v.min():.2f} max={v.max():.2f} '
            f'mean={v.mean():.3f}; '
            f'pK<0: {neg} rows, pK>14: {hi} rows '
            f'(no truncation required, kept as-is)')

    # ─────────────────── H evaluation sets ───────────────────
    head('H evaluation set sizes')
    ood = pd.read_csv(f'{META}/testset_ood295.csv')
    chk('H.1 OOD 295 entries with unique PDB IDs',
        len(ood) == 295 and ood.pdb_id.nunique() == 295,
        f'{len(ood)} entries / {ood.pdb_id.nunique()} unique IDs')
    fep = pd.read_csv(f'{META}/testset_fep42.csv')
    cnt = fep.series.value_counts()
    chk('H.2 FEP 42 series with n ≥ 10 per series',
        fep.series.nunique() == 42 and cnt.min() >= 10,
        f'{fep.series.nunique()} series / {len(fep)} ligands, '
        f'smallest series {cnt.min()}, '
        f'largest {cnt.max()}')
    chk('H.3 FEP ligand count 1087 (expected 1089; the difference of 2 is '
        'intermediate nodes without experimental ΔG)',
        len(fep) == 1087, f'{len(fep)} ligands')

    # ─────────────────── I FEP pK conversion ───────────────────
    head('I FEP pK conversion re-check')
    if 'dG_exp' in fep.columns:
        back = -fep.dG_exp.astype(float) / RT_LN10
        err = float(np.abs(back - fep.pK.astype(float)).max())
        chk('I.1 pK = −ΔG / (RT·ln10)', err < 1e-6,
            f'RT·ln10 = {RT_LN10:.6f} kcal/mol, max deviation {err:.2e}')
    else:
        chk('I.1', None, 'testset_fep42.csv has no dG_exp column, skipping')

    # ─────────────────── J graph features ───────────────────
    head('J graph feature dimensions and alignment')
    for s in ['ood295', 'fep42', 'pdbbind', 'bindingnet', 'bindingdb_dcs',
              'hiqbind', 'gatoraffinity']:
        d = f'{G}/{s}'
        if not os.path.exists(f'{d}/ids.npy'):
            chk(f'J.{s}', None, 'graphs not generated')
            continue
        x = np.load(f'{d}/x.npy', mmap_mode='r')
        ea = np.load(f'{d}/ea.npy', mmap_mode='r')
        ei = np.load(f'{d}/ei.npy', mmap_mode='r')
        ids = np.load(f'{d}/ids.npy')
        np_ = np.load(f'{d}/node_ptr.npy'); ep = np.load(f'{d}/edge_ptr.npy')
        ok = (x.shape[1] == 367 and ea.shape[1] == 4 and ei.shape[0] == 2
              and len(np_) == len(ids) + 1 and len(ep) == len(ids) + 1
              and int(np_[-1]) == x.shape[0] and int(ep[-1]) == ea.shape[0]
              and len(set(ids.tolist())) == len(ids))
        chk(f'J.{s} dims/pointers/uniqueness', ok,
            f'{len(ids)} graphs, x{tuple(x.shape)} '
            f'ea{tuple(ea.shape)} ei{tuple(ei.shape)}')

    # ─────────────────── K config assembly ───────────────────
    head('K config assembly')
    cf = f'{E}/05_configs/configs.json'
    if not os.path.exists(cf):
        chk('K', None, 'configs not assembled')
    else:
        cm = json.load(open(cf))
        for cid, m in sorted(cm.items(), key=lambda kv: int(kv[0])):
            d = pd.read_csv(f'{E}/05_configs/config_{int(cid):02d}.csv')
            chk(f'K.{cid} {m["name"]}',
                len(d) == m['n'] and d.unique_id.nunique() == len(d),
                f'{len(d)} rows, source breakdown {m["by_source"]}')

    # ─── L config-level leakage red line (hard constraint set by user) ───
    head('L config-level leakage red line: training data of any config '
         'must not overlap OOD/FEP')
    if not os.path.exists(cf):
        chk('L', None, 'configs not assembled')
    else:
        # unique_id → set of PDB IDs associated with that sample.
        # The four baseline sources have a pdb_id column; Gator has no real
        # PDB IDs, but the "PDB ID(s) for Ligand-Target Complex" field in the
        # BindingDB metadata is occasionally populated (4149 of 52773 rows),
        # so it is used for an exact complex-level re-check — this is the
        # config-level counterpart of B.gatoraffinity.
        u2p = {}
        for s, d in tab.items():
            if 'pdb_id' in d.columns:
                for u, p in zip(d.unique_id.astype(str),
                                d.pdb_id.fillna('').astype(str)):
                    u2p[u] = {p.lower()[:4]} if p else set()
            elif 'PDB ID(s) for Ligand-Target Complex' in d.columns:
                col = d['PDB ID(s) for Ligand-Target Complex'].fillna('').astype(str)
                for u, v in zip(d.unique_id.astype(str), col):
                    u2p[u] = {t.strip().lower()[:4]
                              for t in v.replace(';', ',').split(',') if t.strip()}
        ood_uid = set(pd.read_csv(f'{META}/testset_ood295.csv').pdb_id.astype(str))
        ft = pd.read_csv(f'{META}/testset_fep42.csv')
        fep_uid = {f'{a}__{b}' for a, b in zip(ft.series.astype(str), ft.lig_idx)}
        cm2 = json.load(open(cf))
        for cid, m in sorted(cm2.items(), key=lambda kv: int(kv[0])):
            d = pd.read_csv(f'{E}/05_configs/config_{int(cid):02d}.csv')
            u = set(d.unique_id.astype(str))
            hp = sorted({x for uu in u for x in u2p.get(uu, ()) if x in excl})
            hu = sorted((u & ood_uid) | (u & fep_uid))
            chk(f'L.{cid} {m["name"]}', not hp and not hu,
                f'{len(d)} rows: PDB ID hits {len(hp)}{hp[:6] if hp else ""}, '
                f'unique_id hits {len(hu)}{hu[:6] if hu else ""}')

    head('Summary')
    n = {k: sum(1 for s, _, _ in RES if s == k) for k in ['PASS', 'FAIL', 'SKIP']}
    print(f"PASS {n['PASS']}  FAIL {n['FAIL']}  SKIP {n['SKIP']}", flush=True)
    for s, t, m in RES:
        if s == 'FAIL':
            print(f'  ✗ {t} — {m}', flush=True)
    json.dump([{'status': s, 'check': t, 'detail': m} for s, t, m in RES],
              open(f'{E}/audit_report.json', 'w'), indent=1, ensure_ascii=False)
    print(f'\nOutputs → {E}/audit_report.json', flush=True)
    return 1 if n['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main())
