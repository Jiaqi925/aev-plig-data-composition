#!/usr/bin/env python
"""
Pre-stage 5: assemble the training index tables of the 10 experiment
configurations.

Configuration composition:
   1 baseline                  = PDBbind + BindingNet + BindingDB-DCS
                                 (direct concatenation)
   2 baseline + hiqbind        + all of HiQBind after deduplication
   3/4/5 baseline + gator_filtered_30/60/100%
   6/7/8 baseline + gator_unfiltered_30/60/100%
   9 bindingnet + dcs + hiqbind  (PDBbind removed)
  10 config 9 + gator_best     (set with --config10 once the FEP wmPCC of
                                configs 3-8 is available)

The leakage check is performed at the same time: each configuration's PDB ID
set is intersected with the OOD 295 / FEP reference PDBs and must be empty;
the Gator side has no real PDB IDs, and its leakage control was already done
in 03b.
If 04_graphs has been generated, samples without a graph are also removed and
their number is reported as-is.

Usage:
  python 05_build_configs.py                       # build configs 1-9
  python 05_build_configs.py --config10 filtered_60  # add config 10
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
META = f'{E}/00_meta'
GRAPHS = f'{E}/04_graphs'
OUT = f'{E}/05_configs'
log = lambda *a: print(*a, flush=True)

NAMES = {1: 'baseline', 2: 'baseline+hiqbind',
         3: 'baseline+gator_filtered_30', 4: 'baseline+gator_filtered_60',
         5: 'baseline+gator_filtered_100',
         6: 'baseline+gator_unfiltered_30', 7: 'baseline+gator_unfiltered_60',
         8: 'baseline+gator_unfiltered_100',
         9: 'bindingnet+dcs+hiqbind', 10: 'bindingnet+dcs+hiqbind+gator_best'}


def load_parts():
    p = {}
    b = f'{E}/01_baseline_index'
    d = pd.read_csv(f'{b}/pdbbind_clean.csv')
    p['pdbbind'] = pd.DataFrame({'source': 'pdbbind', 'unique_id': d.unique_id,
                                 'pdb_id': d.pdb_id, 'pK': d.pK})
    d = pd.read_csv(f'{b}/bindingnet_clean.csv')
    p['bindingnet'] = pd.DataFrame({'source': 'bindingnet', 'unique_id': d.unique_id,
                                    'pdb_id': d.pdb_id, 'pK': d.pK})
    d = pd.read_csv(f'{b}/bindingdb_dcs_clean.csv')
    p['bindingdb_dcs'] = pd.DataFrame({'source': 'bindingdb_dcs', 'unique_id': d.unique_id,
                                       'pdb_id': d.pdb_id, 'pK': d.pK})
    d = pd.read_csv(f'{E}/02_hiqbind/hiqbind_dedup.csv')
    p['hiqbind'] = pd.DataFrame({'source': 'hiqbind', 'unique_id': d.unique_id,
                                 'pdb_id': d.pdb_id, 'pK': d.pK})
    g = pd.read_csv(f'{E}/03_gator/gator_dedup.csv')
    p['_gator'] = g
    for ver in ['filtered', 'unfiltered']:
        for pct in [30, 60, 100]:
            c = f'in_{ver}_{pct}'
            s = g[g[c]]
            p[f'gator_{ver}_{pct}'] = pd.DataFrame(
                {'source': 'gatoraffinity', 'unique_id': s.unique_id,
                 'pdb_id': '', 'pK': s.pK})
    return p


COMPOSE = {
    1: ['pdbbind', 'bindingnet', 'bindingdb_dcs'],
    2: ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'hiqbind'],
    3: ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'gator_filtered_30'],
    4: ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'gator_filtered_60'],
    5: ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'gator_filtered_100'],
    6: ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'gator_unfiltered_30'],
    7: ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'gator_unfiltered_60'],
    8: ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'gator_unfiltered_100'],
    9: ['bindingnet', 'bindingdb_dcs', 'hiqbind'],
}


def graph_ids():
    """Return {source: set(uid)}; if the directory is missing, that source
    maps to None (meaning no filtering)."""
    out = {}
    for s in ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'hiqbind', 'gatoraffinity']:
        f = f'{GRAPHS}/{s}/ids.npy'
        out[s] = set(np.load(f).tolist()) if os.path.exists(f) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config10', default=None,
                    help='e.g. filtered_60 / unfiltered_100; used to '
                         'assemble config 10')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    parts = load_parts()
    gi = graph_ids()
    for s, v in gi.items():
        log(f'  graphs available {s}: '
            f'{"not generated" if v is None else len(v)}')

    leak = json.load(open(f'{META}/leakage_ids.json'))
    leak_ids = set(leak['excluded_pdb_ids'])

    compose = dict(COMPOSE)
    if a.config10:
        compose[10] = ['bindingnet', 'bindingdb_dcs', 'hiqbind', f'gator_{a.config10}']

    meta_path = f'{OUT}/configs.json'
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for cid, keys in sorted(compose.items()):
        d = pd.concat([parts[k] for k in keys], ignore_index=True)
        n0 = len(d)

        # Leakage check (Gator rows have an empty pdb_id and are skipped)
        pid = d.pdb_id.fillna('').astype(str).str.lower().str[:4]
        bad = pid.isin(leak_ids) & (pid != '')
        assert not bad.any(), (f'config {cid} still contains leaked PDB IDs '
                               f'{sorted(set(pid[bad]))[:10]}')

        # Filter by graph availability
        drops = {}
        keep = np.ones(len(d), bool)
        for s, ids in gi.items():
            if ids is None:
                continue
            m = (d.source == s).values
            if not m.any():
                continue
            ok = d.unique_id.isin(ids).values
            drops[s] = int((m & ~ok).sum())
            keep &= (~m) | ok
        d = d[keep].reset_index(drop=True)

        d[['source', 'unique_id', 'pK']].to_csv(f'{OUT}/config_{cid:02d}.csv', index=False)
        cnt = d.source.value_counts().to_dict()
        meta[str(cid)] = {'name': NAMES[cid], 'parts': keys, 'n': int(len(d)),
                          'n_before_graph_filter': int(n0),
                          'dropped_no_graph': drops, 'by_source': cnt}
        log(f'[config {cid:2d}] {NAMES[cid]:<40s} {len(d):>7d} entries  {cnt}'
            + (f'  removed (no graph) {drops}' if any(drops.values()) else ''))

    json.dump(meta, open(meta_path, 'w'), indent=1, ensure_ascii=False)
    log(f'\nOutputs → {OUT}/{{config_XX.csv, configs.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
