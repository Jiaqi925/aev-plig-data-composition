#!/usr/bin/env python
"""
Scaffold for smoke testing only (not one of the experiment configurations).

While 04_graphs has not been fully generated, assemble a temporary config 99
from the **sources whose graphs already exist**, so that the whole pipeline
"training → OOD/FEP inference → stage 6 evaluation" can be debugged early.

Outputs: experiments/05_configs/config_99.csv + the "99" entry in configs.json
         (configs.json is updated incrementally; the real configs 1–10 are
         unaffected)

Usage:
  python 99_make_smoke_config.py                 # default: 3000 from pdbbind
  python 99_make_smoke_config.py --n 800 --sources pdbbind hiqbind
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
GRAPHS = f'{E}/04_graphs'
OUT = f'{E}/05_configs'

SRC_INDEX = {
    'pdbbind': f'{E}/01_baseline_index/pdbbind_clean.csv',
    'bindingnet': f'{E}/01_baseline_index/bindingnet_clean.csv',
    'bindingdb_dcs': f'{E}/01_baseline_index/bindingdb_dcs_clean.csv',
    'hiqbind': f'{E}/02_hiqbind/hiqbind_dedup.csv',
    'gatoraffinity': f'{E}/03_gator/gator_dedup.csv',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='*', default=['pdbbind'])
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--cid', type=int, default=99)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    frames = []
    for s in a.sources:
        f = f'{GRAPHS}/{s}/ids.npy'
        if not os.path.exists(f):
            print(f'  ! {s} graphs not generated, skipped', flush=True)
            continue
        have = set(np.load(f).tolist())
        d = pd.read_csv(SRC_INDEX[s])
        d = d[d.unique_id.isin(have)]
        frames.append(pd.DataFrame({'source': s, 'unique_id': d.unique_id, 'pK': d.pK}))
        print(f'  {s}: graphs {len(have)}, in index {len(d)}', flush=True)
    if not frames:
        print('No source available, exiting.'); return 1

    d = pd.concat(frames, ignore_index=True)
    if a.n and len(d) > a.n:
        d = d.sample(n=a.n, random_state=42).reset_index(drop=True)
    d.to_csv(f'{OUT}/config_{a.cid:02d}.csv', index=False)

    mp = f'{OUT}/configs.json'
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    meta[str(a.cid)] = {'name': f'SMOKE({"+".join(a.sources)})', 'parts': a.sources,
                        'n': int(len(d)), 'by_source': d.source.value_counts().to_dict(),
                        'note': 'smoke-test scaffold, not an experiment config'}
    json.dump(meta, open(mp, 'w'), indent=1, ensure_ascii=False)
    print(f'\nConfig {a.cid}: {len(d)} entries → {OUT}/config_{a.cid:02d}.csv',
          flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
