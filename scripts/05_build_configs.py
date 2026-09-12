#!/usr/bin/env python
"""
阶段 5 前置：装配 10 个实验配置的训练索引表（PLAN §4.1 表 2）。

配置组成：
   1 baseline                       = PDBbind + BindingNet + BindingDB-DCS（直接拼接）
   2 baseline + hiqbind             + HiQBind 去重后全量
   3/4/5 baseline + gator_filtered_30/60/100%
   6/7/8 baseline + gator_unfiltered_30/60/100%
   9 bindingnet + dcs + hiqbind     （移除 PDBbind）
  10 配置 9 + gator_best            （待配置 3-8 的 FEP wmPCC 出来后用 --config10 指定）

同时执行 PLAN §3.4.2 的泄漏验证：每个配置的 PDB ID 集合与 OOD 295 / FEP 参考 PDB
求交集，必须为空；Gator 侧无真实 PDB ID，其泄漏控制已在 03b 完成。
若 04_graphs 已生成，则同时剔除没有图的样本并如实报告数量。

用法：
  python 05_build_configs.py                       # 建配置 1-9
  python 05_build_configs.py --config10 filtered_60  # 补建配置 10
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
    """返回 {source: set(uid)}；缺目录则该源返回 None（表示不过滤）。"""
    out = {}
    for s in ['pdbbind', 'bindingnet', 'bindingdb_dcs', 'hiqbind', 'gatoraffinity']:
        f = f'{GRAPHS}/{s}/ids.npy'
        out[s] = set(np.load(f).tolist()) if os.path.exists(f) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config10', default=None,
                    help='如 filtered_60 / unfiltered_100，用于装配配置 10')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    parts = load_parts()
    gi = graph_ids()
    for s, v in gi.items():
        log(f'  图可用 {s}: {"未生成" if v is None else len(v)}')

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

        # 泄漏验证（Gator 行 pdb_id 为空，不参与）
        pid = d.pdb_id.fillna('').astype(str).str.lower().str[:4]
        bad = pid.isin(leak_ids) & (pid != '')
        assert not bad.any(), f'配置 {cid} 仍含泄漏 PDB ID {sorted(set(pid[bad]))[:10]}'

        # 图可用性过滤
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
        log(f'[配置 {cid:2d}] {NAMES[cid]:<40s} {len(d):>7d} 条  {cnt}'
            + (f'  缺图剔除 {drops}' if any(drops.values()) else ''))

    json.dump(meta, open(meta_path, 'w'), indent=1, ensure_ascii=False)
    log(f'\n产物 → {OUT}/{{config_XX.csv, configs.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
