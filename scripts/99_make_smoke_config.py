#!/usr/bin/env python
"""
冒烟测试专用脚手架（不属于 PLAN 的任何实验配置）。

在 04_graphs 尚未全部生成完时，用**已有图的源**装配一个临时配置 99，
以便提前把「训练 → OOD/FEP 推理 → 阶段 6 评估」整条链路调通。

产物：experiments/05_configs/config_99.csv + configs.json 里的 "99" 条目
      （configs.json 是增量更新，正式配置 1–10 不受影响）

用法：
  python 99_make_smoke_config.py                 # 默认 pdbbind 抽 3000 条
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
            print(f'  ! {s} 图未生成，跳过', flush=True)
            continue
        have = set(np.load(f).tolist())
        d = pd.read_csv(SRC_INDEX[s])
        d = d[d.unique_id.isin(have)]
        frames.append(pd.DataFrame({'source': s, 'unique_id': d.unique_id, 'pK': d.pK}))
        print(f'  {s}: 图 {len(have)}，索引可用 {len(d)}', flush=True)
    if not frames:
        print('没有任何源可用，退出。'); return 1

    d = pd.concat(frames, ignore_index=True)
    if a.n and len(d) > a.n:
        d = d.sample(n=a.n, random_state=42).reset_index(drop=True)
    d.to_csv(f'{OUT}/config_{a.cid:02d}.csv', index=False)

    mp = f'{OUT}/configs.json'
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    meta[str(a.cid)] = {'name': f'SMOKE({"+".join(a.sources)})', 'parts': a.sources,
                        'n': int(len(d)), 'by_source': d.source.value_counts().to_dict(),
                        'note': '冒烟测试脚手架，非 PLAN 配置'}
    json.dump(meta, open(mp, 'w'), indent=1, ensure_ascii=False)
    print(f'\n配置 {a.cid}：{len(d)} 条 → {OUT}/config_{a.cid:02d}.csv', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
