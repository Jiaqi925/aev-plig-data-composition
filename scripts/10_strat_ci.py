#!/usr/bin/env python
"""
补充任务 2：对 OOD 相似度分层的 low / high 层，重跑现成的 BCa 与配对差值代码。

动机：现有 strat.csv 只有点估计，报告里只能写「不宜过度解读」。本脚本给每层
配上 BCa 95% CI，并做各配置 vs 配置1 的层内配对差值检验，使「low 层排名」
这一观察有统计口径。

完全复用 lib_eval 的 bca_ci / paired_diff_ci / pcc / ktau，分层口径与
07_evaluate.py 的 strat_layers 一致（max_seqid < 0.30 判为 low）。
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
    """载入某配置 5 种子集成的 OOD 预测（与 07_evaluate 的 ensemble 同口径：逐样本均值）"""
    ps = []
    for s in SEEDS:
        f = f'{TRAIN}/cfg{cid:02d}/seed{s}/pred_ood295.npy'
        if not os.path.exists(f):
            return None
        ps.append(np.load(f).astype(float))
    return np.mean(np.vstack(ps), axis=0)


def main():
    # ---------- 真值 ----------
    ood = pd.read_csv(f'{IDX}/_testsets/index_oodtest.csv')
    ood = ood[ood.split == 'test'].reset_index(drop=True)
    y = ood['-logKd/Ki'].values.astype(float)

    # ---------- 分层标签：复用 07_evaluate 的 eval_maxid.csv，逐配置取所含源的最大相似度 ----------
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

    log('分层依据：experiments/00_meta/eval_maxid.csv（mmseqs 评测靶点 vs 各源靶点池），'
        '阈值 seqid 0.30，逐配置按其所含源取最大值')

    # ---------- 各配置分层 BCa ----------
    log(f'\n[1/2] 逐配置分层 BCa（n_boot={N_BOOT}）…')
    rows = []
    ens = {}
    lays = {}
    for cid in range(1, 12):
        p = load_ens(cid)
        if p is None:
            log(f'  配置 {cid}: 预测缺失，跳过')
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
                f'[{r_pcc["lo"]:.4f}, {r_pcc["hi"]:.4f}] 宽 {r_pcc["hi"]-r_pcc["lo"]:.4f}')

    pd.DataFrame(rows).to_csv(f'{OUT}/strat_ci.csv', index=False)

    # ---------- 层内配对差值 vs 配置 1 ----------
    log('\n[2/2] 层内配对差值（各配置 − 配置1）…')
    pd_rows = []
    if 1 in ens:
        p1 = ens[1]
        for cid in sorted(ens):
            if cid == 1:
                continue
            p2 = ens[cid]
            # 配对必须在同一批单位上，故用基准配置(1)的分层，避免两配置层成员不同
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
                    f'{"显著" if sig else "不显著"}')
        pd.DataFrame(pd_rows).to_csv(f'{OUT}/strat_paired_diff.csv', index=False)

    # ---------- high − low 层内落差的 CI（同一配置内，两层独立，用差值的正态近似不合适，
    #            改为分别报 CI 并说明区间是否重叠）----------
    log('\n[附] high 与 low 的 CI 是否重叠（重叠 ⇒ 落差在统计上不可断言）：')
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
            f'落差 {h.pcc-l.pcc:.4f} {"重叠" if overlap else "不重叠"}')
    pd.DataFrame(ov_rows).to_csv(f'{OUT}/strat_overlap.csv', index=False)

    json.dump({'n_boot': N_BOOT, 'seeds': SEEDS,
               'layer_rule': ('eval_maxid.csv (mmseqs, 评测靶点 vs 各源靶点池), '
                              'seqid >= 0.30 -> high, else low; 逐配置按所含源取最大值'),
               'paired_diff_layer_basis': 'config 1 (保证配对单位一致)',
               'n_by_config': {str(c): {'high': int((lays[c] == 'high').sum()),
                                        'low': int((lays[c] == 'low').sum())}
                               for c in sorted(lays)},
               'strat_ci': rows, 'paired_diff': pd_rows, 'overlap': ov_rows},
              open(f'{OUT}/strat_ci.json', 'w'), indent=1, default=float)
    log(f'\n产物 → {OUT}/{{strat_ci.csv, strat_paired_diff.csv, '
        f'strat_overlap.csv, strat_ci.json}}')


if __name__ == '__main__':
    main()
