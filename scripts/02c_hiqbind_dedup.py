#!/usr/bin/env python
"""
阶段 2c：HiQBind CleanSplit 三重去重（PLAN §3.1 / §3.2）。

去重对象：PLAN §3.1 明确「HiQBind：执行去重操作（移除与 PDBbind 重复的样本）」，
故比对基准是 PDBbind（泄漏剔除后 19,118 条），不涉及 BindingNet / BindingDB-DCS。
方向：只剔 HiQBind 侧，PDBbind 保持不动（配置 2 = Baseline + HiQBind 去重后全量）。

三重判据（同时满足才判重复）：
  1. 配体 ECFP4 Tanimoto > 0.9
  2. |ΔpK| <= 1.0
  3. 蛋白 TM-align TM-score > 0.8

执行顺序按代价从低到高：先配体指纹（带 popcount 剪枝），再 pK，最后 TM-align。
这不改变判据本身——三者是合取关系，任何顺序得到的重复集合相同。

TM-score 归一化：TM-align 同时给出按 Chain_1 / Chain_2 长度归一的两个值，
本脚本取二者较大值。较大值判定更容易越过 0.8，即倾向于「判为重复」，
对「净新增复合物数」是保守（不高估）的口径。
"""
import os, sys, json, subprocess, re
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = f'{ROOT}/experiments/00_meta'
OUT = f'{ROOT}/experiments/02_hiqbind'
NPROC = min(64, os.cpu_count())
T_TANI, T_DPK, T_TM = 0.9, 1.0, 0.8
log = lambda *a: print(*a, flush=True)

_TM_RE = re.compile(r'TM-score=\s*([0-9.]+)')


def canon(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def fp_of(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


_HF = _PF = _BYPC = _PCS = None      # fork 继承给子进程，避免 pickle 大对象


def match_block(rng):
    """对一批 HiQBind 唯一配体，找出 Tanimoto>阈值 的 PDBbind 唯一配体。"""
    out = []
    for j in rng:
        f = _HF[j]
        if f is None:
            continue
        a = f.GetNumOnBits()
        lo, hi = int(np.ceil(T_TANI * a)), int(np.floor(a / T_TANI))
        cand = [i for pc in _PCS if lo <= pc <= hi for i in _BYPC[pc]]
        if not cand:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(f, [_PF[i] for i in cand])
        out += [(j, i, s) for i, s in zip(cand, sims) if s > T_TANI]
    return out


def tmscore(a):
    """返回 max(按链1归一, 按链2归一) 的 TM-score。"""
    p, q = a
    try:
        r = subprocess.run(['TMalign', p, q], capture_output=True,
                           text=True, timeout=600)
        v = [float(x) for x in _TM_RE.findall(r.stdout)]
        return max(v) if v else -1.0
    except Exception:
        return -1.0


def main():
    log(f'并行 {NPROC}；阈值 Tanimoto>{T_TANI}, |dpK|<={T_DPK}, TM>{T_TM}')

    hq = pd.read_csv(f'{OUT}/hiqbind_clean.csv')
    pb = pd.read_csv(f'{ROOT}/experiments/01_baseline_index/pdbbind_clean.csv')
    ps = pd.read_csv(f'{META}/pdbbind_smiles_seq.csv')
    pb = pb.merge(ps[['pdb_id', 'smiles']], on='pdb_id', how='left')
    pb['prot'] = [f'{ROOT}/datasets/structure/pdbbind/'
                  f'{"refined-set" if r else "general-set"}/{i}/{i}_protein.pdb'
                  for i, r in zip(pb.pdb_id, pb.refined)]
    log(f'HiQBind {len(hq)} 条；PDBbind {len(pb)} 条'
        f'（SMILES 缺失 {pb.smiles.isna().sum()}，这些行无法参与配体判据）')

    # ---------- 1. 规范化 SMILES ----------
    with Pool(NPROC) as p:
        hq['csmi'] = p.map(canon, hq['Ligand SMILES'].astype(str), chunksize=256)
        pbs = pb.smiles.fillna('').astype(str).tolist()
        pb['csmi'] = p.map(canon, pbs, chunksize=256)
    log(f'  规范化失败 HiQBind {hq.csmi.isna().sum()}，PDBbind {pb.csmi.isna().sum()}')

    hu = sorted(set(hq.csmi.dropna()))
    pu = sorted(set(pb.csmi.dropna()))
    log(f'  唯一配体 HiQBind {len(hu)}，PDBbind {len(pu)}')

    with Pool(NPROC) as p:
        hf = p.map(fp_of, hu, chunksize=128)
        pf = p.map(fp_of, pu, chunksize=128)

    # ---------- 2. 配体判据：popcount 剪枝 + Tanimoto ----------
    # T=c/(a+b-c)>0.9 且 c<=min(a,b)  =>  min(a,b) > 0.9*max(a,b)，据此只比相近桶
    global _HF, _PF, _BYPC, _PCS
    bypc = defaultdict(list)
    for i, f in enumerate(pf):
        if f is not None:
            bypc[f.GetNumOnBits()].append(i)
    _HF, _PF, _BYPC, _PCS = hf, pf, bypc, sorted(bypc)

    blocks = [range(k, min(k + 200, len(hu))) for k in range(0, len(hu), 200)]
    log(f'[配体] {len(hu)} × {len(pu)} 唯一配体，{len(blocks)} 块 …')
    with Pool(NPROC) as p:
        lig = [x for b in p.map(match_block, blocks) for x in b]
    log(f'  Tanimoto>{T_TANI} 的配体对 {len(lig)}')

    # ---------- 3. 展开为样本对 + pK 判据 ----------
    hq_by = defaultdict(list)
    for k, s in enumerate(hq.csmi):
        if isinstance(s, str):
            hq_by[s].append(k)
    pb_by = defaultdict(list)
    for k, s in enumerate(pb.csmi):
        if isinstance(s, str):
            pb_by[s].append(k)

    hpk, ppk = hq.pK.values, pb.pK.values
    pairs = []
    for j, i, s in lig:
        for x in hq_by[hu[j]]:
            for y in pb_by[pu[i]]:
                if abs(hpk[x] - ppk[y]) <= T_DPK:
                    pairs.append((x, y, s))
    log(f'[pK] 同时满足 Tanimoto 与 |dpK|<={T_DPK} 的样本对 {len(pairs)}')

    # ---------- 4. TM-align（仅对存活对的唯一结构组合） ----------
    hprot, pprot = hq.protein_pdb.values, pb.prot.values
    uniq = sorted({(hprot[x], pprot[y]) for x, y, _ in pairs})
    log(f'[TM-align] 唯一结构对 {len(uniq)}，开始比对 …')
    with Pool(NPROC) as p:
        tms = p.map(tmscore, uniq, chunksize=4)
    tm = dict(zip(uniq, tms))
    nfail = sum(1 for v in tms if v < 0)
    log(f'  失败 {nfail}；TM>{T_TM} 的结构对 {sum(1 for v in tms if v > T_TM)}')

    # ---------- 5. 判定重复 ----------
    dup = np.zeros(len(hq), bool)
    rec = []
    for x, y, s in pairs:
        v = tm[(hprot[x], pprot[y])]
        if v > T_TM:
            dup[x] = True
            rec.append((hq.unique_id[x], pb.pdb_id[y], s,
                        abs(hpk[x] - ppk[y]), v))
    pd.DataFrame(rec, columns=['hq_unique_id', 'pb_pdb_id', 'tanimoto',
                               'abs_dpK', 'tm_score']).to_csv(
        f'{OUT}/hiqbind_dup_pairs.csv', index=False)
    log(f'\n[判定] 三重判据命中 HiQBind {int(dup.sum())} 条 → 保留 {int((~dup).sum())}')

    keep = hq[~dup].copy().reset_index(drop=True)
    keep.to_csv(f'{OUT}/hiqbind_dedup.csv', index=False)

    # ---------- 6. PLAN §3.2 要求的三项统计 ----------
    pb_pdb = set(pb.pdb_id)
    log(f'  净新增复合物数     {len(keep)}')
    log(f'  其中 PDB ID 不在 PDBbind 中 {int((~keep.pdb_id.isin(pb_pdb)).sum())}')
    log(f'  唯一 UniProt 靶点  {keep["Protein UniProtID"].nunique()}')
    log(f'  唯一 BM 骨架       {keep.scaffold.nunique()}')
    log(f'  pK 范围 {keep.pK.min():.2f} ~ {keep.pK.max():.2f}，均值 {keep.pK.mean():.3f}')

    json.dump({'hiqbind_in': int(len(hq)), 'pdbbind_ref': int(len(pb)),
               'pdbbind_smiles_missing': int(pb.smiles.isna().sum()),
               'lig_pairs': len(lig), 'pairs_after_pK': len(pairs),
               'tm_pairs': len(uniq), 'tm_fail': int(nfail),
               'duplicates_removed': int(dup.sum()),
               'net_new': int(len(keep)),
               'net_new_unseen_pdb': int((~keep.pdb_id.isin(pb_pdb)).sum()),
               'unique_uniprot': int(keep['Protein UniProtID'].nunique()),
               'unique_scaffold': int(keep.scaffold.nunique()),
               'thresholds': {'tanimoto': T_TANI, 'abs_dpK': T_DPK,
                              'tm_score': T_TM, 'tm_norm': 'max(chain1,chain2)'}},
              open(f'{OUT}/report_02c.json', 'w'), indent=1)
    log(f'\n产物 → {OUT}/{{hiqbind_dedup.csv, hiqbind_dup_pairs.csv, report_02c.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
