#!/usr/bin/env python
"""
阶段 2a：构建 HiQBind 索引（pK + 结构路径 + 评测集泄漏剔除）。

严格依据 PLAN.md：
  - 2.1 表1   HiQBind 原始规模 32,275（已核对 = sm 31,572 + poly 703）
  - 3.2       HiQBind 去重：CleanSplit 三重标准；剔除与 OOD/FEP 在 PDB ID 层面重叠的样本
  - 3.2       需统计净新增复合物数、唯一 UniProt 靶点数、配体 Bemis-Murcko 骨架数

本脚本只做「解析 + pK + 删失过滤 + 结构配对 + 评测集剔除 + 统计」，
CleanSplit 三重去重（需 TM-align）由 02b 脚本承接。

pK 规则（与 PLAN §2 决定一一致）：
  - pK = -`Log Binding Affinity`（该列为 log10(M)，已验证与 value+unit 重算吻合，最大差 3.8e-13）
  - 仅保留 `Binding Affinity Sign` == '=' 的精确测量值；剔除 ~ / > / >= / < / <= 的
    删失或近似值。这与 DCS 上从参考实现反解出的口径一致（删失值整行剔除）。
"""
import os, json, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX  = f'{ROOT}/datasets/index/hiqbind'
STR  = f'{ROOT}/datasets/structure/hiqbind'
OUT  = f'{ROOT}/experiments/02_hiqbind'
META = f'{ROOT}/experiments/00_meta'
os.makedirs(OUT, exist_ok=True)

log = lambda *a: print(*a, flush=True)


def scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def main():
    rep = {}
    d = pd.read_csv(f'{IDX}/hiqbind_metadata.csv')
    n0 = len(d)
    log(f'[HiQBind] 原始 {n0} 行（PLAN 表1 记载 32,275）')

    # 标记 sm / poly 来源，以便定位结构目录
    sm = set(pd.read_csv(f'{IDX}/hiqbind_sm_metadata.csv')
             .apply(lambda r: (r['PDBID'], r['Ligand Name'], r['Ligand Chain'],
                               r['Ligand Residue Number']), axis=1))
    key = d.apply(lambda r: (r['PDBID'], r['Ligand Name'], r['Ligand Chain'],
                             r['Ligand Residue Number']), axis=1)
    d['subset'] = np.where(key.isin(sm), 'sm', 'poly')
    log(f'  子集 {d.subset.value_counts().to_dict()}')

    # --- pK ---
    d['pK'] = -d['Log Binding Affinity']
    d['meas_type'] = d['Binding Affinity Measurement'].str.upper()

    # --- 删失/近似值剔除 ---
    sign = d['Binding Affinity Sign'].astype(str).str.strip()
    exact = sign == '='
    log(f'  Sign 分布 {sign.value_counts().to_dict()}')
    log(f'  剔除非精确测量 {(~exact).sum()} 行 → 保留 {exact.sum()}')
    d = d[exact].copy()

    # pK 有限性
    fin = np.isfinite(d['pK'])
    if (~fin).sum():
        log(f'  剔除 pK 非有限 {(~fin).sum()} 行')
        d = d[fin].copy()

    # --- 结构目录配对 ---
    d['complex_dir'] = (d['PDBID'].astype(str) + '_' + d['Ligand Name'].astype(str) + '_'
                        + d['Ligand Chain'].astype(str) + '_'
                        + d['Ligand Residue Number'].astype(str))
    sub_dir = {'sm': 'raw_data_hiq_sm', 'poly': 'raw_data_hiq_poly'}
    d['ligand_sdf'] = [os.path.join(STR, sub_dir[s], p, c, f'{c}_ligand_refined.sdf')
                       for s, p, c in zip(d.subset, d.PDBID, d.complex_dir)]
    d['protein_pdb'] = [os.path.join(STR, sub_dir[s], p, c, f'{c}_protein_refined.pdb')
                        for s, p, c in zip(d.subset, d.PDBID, d.complex_dir)]

    has = [os.path.exists(a) and os.path.exists(b)
           for a, b in zip(d.ligand_sdf, d.protein_pdb)]
    log(f'  结构文件齐备 {sum(has)} / {len(d)}（缺失 {len(d) - sum(has)}）')
    rep['missing_structure'] = int(len(d) - sum(has))
    d = d[has].copy()

    # --- 评测集泄漏剔除（PDB ID 前 4 位，PLAN §3.4.2）---
    ids = json.load(open(f'{META}/leakage_ids.json'))
    ood, fep = set(ids['ood_pdb_ids']), set(ids['fep_pdb_ids'])
    d['pdb_id'] = d['PDBID'].astype(str).str.lower().str[:4]
    hit_ood = d['pdb_id'].isin(ood)
    hit_fep = d['pdb_id'].isin(fep)
    hit = hit_ood | hit_fep
    log(f'  泄漏剔除 {len(d)} → {(~hit).sum()}'
        f'（命中 {hit.sum()}；OOD {hit_ood.sum()}、FEP {hit_fep.sum()}）')
    rep['leak_removed'] = {'total': int(hit.sum()), 'ood': int(hit_ood.sum()),
                           'fep': int(hit_fep.sum())}
    d = d[~hit].copy().reset_index(drop=True)

    # --- 统计（PLAN §3.2 要求的三项）---
    d['unique_id'] = d['complex_dir']
    d['source'] = 'hiqbind'
    log('  计算 Bemis-Murcko 骨架 …')
    d['scaffold'] = [scaffold(s) for s in d['Ligand SMILES']]

    log(f'\n[HiQBind] 清洗后 {len(d)} 条')
    log(f'  唯一 PDB ID        {d.pdb_id.nunique()}')
    log(f'  唯一 UniProt 靶点  {d["Protein UniProtID"].nunique()}')
    log(f'  唯一 Bemis-Murcko  {d.scaffold.nunique()}（解析失败 {d.scaffold.isna().sum()}）')
    log(f'  pK 范围 {d.pK.min():.2f} ~ {d.pK.max():.2f}，均值 {d.pK.mean():.3f}')
    log(f'  测量类型 {d.meas_type.value_counts().to_dict()}')
    res = pd.to_numeric(d['Resolution'], errors='coerce')
    log(f'  分辨率 中位 {res.median():.2f} Å，>2.5 Å 的有 {(res > 2.5).sum()} 条'
        f'（缺失 {res.isna().sum()}）')

    rep.update({'raw': int(n0), 'after_clean': int(len(d)),
                'unique_pdb': int(d.pdb_id.nunique()),
                'unique_uniprot': int(d['Protein UniProtID'].nunique()),
                'unique_scaffold': int(d.scaffold.nunique()),
                'res_gt_2p5': int((res > 2.5).sum())})

    cols = ['source', 'unique_id', 'pdb_id', 'PDBID', 'Ligand Name', 'Ligand Chain',
            'Ligand Residue Number', 'subset', 'pK', 'meas_type', 'Resolution',
            'Protein UniProtID', 'Ligand SMILES', 'scaffold', 'ligand_sdf', 'protein_pdb']
    d[cols].to_csv(f'{OUT}/hiqbind_clean.csv', index=False)
    json.dump(rep, open(f'{OUT}/report_02a.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\n产物 → {OUT}/hiqbind_clean.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
