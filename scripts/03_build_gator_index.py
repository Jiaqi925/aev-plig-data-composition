#!/usr/bin/env python
"""
阶段 3a：构建 GatorAffinity-DB 索引（pK + 结构配对 + 质量分 + 序列）。

严格依据 PLAN §3.3：
  - pK 仅取 `prediction_type` 指向的列，且只接受 Ki / Kd（IC50/EC50 不是热力学结合常数，
    依赖酶浓度与底物条件，与其余数据源的 pK 口径不可比）
  - 质量筛选 Boltz-1 confidence_score >= 0.7，同时保留 filtered / unfiltered 两版
  - 与实际下载的结构目录取交集（计划记载 65,031，实际 65,001）
  - 内部去重：BindingDB MonomerID + 规范化 SMILES
  - 评测集分离不走 PDB ID（Gator 无真实 PDB ID），改用序列一致性 < 30% + Tanimoto < 0.9，
    由 03b 脚本承接；本脚本只准备 SMILES / 序列 / FASTA

索引 943MB，按需读列以控内存。
"""
import os, sys, json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX = f'{ROOT}/datasets/index/gatoraffinity/GatorAffnity-DB_index.csv'
STR = f'{ROOT}/datasets/structure/gatoraffinity'
OUT = f'{ROOT}/experiments/03_gator'
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)

USECOLS = ['GatorAffinity_ID', 'has_structure_prediction', 'prediction_type',
           'confidence_score', 'Ligand SMILES', 'BindingDB MonomerID',
           'Target Name', 'Ki (nM)', 'Kd (nM)',
           'BindingDB Target Chain Sequence',
           'UniProt (SwissProt) Primary ID of Target Chain',
           'PDB ID(s) for Ligand-Target Complex',
           'has_raw_structure', 'has_fixed_structure']

CONF_THR = 0.7


def pk_from_nm(v):
    """nM -> p 值；非正数或非有限返回 NaN。"""
    v = pd.to_numeric(v, errors='coerce')
    return np.where(v > 0, 9.0 - np.log10(v.where(v > 0)), np.nan)


def canon(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def scaffold(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else None
    except Exception:
        return None


def main():
    rep = {}
    log(f'读索引（按需列）…')
    d = pd.read_csv(IDX, usecols=USECOLS, low_memory=False)
    rep['raw'] = int(len(d))
    log(f'[Gator] 原始 {len(d)} 行（PLAN 表1 记载 463,867）')

    # --- pK：仅 Ki / Kd，按 prediction_type 取对应列 ---
    pt = d['prediction_type'].astype(str).str.strip().str.upper()
    log(f'  prediction_type 分布 {pt.value_counts().to_dict()}')
    ki, kd = pk_from_nm(d['Ki (nM)']), pk_from_nm(d['Kd (nM)'])
    d['pK'] = np.where(pt == 'KI', ki, np.where(pt == 'KD', kd, np.nan))
    d['meas_type'] = pt
    keep = pt.isin(['KI', 'KD']) & np.isfinite(d['pK'])
    log(f'  仅保留 Ki/Kd 且 pK 有限：{len(d)} → {keep.sum()}')
    rep['after_pk'] = int(keep.sum())
    d = d[keep].copy()

    # --- 结构配对（与实际下载目录取交集）---
    # 索引里 GatorAffinity_ID 是纯整数，结构目录名为 GID{id}
    have = set(os.listdir(STR))
    rep['structure_dirs'] = len(have)
    log(f'  实际结构目录 {len(have)} 个（PLAN 记载 65,031）')
    d['gid'] = 'GID' + d['GatorAffinity_ID'].astype(int).astype(str)
    d = d[d.gid.isin(have)].copy()
    log(f'  与结构目录取交集 → {len(d)}')

    g = d['gid']
    d['ligand_sdf'] = [f'{STR}/{i}/ligand/{i}_ligand_0_fixed.sdf' for i in g]
    d['protein_pdb'] = [f'{STR}/{i}/protein/{i}_protein_0_fixed.pdb' for i in g]
    ok = [os.path.exists(a) and os.path.exists(b)
          for a, b in zip(d.ligand_sdf, d.protein_pdb)]
    log(f'  结构文件齐备 {sum(ok)} / {len(d)}（缺失 {len(d) - sum(ok)}）')
    rep['missing_structure'] = int(len(d) - sum(ok))
    d = d[ok].copy()

    # --- 内部去重：BindingDB MonomerID + 规范化 SMILES ---
    log('  规范化 SMILES …')
    d['smiles_canon'] = [canon(s) for s in d['Ligand SMILES']]
    bad = d.smiles_canon.isna().sum()
    log(f'  SMILES 不可解析 {bad} 行（剔除）')
    rep['smiles_unparsable'] = int(bad)
    d = d[d.smiles_canon.notna()].copy()

    n_before = len(d)
    # 去重键 = 靶点 + 配体。
    # 注意：PLAN §3.3.1 写的是「BindingDB ID + 配体 SMILES」，但实测 BindingDB MonomerID
    # 是**配体**标识符（158,836 个 MonomerID 对应 158,938 个 SMILES，近乎一一对应），
    # 若按 MonomerID+SMILES 去重等同于「只按配体去重」，会把同一配体结合不同靶点的记录
    # 一并合并——而 36.2% 的配体结合多个靶点（最多 384 个），那些是彼此独立的有效样本。
    # 故按「靶点 UniProt + 配体 SMILES」去重，这才是「同一蛋白-配体对的重复记录」的本意。
    # 靶点标识优先用 UniProt Primary ID，缺失时回退到 Target Name。
    uni = d['UniProt (SwissProt) Primary ID of Target Chain'].astype(str).str.strip()
    tgt = uni.where(uni.notna() & (uni != '') & (uni.str.lower() != 'nan'),
                    d['Target Name'].astype(str).str.strip())
    d['target_key'] = tgt
    d['dup_key'] = tgt + '|' + d['smiles_canon']
    conf = pd.to_numeric(d['confidence_score'], errors='coerce')
    d['conf'] = conf
    # 同一蛋白-配体对的多条测量：pK 取中位数，结构保留 confidence 最高者
    d['pK_median'] = d.groupby('dup_key')['pK'].transform('median')
    d = (d.sort_values('conf', ascending=False)
           .drop_duplicates('dup_key', keep='first').copy())
    d['pK'] = d['pK_median']
    log(f'  内部去重（UniProt+SMILES，pK 取中位数、留最高 confidence）：{n_before} → {len(d)}')
    rep['after_internal_dedup'] = int(len(d))

    # --- 质量分 ---
    d['pass_conf'] = d['conf'] >= CONF_THR
    log(f'  confidence >= {CONF_THR}：{d.pass_conf.sum()} / {len(d)}'
        f'（缺失 {d.conf.isna().sum()}）')
    log(f'  confidence 分位 {d.conf.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()}')
    rep['pass_conf'] = int(d.pass_conf.sum())

    # --- 序列（供 MMseqs2 与 03b 泄漏分离）---
    d['seq'] = d['BindingDB Target Chain Sequence'].astype(str).str.strip().str.upper()
    d.loc[~d.seq.str.fullmatch(r'[A-Z]+', na=False), 'seq'] = ''
    log(f'  序列为空 {(d.seq == "").sum()}；长度中位 '
        f'{d[d.seq != ""].seq.str.len().median():.0f}')
    rep['seq_empty'] = int((d.seq == '').sum())

    d['scaffold'] = [scaffold(s) for s in d.smiles_canon]
    d['source'] = 'gatoraffinity'
    d['unique_id'] = d['gid']

    log(f'\n[Gator] 清洗后 {len(d)} 条')
    log(f'  唯一配体      {d.smiles_canon.nunique()}')
    log(f'  唯一 UniProt  {d["UniProt (SwissProt) Primary ID of Target Chain"].nunique()}')
    log(f'  唯一 BM 骨架  {d.scaffold.nunique()}（失败 {d.scaffold.isna().sum()}）')
    log(f'  pK 范围 {d.pK.min():.2f} ~ {d.pK.max():.2f}，均值 {d.pK.mean():.3f}')
    log(f'  测量类型 {d.meas_type.value_counts().to_dict()}')

    rep.update({'after_clean': int(len(d)),
                'unique_ligand': int(d.smiles_canon.nunique()),
                'unique_uniprot': int(d['UniProt (SwissProt) Primary ID of Target Chain'].nunique()),
                'unique_scaffold': int(d.scaffold.nunique()),
                'conf_threshold': CONF_THR})

    cols = ['source', 'unique_id', 'gid', 'GatorAffinity_ID', 'meas_type', 'pK', 'conf',
            'pass_conf', 'Ligand SMILES', 'smiles_canon', 'scaffold',
            'BindingDB MonomerID', 'Target Name', 'target_key',
            'UniProt (SwissProt) Primary ID of Target Chain',
            'PDB ID(s) for Ligand-Target Complex', 'seq',
            'ligand_sdf', 'protein_pdb']
    d[cols].to_csv(f'{OUT}/gator_clean.csv', index=False)

    with open(f'{OUT}/pool_gator.fasta', 'w') as f:
        for i, s in zip(d.unique_id, d.seq):
            if s:
                f.write(f'>GA|{i}\n{s}\n')

    json.dump(rep, open(f'{OUT}/report_03a.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\n产物 → {OUT}/{{gator_clean.csv, pool_gator.fasta, report_03a.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
