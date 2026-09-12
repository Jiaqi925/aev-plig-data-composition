#!/usr/bin/env python
"""
阶段 1：构建 Baseline 三源索引（PDBbind + BindingNet + BindingDB-DCS）并执行评测集泄漏剔除。

严格依据 PLAN.md：
  - 2.1 表1  原始规模：PDBbind 19,443 / BindingNet 69,816 / DCS 8,822
  - 3.1      基线三源直接拼接，不执行去重
  - 3.4.2    以 PDB ID（前 4 位）为统一键，剔除 OOD(295) 与 FEP(42 系列) 涉及的样本

pK 换算规则（反解自 AEV-PLIG 参考实现 data/bindingdb_processed.csv，
在全部 8,822 行上 100% 复现，故沿用而非另立标准）：
  1. 测量类型优先级 Kd > Ki > IC50，取优先级最高且有值的一列
  2. 该列内可能有多个分号分隔的浓度值，各自换算为 p 值后取中位数
  3. p = 9 - log10(浓度 nM)；浓度为 0 时 p = +inf
  4. 剔除含不等号（> 或 <）的删失值行
  5. 仅当中位数非有限时剔除该行。中位数对个别 0 值天然稳健（inf 只排到序列末端，
     不影响中位数取值），这与参考实现的行为逐行一致。
PDBbind 与 BindingNet 的索引已自带 pK 列，直接沿用，不重新换算。
"""
import os, re, glob, json, sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDX  = f'{ROOT}/datasets/index'
STR  = f'{ROOT}/datasets/structure'
OUT  = f'{ROOT}/experiments/01_baseline_index'
META = f'{ROOT}/experiments/00_meta'
os.makedirs(OUT, exist_ok=True)

log = lambda *a: print(*a, flush=True)
rep = {}


# ----------------------------------------------------------------------------- PDBbind
def build_pdbbind():
    rows = []
    for ln in open(f'{IDX}/pdbbind/INDEX_general_PL_data.2020'):
        if ln.startswith('#'):
            continue
        p = ln.split()
        if len(p) < 5:
            continue
        rows.append((p[0], float(p[3]), p[4]))
    d = pd.DataFrame(rows, columns=['PDB_code', 'pK', 'measurement'])
    d['meas_type'] = d['measurement'].str.extract(r'^(Ki|Kd|IC50)')[0]

    refined = set()
    for ln in open(f'{IDX}/pdbbind/INDEX_refined_data.2020'):
        if not ln.startswith('#') and len(ln.split()) >= 5:
            refined.add(ln.split()[0])
    d['refined'] = d['PDB_code'].isin(refined)
    d['pdb_id']  = d['PDB_code'].str.lower()
    d['source']  = 'pdbbind'
    d['unique_id'] = d['PDB_code']
    log(f'[PDBbind] 原始 {len(d)} 条 (refined {d.refined.sum()})，'
        f'类型 {d.meas_type.value_counts().to_dict()}')
    return d


# ----------------------------------------------------------------------------- BindingNet
def build_bindingnet():
    d = pd.read_csv(f'{IDX}/bindingnet/BindingNet_Uw_final_median.csv', sep='\t')
    d = d.rename(columns={'-logAffi': 'pK'})
    parts = d['unique_identify'].str.split('_')
    d['target_chembl'] = parts.str[0]
    d['pdb_id']        = parts.str[1].str.lower()
    d['ligand_chembl'] = parts.str[2]
    d['source']    = 'bindingnet'
    d['unique_id'] = d['unique_identify']
    log(f'[BindingNet] 原始 {len(d)} 条，唯一 PDB {d.pdb_id.nunique()}')
    return d


# ----------------------------------------------------------------------------- BindingDB-DCS
CENSORED = re.compile(r'[<>]')

def dcs_pk(field):
    """按规则把一个 'a;b;c' 浓度字段换算为 pK；返回 (pK, 原因)。"""
    vals = [t.strip() for t in str(field).split(';') if t.strip()]
    if any(CENSORED.search(v) for v in vals):
        return None, 'censored'
    try:
        nums = [float(v) for v in vals]
    except ValueError:
        return None, 'unparsable'
    if any(v < 0 for v in nums):
        return None, 'negative'
    with np.errstate(divide='ignore'):
        pk = float(np.median([9.0 - np.log10(v) for v in nums]))
    if not np.isfinite(pk):
        return None, 'non_finite_pK'
    return pk, None


def build_dcs():
    recs, drop = [], {'censored': 0, 'non_finite_pK': 0, 'negative': 0, 'unparsable': 0,
                      'no_affinity': 0, 'mol2_unreadable': 0, 'mol2_missing': 0}
    files = sorted(glob.glob(f'{IDX}/_bindingdb_series/*.csv'))
    log(f'[DCS] 系列 csv {len(files)} 个')
    for fp in files:
        base = os.path.basename(fp)[:-4]
        folder, pdb = base.split('__')
        d = pd.read_csv(fp)
        for _, r in d.iterrows():
            m = re.search(r'([^/]+\.mol2)$', str(r['Structure File']))
            if not m:
                drop['no_affinity'] += 1
                continue
            mol2 = m.group(1)

            # 任一列出现删失值即整行剔除（与参考实现一致）
            if any(CENSORED.search(str(r[c])) for c in ('Ki(nM)', 'IC50(nM)', 'Kd(nM)')
                   if pd.notna(r[c])):
                drop['censored'] += 1
                continue

            pk, why, mt = None, None, None
            for col, name in (('Kd(nM)', 'Kd'), ('Ki(nM)', 'Ki'), ('IC50(nM)', 'IC50')):
                if pd.isna(r[col]):
                    continue
                pk, why = dcs_pk(r[col])
                mt = name
                break
            if mt is None:
                drop['no_affinity'] += 1
                continue
            if pk is None:
                drop[why] += 1
                continue

            path = os.path.join(STR, 'bindingdb_dcs', folder, mol2)
            if not os.path.exists(path):
                drop['mol2_missing'] += 1
                continue
            if Chem.MolFromMol2File(path) is None:
                drop['mol2_unreadable'] += 1
                continue

            recs.append((folder, pdb, mol2, f'{folder}_{mol2[:-5]}', pk, mt))

    d = pd.DataFrame(recs, columns=['folder', 'pdb', 'mol2_file', 'unique_id', 'pK', 'meas_type'])
    d['pdb_id'] = d['pdb'].str.lower()
    d['source'] = 'bindingdb_dcs'
    log(f'[DCS] 保留 {len(d)} 条，系列 {d.folder.nunique()} 个，剔除明细 {drop}')
    rep['dcs_drop'] = drop
    return d


# ----------------------------------------------------------------------------- 泄漏剔除
def main():
    ids = json.load(open(f'{META}/leakage_ids.json'))
    excl = set(ids['excluded_pdb_ids'])
    log(f'需剔除 PDB ID：OOD {len(ids["ood_pdb_ids"])} + FEP {len(ids["fep_pdb_ids"])} '
        f'→ 并集 {len(excl)}')

    out = {}
    for name, fn in (('pdbbind', build_pdbbind),
                     ('bindingnet', build_bindingnet),
                     ('bindingdb_dcs', build_dcs)):
        d = fn()
        n0 = len(d)
        hit = d['pdb_id'].isin(excl)
        d_clean = d[~hit].reset_index(drop=True)
        n_ood = d['pdb_id'].isin(set(ids['ood_pdb_ids'])).sum()
        n_fep = d['pdb_id'].isin(set(ids['fep_pdb_ids'])).sum()
        log(f'[{name}] 泄漏剔除 {n0} → {len(d_clean)} '
            f'(命中 {hit.sum()}；其中 OOD {n_ood}、FEP {n_fep})')
        rep[name] = {'raw': int(n0), 'after_leak': int(len(d_clean)),
                     'removed': int(hit.sum()), 'hit_ood': int(n_ood), 'hit_fep': int(n_fep),
                     'unique_pdb_raw': int(d.pdb_id.nunique()),
                     'unique_pdb_after': int(d_clean.pdb_id.nunique())}
        d.to_csv(f'{OUT}/{name}_raw.csv', index=False)
        d_clean.to_csv(f'{OUT}/{name}_clean.csv', index=False)
        out[name] = d_clean

    # 泄漏验证
    log('\n=== 泄漏验证 ===')
    ok = True
    for name, d in out.items():
        i_ood = set(d.pdb_id) & set(ids['ood_pdb_ids'])
        i_fep = set(d.pdb_id) & set(ids['fep_pdb_ids'])
        log(f'  {name}: ∩OOD={len(i_ood)} ∩FEP={len(i_fep)} '
            f'{"✅" if not (i_ood or i_fep) else "❌ " + str(sorted(i_ood | i_fep)[:5])}')
        ok &= not (i_ood or i_fep)

    base = pd.concat([out['pdbbind'][['source', 'unique_id', 'pdb_id', 'pK']],
                      out['bindingnet'][['source', 'unique_id', 'pdb_id', 'pK']],
                      out['bindingdb_dcs'][['source', 'unique_id', 'pdb_id', 'pK']]],
                     ignore_index=True)
    base.to_csv(f'{OUT}/baseline_clean.csv', index=False)
    log(f'\n[Baseline] 三源拼接（不去重）共 {len(base)} 条，唯一 PDB {base.pdb_id.nunique()}')
    log(f'  pK 范围 {base.pK.min():.2f} ~ {base.pK.max():.2f}，均值 {base.pK.mean():.3f}')
    log(f'  组成 {base.source.value_counts().to_dict()}')

    rep['baseline_total'] = int(len(base))
    rep['leak_check_pass'] = bool(ok)
    json.dump(rep, open(f'{OUT}/report.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\n报告 → {OUT}/report.json')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
