#!/usr/bin/env python
"""
阶段 2b 补丁：修复 PDBbind 侧无法回读的配体 SMILES。

02b 从 SDF 提取 SMILES 时，对 sanitize 失败的分子退回 sanitize=False 读取，
产生了 406 条化学上非法的 SMILES（如 C 带 5 个 H、Cl-Cl 键），RDKit 无法回读，
后续 ECFP4 指纹也就算不出来。

影响方向不容忽视：PDBbind 侧配体缺指纹 → 对应的 HiQBind 样本无法命中配体判据
→ 被误判为「净新增」。这会**高估**净新增数，与保守口径相反，故必须修。

修法：对这批分子改从 {pdb}_ligand.mol2 读取（mol2 带显式键级，PDBbind 的
mol2 键级标注比 SDF 更可靠）。可救回 368 条；余 38 条（多为金属簇/异常价态）
两种格式都无法消毒，如实置空并计数上报，不做任何猜测性修补。
"""
import os, sys, json
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = f'{ROOT}/experiments/00_meta'
S = f'{ROOT}/datasets/structure/pdbbind'
log = lambda *a: print(*a, flush=True)


def readable(s):
    return isinstance(s, str) and s and Chem.MolFromSmiles(s) is not None


def from_mol2(a):
    pdb, refined = a
    f = 'refined-set' if refined else 'general-set'
    try:
        m = Chem.MolFromMol2File(f'{S}/{f}/{pdb}/{pdb}_ligand.mol2', sanitize=True)
        if m is not None:
            return pdb, Chem.MolToSmiles(m)
    except Exception:
        pass
    return pdb, None


def main():
    d = pd.read_csv(f'{META}/pdbbind_smiles_seq.csv')
    pb = pd.read_csv(f'{ROOT}/experiments/01_baseline_index/pdbbind_clean.csv')
    ref = dict(zip(pb.pdb_id, pb.refined))

    with Pool(min(64, os.cpu_count())) as p:
        ok = p.map(readable, d.smiles.tolist(), chunksize=256)
    bad = [i for i, o in zip(d.pdb_id, ok) if not o]
    log(f'总 {len(d)} 条，不可回读 {len(bad)} 条 → 改从 mol2 读取')

    with Pool(min(64, os.cpu_count())) as p:
        fixed = dict(p.map(from_mol2, [(i, ref.get(i, False)) for i in bad],
                           chunksize=8))
    n_ok = sum(1 for v in fixed.values() if v)
    log(f'  mol2 救回 {n_ok}，仍失败 {len(bad) - n_ok}')

    d['smiles_src'] = 'sdf'
    smi = d.smiles.tolist()
    src = d.smiles_src.tolist()
    idx = {i: k for k, i in enumerate(d.pdb_id)}
    for i in bad:
        k = idx[i]
        v = fixed.get(i)
        smi[k], src[k] = (v, 'mol2') if v else (None, 'fail')
    d['smiles'], d['smiles_src'] = smi, src

    log(f'  最终有效 SMILES {d.smiles.notna().sum()} / {len(d)}'
        f'（来源 {d.smiles_src.value_counts().to_dict()}）')
    d.to_csv(f'{META}/pdbbind_smiles_seq.csv', index=False)

    r = json.load(open(f'{META}/report_02b.json'))
    r.update({'smiles_unreadable_before_fix': len(bad),
              'smiles_recovered_from_mol2': int(n_ok),
              'smiles_final_missing': int(d.smiles.isna().sum())})
    json.dump(r, open(f'{META}/report_02b.json', 'w'), indent=1)
    log(f'\n已更新 {META}/pdbbind_smiles_seq.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
