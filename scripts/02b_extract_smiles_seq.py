#!/usr/bin/env python
"""
阶段 2b：为 CleanSplit 三重去重准备输入——配体 SMILES 与蛋白序列。

三重判据（PLAN §3.2）需要：
  1. 蛋白结构相似度 TM-score > 0.8   → 需蛋白结构文件（TM-align）+ 序列（MMseqs2 分块）
  2. 配体 ECFP4 Tanimoto > 0.9       → 需配体 SMILES
  3. |ΔpK| <= 1.0                    → 已有

PDBbind 索引不含 SMILES，从 structure/pdbbind/sdf/{pdb}_ligand.sdf 读取；
HiQBind 元数据自带 SMILES，只补序列。

序列用于 MMseqs2 候选分块（宽松预筛），取每个复合物**最长链**作为代表；
最终 TM-score 判定仍在完整结构上由 TM-align 完成，故分块的近似性不影响判据严格性。
"""
import os, sys, json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{ROOT}/experiments/00_meta'
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O'}


def longest_chain_seq(pdb_path):
    """解析 PDB，按 CA 原子提取各链序列，返回最长链序列。"""
    try:
        chains = {}
        seen = set()
        with open(pdb_path, 'r', errors='ignore') as f:
            for ln in f:
                if ln.startswith('ENDMDL'):
                    break
                if ln.startswith('ATOM') and ln[12:16].strip() == 'CA':
                    res = ln[17:20].strip().upper()
                    if res not in AA3:
                        continue
                    ch = ln[21]
                    rid = (ch, ln[22:27])          # 链 + resSeq + iCode
                    if rid in seen:
                        continue
                    seen.add(rid)
                    chains.setdefault(ch, []).append(AA3[res])
        if not chains:
            return ''
        return ''.join(max(chains.values(), key=len))
    except Exception:
        return ''


def smiles_from_sdf(p):
    try:
        for m in Chem.SDMolSupplier(p, sanitize=True):
            if m is not None:
                return Chem.MolToSmiles(m)
        # sanitize 失败时退回不消毒读取
        for m in Chem.SDMolSupplier(p, sanitize=False):
            if m is not None:
                try:
                    Chem.SanitizeMol(m)
                except Exception:
                    pass
                return Chem.MolToSmiles(m)
    except Exception:
        pass
    return None


def job_pdbbind(a):
    pdb, sdf, prot = a
    return pdb, smiles_from_sdf(sdf), longest_chain_seq(prot)


def job_seq(a):
    uid, prot = a
    return uid, longest_chain_seq(prot)


def main():
    nproc = min(64, os.cpu_count())
    log(f'并行进程 {nproc}')

    # ---------- PDBbind ----------
    pb = pd.read_csv(f'{ROOT}/experiments/01_baseline_index/pdbbind_clean.csv')
    S = f'{ROOT}/datasets/structure/pdbbind'
    args = []
    for pdb, ref in zip(pb.pdb_id, pb.refined):
        folder = 'refined-set' if ref else 'general-set'
        args.append((pdb, f'{S}/sdf/{pdb}_ligand.sdf',
                     f'{S}/{folder}/{pdb}/{pdb}_protein.pdb'))
    log(f'[PDBbind] 提取 {len(args)} 条 SMILES + 序列 …')
    with Pool(nproc) as p:
        res = p.map(job_pdbbind, args, chunksize=64)
    r = pd.DataFrame(res, columns=['pdb_id', 'smiles', 'seq'])
    log(f'  SMILES 失败 {r.smiles.isna().sum()}，序列为空 {(r.seq == "").sum()}')
    log(f'  序列长度 中位 {r[r.seq != ""].seq.str.len().median():.0f}')
    r.to_csv(f'{OUT}/pdbbind_smiles_seq.csv', index=False)

    # ---------- HiQBind ----------
    hq = pd.read_csv(f'{ROOT}/experiments/02_hiqbind/hiqbind_clean.csv')
    log(f'[HiQBind] 提取 {len(hq)} 条序列（SMILES 已有）…')
    with Pool(nproc) as p:
        res = p.map(job_seq, list(zip(hq.unique_id, hq.protein_pdb)), chunksize=64)
    h = pd.DataFrame(res, columns=['unique_id', 'seq'])
    log(f'  序列为空 {(h.seq == "").sum()}')
    log(f'  序列长度 中位 {h[h.seq != ""].seq.str.len().median():.0f}')
    h.to_csv(f'{OUT}/hiqbind_seq.csv', index=False)

    # ---------- 写 FASTA 供 MMseqs2 ----------
    with open(f'{OUT}/pool_pdbbind.fasta', 'w') as f:
        for i, s in zip(r.pdb_id, r.seq):
            if s:
                f.write(f'>PB|{i}\n{s}\n')
    with open(f'{OUT}/pool_hiqbind.fasta', 'w') as f:
        for i, s in zip(h.unique_id, h.seq):
            if s:
                f.write(f'>HQ|{i}\n{s}\n')

    json.dump({'pdbbind_n': int(len(r)),
               'pdbbind_smiles_fail': int(r.smiles.isna().sum()),
               'pdbbind_seq_empty': int((r.seq == '').sum()),
               'hiqbind_n': int(len(h)),
               'hiqbind_seq_empty': int((h.seq == '').sum())},
              open(f'{OUT}/report_02b.json', 'w'), indent=1)
    log(f'\n产物 → {OUT}/{{pdbbind_smiles_seq.csv, hiqbind_seq.csv, pool_*.fasta}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
