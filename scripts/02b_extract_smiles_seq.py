#!/usr/bin/env python
"""
Stage 2b: prepare the inputs for CleanSplit three-criterion deduplication —
ligand SMILES and protein sequences.

The three criteria need:
  1. protein structural similarity TM-score > 0.8
       → protein structure files (TM-align) + sequences (MMseqs2 blocking)
  2. ligand ECFP4 Tanimoto > 0.9
       → ligand SMILES
  3. |ΔpK| <= 1.0
       → already available

The PDBbind index has no SMILES, so they are read from
structure/pdbbind/sdf/{pdb}_ligand.sdf; the HiQBind metadata already include
SMILES, so only sequences are added.

Sequences are used for MMseqs2 candidate blocking (a loose pre-filter), with
the **longest chain** of each complex as its representative; the final
TM-score decision is still made by TM-align on the full structures, so the
approximate nature of the blocking does not weaken the strictness of the
criterion.
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
    """Parse a PDB file, extract each chain's sequence from CA atoms, and
    return the sequence of the longest chain."""
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
                    rid = (ch, ln[22:27])          # chain + resSeq + iCode
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
        # If sanitization fails, fall back to reading without sanitization
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
    log(f'parallel processes {nproc}')

    # ---------- PDBbind ----------
    pb = pd.read_csv(f'{ROOT}/experiments/01_baseline_index/pdbbind_clean.csv')
    S = f'{ROOT}/datasets/structure/pdbbind'
    args = []
    for pdb, ref in zip(pb.pdb_id, pb.refined):
        folder = 'refined-set' if ref else 'general-set'
        args.append((pdb, f'{S}/sdf/{pdb}_ligand.sdf',
                     f'{S}/{folder}/{pdb}/{pdb}_protein.pdb'))
    log(f'[PDBbind] extracting SMILES + sequences for {len(args)} entries …')
    with Pool(nproc) as p:
        res = p.map(job_pdbbind, args, chunksize=64)
    r = pd.DataFrame(res, columns=['pdb_id', 'smiles', 'seq'])
    log(f'  SMILES failed {r.smiles.isna().sum()}, '
        f'empty sequences {(r.seq == "").sum()}')
    log(f'  median sequence length '
        f'{r[r.seq != ""].seq.str.len().median():.0f}')
    r.to_csv(f'{OUT}/pdbbind_smiles_seq.csv', index=False)

    # ---------- HiQBind ----------
    hq = pd.read_csv(f'{ROOT}/experiments/02_hiqbind/hiqbind_clean.csv')
    log(f'[HiQBind] extracting {len(hq)} sequences (SMILES already '
        f'available) …')
    with Pool(nproc) as p:
        res = p.map(job_seq, list(zip(hq.unique_id, hq.protein_pdb)), chunksize=64)
    h = pd.DataFrame(res, columns=['unique_id', 'seq'])
    log(f'  empty sequences {(h.seq == "").sum()}')
    log(f'  median sequence length '
        f'{h[h.seq != ""].seq.str.len().median():.0f}')
    h.to_csv(f'{OUT}/hiqbind_seq.csv', index=False)

    # ---------- Write FASTA for MMseqs2 ----------
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
    log(f'\nOutputs → {OUT}/{{pdbbind_smiles_seq.csv, hiqbind_seq.csv, '
        f'pool_*.fasta}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
