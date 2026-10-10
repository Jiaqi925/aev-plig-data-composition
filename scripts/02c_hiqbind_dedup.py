#!/usr/bin/env python
"""
Stage 2c: three-criterion deduplication of HiQBind CleanSplit.

Deduplication target: the original design explicitly states "HiQBind: perform
deduplication (remove samples duplicated with PDBbind)", so the reference set
is PDBbind (19,118 entries after leakage removal); BindingNet /
BindingDB-DCS are not involved.
Direction: only the HiQBind side is pruned; PDBbind is left untouched
(config 2 = Baseline + all of HiQBind after deduplication).

Three criteria (a pair is a duplicate only if all hold):
  1. ligand ECFP4 Tanimoto > 0.9
  2. |ΔpK| <= 1.0
  3. protein TM-align TM-score > 0.8

Order of execution goes from cheapest to most expensive: ligand fingerprints
first (with popcount pruning), then pK, finally TM-align.
This does not change the criteria themselves — the three are conjunctive, so
any order yields the same set of duplicates.

TM-score normalization: TM-align reports two values, normalized by the length
of Chain_1 and of Chain_2 respectively; this script takes the larger of the
two. The larger value crosses 0.8 more easily, i.e. it leans towards "judge as
duplicate", which is a conservative (non-overestimating) convention for the
"net new complexes" count.
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


_HF = _PF = _BYPC = _PCS = None      # inherited via fork; avoids pickling


def match_block(rng):
    """For a batch of unique HiQBind ligands, find the unique PDBbind ligands
    with Tanimoto > threshold."""
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
    """Return the TM-score as max(normalized by chain 1, normalized by
    chain 2)."""
    p, q = a
    try:
        r = subprocess.run(['TMalign', p, q], capture_output=True,
                           text=True, timeout=600)
        v = [float(x) for x in _TM_RE.findall(r.stdout)]
        return max(v) if v else -1.0
    except Exception:
        return -1.0


def main():
    log(f'processes {NPROC}; thresholds Tanimoto>{T_TANI}, '
        f'|dpK|<={T_DPK}, TM>{T_TM}')

    hq = pd.read_csv(f'{OUT}/hiqbind_clean.csv')
    pb = pd.read_csv(f'{ROOT}/experiments/01_baseline_index/pdbbind_clean.csv')
    ps = pd.read_csv(f'{META}/pdbbind_smiles_seq.csv')
    pb = pb.merge(ps[['pdb_id', 'smiles']], on='pdb_id', how='left')
    pb['prot'] = [f'{ROOT}/datasets/structure/pdbbind/'
                  f'{"refined-set" if r else "general-set"}/{i}/{i}_protein.pdb'
                  for i, r in zip(pb.pdb_id, pb.refined)]
    log(f'HiQBind {len(hq)} entries; PDBbind {len(pb)} entries'
        f' (SMILES missing {pb.smiles.isna().sum()}; these rows cannot take'
        f' part in the ligand criterion)')

    # ---------- 1. Canonical SMILES ----------
    with Pool(NPROC) as p:
        hq['csmi'] = p.map(canon, hq['Ligand SMILES'].astype(str), chunksize=256)
        pbs = pb.smiles.fillna('').astype(str).tolist()
        pb['csmi'] = p.map(canon, pbs, chunksize=256)
    log(f'  canonicalization failed: HiQBind {hq.csmi.isna().sum()}, '
        f'PDBbind {pb.csmi.isna().sum()}')

    hu = sorted(set(hq.csmi.dropna()))
    pu = sorted(set(pb.csmi.dropna()))
    log(f'  unique ligands HiQBind {len(hu)}, PDBbind {len(pu)}')

    with Pool(NPROC) as p:
        hf = p.map(fp_of, hu, chunksize=128)
        pf = p.map(fp_of, pu, chunksize=128)

    # ---------- 2. Ligand criterion: popcount pruning + Tanimoto ----------
    # T=c/(a+b-c)>0.9 and c<=min(a,b)  =>  min(a,b) > 0.9*max(a,b), so only
    # nearby popcount buckets need to be compared
    global _HF, _PF, _BYPC, _PCS
    bypc = defaultdict(list)
    for i, f in enumerate(pf):
        if f is not None:
            bypc[f.GetNumOnBits()].append(i)
    _HF, _PF, _BYPC, _PCS = hf, pf, bypc, sorted(bypc)

    blocks = [range(k, min(k + 200, len(hu))) for k in range(0, len(hu), 200)]
    log(f'[ligand] {len(hu)} × {len(pu)} unique ligands, '
        f'{len(blocks)} blocks …')
    with Pool(NPROC) as p:
        lig = [x for b in p.map(match_block, blocks) for x in b]
    log(f'  ligand pairs with Tanimoto>{T_TANI}: {len(lig)}')

    # ---------- 3. Expand to sample pairs + pK criterion ----------
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
    log(f'[pK] sample pairs meeting both Tanimoto and '
        f'|dpK|<={T_DPK}: {len(pairs)}')

    # ---------- 4. TM-align (unique structure pairs of survivors) ----------
    hprot, pprot = hq.protein_pdb.values, pb.prot.values
    uniq = sorted({(hprot[x], pprot[y]) for x, y, _ in pairs})
    log(f'[TM-align] {len(uniq)} unique structure pairs, aligning …')
    with Pool(NPROC) as p:
        tms = p.map(tmscore, uniq, chunksize=4)
    tm = dict(zip(uniq, tms))
    nfail = sum(1 for v in tms if v < 0)
    log(f'  failed {nfail}; structure pairs with '
        f'TM>{T_TM}: {sum(1 for v in tms if v > T_TM)}')

    # ---------- 5. Determine duplicates ----------
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
    log(f'\n[verdict] HiQBind entries hit by all three criteria '
        f'{int(dup.sum())} → kept {int((~dup).sum())}')

    keep = hq[~dup].copy().reset_index(drop=True)
    keep.to_csv(f'{OUT}/hiqbind_dedup.csv', index=False)

    # ---------- 6. The three required statistics ----------
    pb_pdb = set(pb.pdb_id)
    log(f'  net new complexes          {len(keep)}')
    log(f'  of which PDB ID not in PDBbind '
        f'{int((~keep.pdb_id.isin(pb_pdb)).sum())}')
    log(f'  unique UniProt targets     {keep["Protein UniProtID"].nunique()}')
    log(f'  unique BM scaffolds        {keep.scaffold.nunique()}')
    log(f'  pK range {keep.pK.min():.2f} ~ {keep.pK.max():.2f}, '
        f'mean {keep.pK.mean():.3f}')

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
    log(f'\nOutputs → {OUT}/{{hiqbind_dedup.csv, hiqbind_dup_pairs.csv, '
        f'report_02c.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
