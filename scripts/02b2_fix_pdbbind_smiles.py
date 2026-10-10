#!/usr/bin/env python
"""
Stage 2b patch: fix PDBbind-side ligand SMILES that cannot be read back.

When 02b extracted SMILES from SDF, molecules that failed sanitization were
re-read with sanitize=False, producing 406 chemically invalid SMILES (e.g. C
with 5 H, Cl-Cl bonds) that RDKit cannot read back, so their ECFP4
fingerprints cannot be computed either.

The direction of the effect matters: a PDBbind-side ligand without a
fingerprint → the corresponding HiQBind samples cannot hit the ligand
criterion → they are wrongly counted as "net new". This **overestimates** the
net new count, the opposite of the conservative convention, so it must be
fixed.

Fix: re-read these molecules from {pdb}_ligand.mol2 instead (mol2 carries
explicit bond orders, and PDBbind's mol2 bond-order annotations are more
reliable than its SDF). This recovers 368 entries; the remaining 38 (mostly
metal clusters / unusual valences) cannot be sanitized in either format, so
they are set to empty and counted in the report as-is, without any
speculative repair.
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
    log(f'{len(d)} entries in total, {len(bad)} cannot be read back → '
        f're-reading from mol2')

    with Pool(min(64, os.cpu_count())) as p:
        fixed = dict(p.map(from_mol2, [(i, ref.get(i, False)) for i in bad],
                           chunksize=8))
    n_ok = sum(1 for v in fixed.values() if v)
    log(f'  recovered from mol2 {n_ok}, still failing {len(bad) - n_ok}')

    d['smiles_src'] = 'sdf'
    smi = d.smiles.tolist()
    src = d.smiles_src.tolist()
    idx = {i: k for k, i in enumerate(d.pdb_id)}
    for i in bad:
        k = idx[i]
        v = fixed.get(i)
        smi[k], src[k] = (v, 'mol2') if v else (None, 'fail')
    d['smiles'], d['smiles_src'] = smi, src

    log(f'  final valid SMILES {d.smiles.notna().sum()} / {len(d)}'
        f' (sources {d.smiles_src.value_counts().to_dict()})')
    d.to_csv(f'{META}/pdbbind_smiles_seq.csv', index=False)

    r = json.load(open(f'{META}/report_02b.json'))
    r.update({'smiles_unreadable_before_fix': len(bad),
              'smiles_recovered_from_mol2': int(n_ok),
              'smiles_final_missing': int(d.smiles.isna().sum())})
    json.dump(r, open(f'{META}/report_02b.json', 'w'), indent=1)
    log(f'\nUpdated {META}/pdbbind_smiles_seq.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
