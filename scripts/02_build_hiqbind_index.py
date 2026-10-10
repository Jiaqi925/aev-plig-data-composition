#!/usr/bin/env python
"""
Stage 2a: build the HiQBind index (pK + structure paths + evaluation-set
leakage removal).

Requirements followed:
  - HiQBind raw size: expected 32,275 (checked = sm 31,572 + poly 703)
  - HiQBind deduplication: CleanSplit three-criterion standard; remove samples
    that overlap with OOD/FEP at the PDB ID level
  - Report the number of net new complexes, unique UniProt targets, and ligand
    Bemis-Murcko scaffolds

This script only does "parsing + pK + censored-value filtering + structure
pairing + evaluation-set removal + statistics"; the CleanSplit three-criterion
deduplication (which needs TM-align) is handled by the 02c script.

pK rules:
  - pK = -`Log Binding Affinity` (that column is log10(M); verified to match
    a recomputation from value+unit, max difference 3.8e-13)
  - Only exact measurements with `Binding Affinity Sign` == '=' are kept;
    censored or approximate values with ~ / > / >= / < / <= are removed. This
    matches the convention reverse-engineered from the reference
    implementation on DCS (rows with censored values are dropped entirely).
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
    log(f'[HiQBind] raw {n0} rows (expected 32,275)')

    # Tag the sm / poly origin so the structure directory can be located
    sm = set(pd.read_csv(f'{IDX}/hiqbind_sm_metadata.csv')
             .apply(lambda r: (r['PDBID'], r['Ligand Name'], r['Ligand Chain'],
                               r['Ligand Residue Number']), axis=1))
    key = d.apply(lambda r: (r['PDBID'], r['Ligand Name'], r['Ligand Chain'],
                             r['Ligand Residue Number']), axis=1)
    d['subset'] = np.where(key.isin(sm), 'sm', 'poly')
    log(f'  subsets {d.subset.value_counts().to_dict()}')

    # --- pK ---
    d['pK'] = -d['Log Binding Affinity']
    d['meas_type'] = d['Binding Affinity Measurement'].str.upper()

    # --- Remove censored/approximate values ---
    sign = d['Binding Affinity Sign'].astype(str).str.strip()
    exact = sign == '='
    log(f'  Sign distribution {sign.value_counts().to_dict()}')
    log(f'  removed {(~exact).sum()} non-exact measurements '
        f'→ kept {exact.sum()}')
    d = d[exact].copy()

    # pK must be finite
    fin = np.isfinite(d['pK'])
    if (~fin).sum():
        log(f'  removed {(~fin).sum()} rows with non-finite pK')
        d = d[fin].copy()

    # --- Match structure directories ---
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
    log(f'  structure files complete {sum(has)} / {len(d)} '
        f'(missing {len(d) - sum(has)})')
    rep['missing_structure'] = int(len(d) - sum(has))
    d = d[has].copy()

    # --- Evaluation-set leakage removal (first 4 characters of PDB ID) ---
    ids = json.load(open(f'{META}/leakage_ids.json'))
    ood, fep = set(ids['ood_pdb_ids']), set(ids['fep_pdb_ids'])
    d['pdb_id'] = d['PDBID'].astype(str).str.lower().str[:4]
    hit_ood = d['pdb_id'].isin(ood)
    hit_fep = d['pdb_id'].isin(fep)
    hit = hit_ood | hit_fep
    log(f'  leakage removal {len(d)} → {(~hit).sum()}'
        f' (hits {hit.sum()}; OOD {hit_ood.sum()}, FEP {hit_fep.sum()})')
    rep['leak_removed'] = {'total': int(hit.sum()), 'ood': int(hit_ood.sum()),
                           'fep': int(hit_fep.sum())}
    d = d[~hit].copy().reset_index(drop=True)

    # --- Statistics (the three required items) ---
    d['unique_id'] = d['complex_dir']
    d['source'] = 'hiqbind'
    log('  computing Bemis-Murcko scaffolds …')
    d['scaffold'] = [scaffold(s) for s in d['Ligand SMILES']]

    log(f'\n[HiQBind] {len(d)} entries after cleaning')
    log(f'  unique PDB IDs         {d.pdb_id.nunique()}')
    log(f'  unique UniProt targets {d["Protein UniProtID"].nunique()}')
    log(f'  unique Bemis-Murcko    {d.scaffold.nunique()} '
        f'(parse failures {d.scaffold.isna().sum()})')
    log(f'  pK range {d.pK.min():.2f} ~ {d.pK.max():.2f}, '
        f'mean {d.pK.mean():.3f}')
    log(f'  measurement types {d.meas_type.value_counts().to_dict()}')
    res = pd.to_numeric(d['Resolution'], errors='coerce')
    log(f'  resolution median {res.median():.2f} Å, '
        f'{(res > 2.5).sum()} entries > 2.5 Å'
        f' (missing {res.isna().sum()})')

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
    log(f'\nOutputs → {OUT}/hiqbind_clean.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
