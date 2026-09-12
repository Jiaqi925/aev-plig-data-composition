#!/usr/bin/env python
"""
Stage 3a: build the GatorAffinity-DB index (pK + structure pairing +
quality score + sequences).

  - pK is taken only from the column pointed to by `prediction_type`,
    and only Ki / Kd are accepted (IC50/EC50 are not thermodynamic
    binding constants: they depend on enzyme concentration and substrate
    conditions, and are therefore not comparable with the pK convention
    used by the other data sources)
  - Quality filter: Boltz-1 confidence_score >= 0.7, while keeping both
    the filtered and the unfiltered version
  - Intersected with the structure directories actually downloaded
    (expected 65,031, actually 65,001)
  - Internal deduplication: BindingDB MonomerID + canonical SMILES
  - Evaluation-set separation does not go through PDB IDs (Gator has no
    real PDB IDs); it uses sequence identity < 30% + Tanimoto < 0.9
    instead, which is handled by script 03b. This script only prepares
    SMILES / sequences / FASTA

The index is 943MB, so columns are read on demand to limit memory use.
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
    """nM -> p value; returns NaN for non-positive or non-finite input."""
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
    log(f'Reading index (columns on demand)...')
    d = pd.read_csv(IDX, usecols=USECOLS, low_memory=False)
    rep['raw'] = int(len(d))
    log(f'[Gator] raw {len(d)} rows (expected 463,867)')

    # --- pK: Ki / Kd only, reading the column given by prediction_type ---
    pt = d['prediction_type'].astype(str).str.strip().str.upper()
    log(f'  prediction_type distribution {pt.value_counts().to_dict()}')
    ki, kd = pk_from_nm(d['Ki (nM)']), pk_from_nm(d['Kd (nM)'])
    d['pK'] = np.where(pt == 'KI', ki, np.where(pt == 'KD', kd, np.nan))
    d['meas_type'] = pt
    keep = pt.isin(['KI', 'KD']) & np.isfinite(d['pK'])
    log(f'  keeping only Ki/Kd with finite pK: {len(d)} -> {keep.sum()}')
    rep['after_pk'] = int(keep.sum())
    d = d[keep].copy()

    # --- Structure pairing (intersect with the downloaded directories) ---
    # In the index GatorAffinity_ID is a plain integer, while the
    # structure directories are named GID{id}
    have = set(os.listdir(STR))
    rep['structure_dirs'] = len(have)
    log(f'  structure directories on disk {len(have)} (expected 65,031)')
    d['gid'] = 'GID' + d['GatorAffinity_ID'].astype(int).astype(str)
    d = d[d.gid.isin(have)].copy()
    log(f'  intersected with structure directories -> {len(d)}')

    g = d['gid']
    d['ligand_sdf'] = [f'{STR}/{i}/ligand/{i}_ligand_0_fixed.sdf' for i in g]
    d['protein_pdb'] = [f'{STR}/{i}/protein/{i}_protein_0_fixed.pdb' for i in g]
    ok = [os.path.exists(a) and os.path.exists(b)
          for a, b in zip(d.ligand_sdf, d.protein_pdb)]
    log(f'  complete structure files {sum(ok)} / {len(d)}'
        f' (missing {len(d) - sum(ok)})')
    rep['missing_structure'] = int(len(d) - sum(ok))
    d = d[ok].copy()

    # --- Internal deduplication: BindingDB MonomerID + canonical SMILES ---
    log('  canonicalizing SMILES ...')
    d['smiles_canon'] = [canon(s) for s in d['Ligand SMILES']]
    bad = d.smiles_canon.isna().sum()
    log(f'  unparsable SMILES {bad} rows (dropped)')
    rep['smiles_unparsable'] = int(bad)
    d = d[d.smiles_canon.notna()].copy()

    n_before = len(d)
    # Deduplication key = target + ligand.
    # Note: the initial design specified "BindingDB ID + ligand SMILES",
    # but measurement shows that BindingDB MonomerID is a **ligand**
    # identifier (158,836 MonomerIDs map to 158,938 SMILES, nearly
    # one-to-one), so deduplication on MonomerID+SMILES amounts to
    # "deduplicating by ligand only", which would merge records of the
    # same ligand bound to different targets -- yet 36.2% of the ligands
    # bind multiple targets (up to 384), and those are mutually
    # independent, valid samples.
    # We therefore deduplicate on "target UniProt + ligand SMILES", which
    # is what "duplicate records of the same protein-ligand pair" was
    # meant to express.
    # For the target identifier, prefer the UniProt Primary ID and fall
    # back to Target Name when it is missing.
    uni = d['UniProt (SwissProt) Primary ID of Target Chain'].astype(str).str.strip()
    tgt = uni.where(uni.notna() & (uni != '') & (uni.str.lower() != 'nan'),
                    d['Target Name'].astype(str).str.strip())
    d['target_key'] = tgt
    d['dup_key'] = tgt + '|' + d['smiles_canon']
    conf = pd.to_numeric(d['confidence_score'], errors='coerce')
    d['conf'] = conf
    # Several measurements of the same protein-ligand pair: take the
    # median pK, and keep the structure with the highest confidence
    d['pK_median'] = d.groupby('dup_key')['pK'].transform('median')
    d = (d.sort_values('conf', ascending=False)
           .drop_duplicates('dup_key', keep='first').copy())
    d['pK'] = d['pK_median']
    log(f'  internal deduplication (UniProt+SMILES, median pK, highest'
        f' confidence kept): {n_before} -> {len(d)}')
    rep['after_internal_dedup'] = int(len(d))

    # --- Quality score ---
    d['pass_conf'] = d['conf'] >= CONF_THR
    log(f'  confidence >= {CONF_THR}: {d.pass_conf.sum()} / {len(d)}'
        f' (missing {d.conf.isna().sum()})')
    log(f'  confidence quantiles'
        f' {d.conf.quantile([.1, .25, .5, .75, .9]).round(3).to_dict()}')
    rep['pass_conf'] = int(d.pass_conf.sum())

    # --- Sequences (for MMseqs2 and the 03b leakage separation) ---
    d['seq'] = d['BindingDB Target Chain Sequence'].astype(str).str.strip().str.upper()
    d.loc[~d.seq.str.fullmatch(r'[A-Z]+', na=False), 'seq'] = ''
    log(f'  empty sequences {(d.seq == "").sum()}; median length '
        f'{d[d.seq != ""].seq.str.len().median():.0f}')
    rep['seq_empty'] = int((d.seq == '').sum())

    d['scaffold'] = [scaffold(s) for s in d.smiles_canon]
    d['source'] = 'gatoraffinity'
    d['unique_id'] = d['gid']

    log(f'\n[Gator] {len(d)} records after cleaning')
    log(f'  unique ligands   {d.smiles_canon.nunique()}')
    log(f'  unique UniProt   '
        f'{d["UniProt (SwissProt) Primary ID of Target Chain"].nunique()}')
    log(f'  unique Bemis-Murcko scaffolds {d.scaffold.nunique()}'
        f' (failed {d.scaffold.isna().sum()})')
    log(f'  pK range {d.pK.min():.2f} ~ {d.pK.max():.2f},'
        f' mean {d.pK.mean():.3f}')
    log(f'  measurement types {d.meas_type.value_counts().to_dict()}')

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
    log(f'\noutputs -> {OUT}/'
        f'{{gator_clean.csv, pool_gator.fasta, report_03a.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
