#!/usr/bin/env python
"""
Stage 1: build the Baseline three-source index (PDBbind + BindingNet +
BindingDB-DCS) and perform evaluation-set leakage removal.

Requirements followed:
  - Raw sizes: expected PDBbind 19,443 / BindingNet 69,816 / DCS 8,822
  - The three baseline sources are concatenated directly, without
    deduplication
  - PDB ID (first 4 characters) is the unified key; samples involved in
    OOD (295) and FEP (42 series) are removed

pK conversion rules (reverse-engineered from the AEV-PLIG reference
implementation's data/bindingdb_processed.csv and reproduced 100% on all
8,822 rows, hence adopted rather than defining a separate standard):
  1. Measurement-type priority Kd > Ki > IC50; take the highest-priority
     column that has a value
  2. That column may hold several semicolon-separated concentrations; each is
     converted to a p value and the median is taken
  3. p = 9 - log10(concentration in nM); p = +inf when the concentration is 0
  4. Rows with censored values containing an inequality sign (> or <) are
     removed
  5. A row is removed only if the median is non-finite. The median is
     naturally robust to the occasional 0 value (inf is only sorted to the
     end of the sequence and does not affect the median), which matches the
     reference implementation's behavior row by row.
The PDBbind and BindingNet indices already include a pK column, which is used
as-is without reconversion.
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
    log(f'[PDBbind] raw {len(d)} entries (refined {d.refined.sum()}), '
        f'types {d.meas_type.value_counts().to_dict()}')
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
    log(f'[BindingNet] raw {len(d)} entries, unique PDB {d.pdb_id.nunique()}')
    return d


# ----------------------------------------------------------------------------- BindingDB-DCS
CENSORED = re.compile(r'[<>]')

def dcs_pk(field):
    """Convert one 'a;b;c' concentration field to pK according to the rules;
    return (pK, reason)."""
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
    log(f'[DCS] {len(files)} series csv files')
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

            # A censored value in any column drops the whole row (same as the
            # reference implementation)
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
    log(f'[DCS] kept {len(d)} entries, {d.folder.nunique()} series, '
        f'removal breakdown {drop}')
    rep['dcs_drop'] = drop
    return d


# ----------------------------------------------------------------------------- Leakage removal
def main():
    ids = json.load(open(f'{META}/leakage_ids.json'))
    excl = set(ids['excluded_pdb_ids'])
    log(f'PDB IDs to remove: OOD {len(ids["ood_pdb_ids"])} '
        f'+ FEP {len(ids["fep_pdb_ids"])} → union {len(excl)}')

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
        log(f'[{name}] leakage removal {n0} → {len(d_clean)} '
            f'(hits {hit.sum()}; of which OOD {n_ood}, FEP {n_fep})')
        rep[name] = {'raw': int(n0), 'after_leak': int(len(d_clean)),
                     'removed': int(hit.sum()), 'hit_ood': int(n_ood), 'hit_fep': int(n_fep),
                     'unique_pdb_raw': int(d.pdb_id.nunique()),
                     'unique_pdb_after': int(d_clean.pdb_id.nunique())}
        d.to_csv(f'{OUT}/{name}_raw.csv', index=False)
        d_clean.to_csv(f'{OUT}/{name}_clean.csv', index=False)
        out[name] = d_clean

    # Leakage check
    log('\n=== Leakage check ===')
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
    log(f'\n[Baseline] three sources concatenated (no deduplication): '
        f'{len(base)} entries, unique PDB {base.pdb_id.nunique()}')
    log(f'  pK range {base.pK.min():.2f} ~ {base.pK.max():.2f}, '
        f'mean {base.pK.mean():.3f}')
    log(f'  composition {base.source.value_counts().to_dict()}')

    rep['baseline_total'] = int(len(base))
    rep['leak_check_pass'] = bool(ok)
    json.dump(rep, open(f'{OUT}/report.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\nReport → {OUT}/report.json')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
