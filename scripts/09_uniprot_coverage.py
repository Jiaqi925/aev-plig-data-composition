#!/usr/bin/env python
"""
Supplementary task 1: intersect the target UniProt set of OOD 295 with each of
the three training sources and report coverage.

Conventions:
  - OOD / PDBbind have no UniProt column; they are mapped by PDB ID using
    SIFTS pdb_chain_uniprot.tsv.gz
    (one PDB may map to several chains / UniProt IDs; all are included in that
    PDB's UniProt set)
  - HiQBind uses its own 'Protein UniProtID'
  - Gator uses its own 'UniProt (SwissProt) Primary ID of Target Chain'
  - coverage = |OOD ∩ source| / |OOD|, i.e. the fraction of OOD targets whose
    UniProt can also be found in that source
"""
import gzip
import json
import os

import pandas as pd

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
IDX = f'{ROOT}/datasets/index'
OUT = f'{E}/09_uniprot_coverage'

os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)


def load_sifts():
    """PDB ID (4 lowercase characters) -> set(UniProt)"""
    m = {}
    with gzip.open(f'{IDX}/_mappings/pdb_chain_uniprot.tsv.gz', 'rt') as f:
        for line in f:
            if line.startswith('#') or line.startswith('PDB\t'):
                continue
            p = line.split('\t')
            if len(p) < 3:
                continue
            m.setdefault(p[0].strip().lower(), set()).add(p[2].strip())
    return m


def norm(s):
    """Normalize a UniProt ID: strip the isoform suffix, uppercase, remove
    whitespace"""
    s = str(s).strip().upper()
    return s.split('-')[0] if s and s != 'NAN' else ''


def expand(series):
    """A single cell may hold several UniProt IDs (separated by semicolons or
    commas)"""
    out = set()
    for v in series.dropna():
        for tok in str(v).replace(',', ';').split(';'):
            u = norm(tok)
            if u:
                out.add(u)
    return out


def main():
    log('[1/4] loading SIFTS mapping …')
    sifts = load_sifts()
    log(f'  SIFTS covers {len(sifts)} PDB IDs')

    # ---------- OOD 295 ----------
    ood = pd.read_csv(f'{IDX}/_testsets/index_oodtest.csv')
    ood = ood[ood.split == 'test'].reset_index(drop=True)
    ood_pdb = [str(x).strip().lower()[:4] for x in ood.PDB_code]
    assert len(ood_pdb) == 295, len(ood_pdb)
    ood_uni = set()
    ood_mapped = 0
    for p in ood_pdb:
        u = {norm(x) for x in sifts.get(p, set())}
        u.discard('')
        if u:
            ood_mapped += 1
            ood_uni |= u
    log(f'\n[2/4] OOD 295: {ood_mapped}/295 PDBs mapped to UniProt, '
        f'{len(ood_uni)} unique UniProt IDs in total')

    # ---------- Three training sources ----------
    log('\n[3/4] loading the UniProt sets of the three training sources …')

    pb = pd.read_csv(f'{E}/01_baseline_index/pdbbind_clean.csv')
    pb_uni = set()
    pb_mapped = 0
    for p in [str(x).strip().lower()[:4] for x in pb.pdb_id]:
        u = {norm(x) for x in sifts.get(p, set())}
        u.discard('')
        if u:
            pb_mapped += 1
            pb_uni |= u
    log(f'  PDBbind  {len(pb):>6d} entries → {pb_mapped} mappable, '
        f'{len(pb_uni)} unique UniProt IDs')

    hq = pd.read_csv(f'{E}/02_hiqbind/hiqbind_dedup.csv')
    hq_uni = expand(hq['Protein UniProtID'])
    log(f'  HiQBind  {len(hq):>6d} entries → {len(hq_uni)} unique UniProt IDs '
        f'(source-provided field)')

    ga = pd.read_csv(f'{E}/03_gator/gator_dedup.csv')
    ga_col = 'UniProt (SwissProt) Primary ID of Target Chain'
    ga_uni = expand(ga[ga_col])
    ga_f_uni = expand(ga.loc[ga.in_filtered_100, ga_col])
    log(f'  Gator    {len(ga):>6d} entries → {len(ga_uni)} unique UniProt IDs '
        f'(source-provided field)')
    log(f'    of which filtered version (conf≥0.7) → {len(ga_f_uni)} unique '
        f'UniProt IDs')

    # ---------- Intersection and coverage ----------
    log('\n[4/4] OOD target coverage =  |OOD ∩ source| / |OOD|')
    rows = []
    srcs = [
        ('PDBbind', len(pb), pb_uni),
        ('HiQBind', len(hq), hq_uni),
        ('Gator(未筛选全量)', len(ga), ga_uni),
        ('Gator(筛选版100%)', int(ga.in_filtered_100.sum()), ga_f_uni),
    ]
    for name, n_rec, uni in srcs:
        inter = ood_uni & uni
        cov = len(inter) / len(ood_uni) if ood_uni else 0.0
        rows.append({'source': name, 'n_records': n_rec,
                     'n_uniprot_source': len(uni),
                     'n_uniprot_ood': len(ood_uni),
                     'n_intersect': len(inter),
                     'coverage': cov})
        log(f'  {name:<20s} source {len(uni):>5d} UniProt IDs, '
            f'intersection {len(inter):>3d}, coverage {cov*100:>5.1f}%')

    # Unions of sources (baseline contains PDBbind, config 2 adds HiQBind,
    # configs 3-8 add Gator)
    for name, uni in [('PDBbind ∪ HiQBind', pb_uni | hq_uni),
                      ('PDBbind ∪ HiQBind ∪ Gator', pb_uni | hq_uni | ga_uni)]:
        inter = ood_uni & uni
        cov = len(inter) / len(ood_uni)
        rows.append({'source': name, 'n_records': -1,
                     'n_uniprot_source': len(uni),
                     'n_uniprot_ood': len(ood_uni),
                     'n_intersect': len(inter), 'coverage': cov})
        log(f'  {name:<20s} source {len(uni):>5d} UniProt IDs, '
            f'intersection {len(inter):>3d}, coverage {cov*100:>5.1f}%')

    # ---------- Per PDB: how many OOD complexes have a covered target --------
    log('\n[extra] covered OOD complexes, counted per complex (PDB):')
    pdb_rows = []
    for name, _, uni in srcs:
        hit = 0
        for p in ood_pdb:
            u = {norm(x) for x in sifts.get(p, set())}
            u.discard('')
            if u & uni:
                hit += 1
        pdb_rows.append({'source': name, 'n_ood_pdb': 295, 'n_hit': hit,
                         'coverage': hit / 295})
        log(f'  {name:<20s} {hit:>3d}/295 = {hit/295*100:>5.1f}%')

    pd.DataFrame(rows).to_csv(f'{OUT}/uniprot_coverage.csv', index=False)
    pd.DataFrame(pdb_rows).to_csv(f'{OUT}/uniprot_coverage_by_pdb.csv', index=False)
    json.dump({'ood_n_pdb': 295, 'ood_n_pdb_mapped': ood_mapped,
               'ood_n_uniprot': len(ood_uni),
               'by_uniprot': rows, 'by_pdb': pdb_rows,
               'sifts_version': 'pdb_chain_uniprot.tsv.gz (PDB 31.26 / UniProt 2026.03)'},
              open(f'{OUT}/coverage.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\nOutputs → {OUT}/{{uniprot_coverage.csv, '
        f'uniprot_coverage_by_pdb.csv, coverage.json}}')


if __name__ == '__main__':
    main()
