#!/usr/bin/env python
"""
补充任务 1：OOD 295 的靶点 UniProt 集合与三个训练源求交，报覆盖率。

口径说明：
  - OOD / PDBbind 无 UniProt 列，用 SIFTS pdb_chain_uniprot.tsv.gz 按 PDB ID 映射
    （一个 PDB 可对应多条链/多个 UniProt，全部纳入该 PDB 的 UniProt 集合）
  - HiQBind 用自带 'Protein UniProtID'
  - Gator 用自带 'UniProt (SwissProt) Primary ID of Target Chain'
  - 覆盖率 = |OOD ∩ 源| / |OOD|，即 OOD 靶点中有多少能在该源里找到同一 UniProt
"""
import gzip
import json
import pandas as pd

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E = f'{ROOT}/experiments'
IDX = f'{ROOT}/datasets/index'
OUT = f'{E}/09_uniprot_coverage'

import os
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(*a, flush=True)


def load_sifts():
    """PDB ID(4位小写) -> set(UniProt)"""
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
    """UniProt 归一化：去同工异构体后缀、大写去空白"""
    s = str(s).strip().upper()
    return s.split('-')[0] if s and s != 'NAN' else ''


def expand(series):
    """一个单元格可能有多个 UniProt（分号/逗号分隔）"""
    out = set()
    for v in series.dropna():
        for tok in str(v).replace(',', ';').split(';'):
            u = norm(tok)
            if u:
                out.add(u)
    return out


def main():
    log('[1/4] 载入 SIFTS 映射 …')
    sifts = load_sifts()
    log(f'  SIFTS 覆盖 {len(sifts)} 个 PDB ID')

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
    log(f'\n[2/4] OOD 295：{ood_mapped}/295 个 PDB 成功映射到 UniProt，'
        f'共 {len(ood_uni)} 个唯一 UniProt')

    # ---------- 三个训练源 ----------
    log('\n[3/4] 载入三个训练源的 UniProt 集合 …')

    pb = pd.read_csv(f'{E}/01_baseline_index/pdbbind_clean.csv')
    pb_uni = set()
    pb_mapped = 0
    for p in [str(x).strip().lower()[:4] for x in pb.pdb_id]:
        u = {norm(x) for x in sifts.get(p, set())}
        u.discard('')
        if u:
            pb_mapped += 1
            pb_uni |= u
    log(f'  PDBbind  {len(pb):>6d} 条 → {pb_mapped} 条可映射，{len(pb_uni)} 个唯一 UniProt')

    hq = pd.read_csv(f'{E}/02_hiqbind/hiqbind_dedup.csv')
    hq_uni = expand(hq['Protein UniProtID'])
    log(f'  HiQBind  {len(hq):>6d} 条 → {len(hq_uni)} 个唯一 UniProt（源自带字段）')

    ga = pd.read_csv(f'{E}/03_gator/gator_dedup.csv')
    ga_col = 'UniProt (SwissProt) Primary ID of Target Chain'
    ga_uni = expand(ga[ga_col])
    ga_f_uni = expand(ga.loc[ga.in_filtered_100, ga_col])
    log(f'  Gator    {len(ga):>6d} 条 → {len(ga_uni)} 个唯一 UniProt（源自带字段）')
    log(f'    其中筛选版(conf≥0.7) → {len(ga_f_uni)} 个唯一 UniProt')

    # ---------- 求交与覆盖率 ----------
    log('\n[4/4] OOD 靶点覆盖率 =  |OOD ∩ 源| / |OOD|')
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
        log(f'  {name:<20s} 源 {len(uni):>5d} 个 UniProt，交集 {len(inter):>3d}，'
            f'覆盖率 {cov*100:>5.1f}%')

    # 三源并集（baseline 中含 PDBbind，配置2 加 HiQBind，配置3-8 加 Gator）
    for name, uni in [('PDBbind ∪ HiQBind', pb_uni | hq_uni),
                      ('PDBbind ∪ HiQBind ∪ Gator', pb_uni | hq_uni | ga_uni)]:
        inter = ood_uni & uni
        cov = len(inter) / len(ood_uni)
        rows.append({'source': name, 'n_records': -1,
                     'n_uniprot_source': len(uni),
                     'n_uniprot_ood': len(ood_uni),
                     'n_intersect': len(inter), 'coverage': cov})
        log(f'  {name:<20s} 源 {len(uni):>5d} 个 UniProt，交集 {len(inter):>3d}，'
            f'覆盖率 {cov*100:>5.1f}%')

    # ---------- 按 PDB 粒度：多少个 OOD 复合物的靶点被覆盖 ----------
    log('\n[附] 按 OOD 复合物（PDB）粒度统计被覆盖的个数：')
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
    log(f'\n产物 → {OUT}/{{uniprot_coverage.csv, uniprot_coverage_by_pdb.csv, coverage.json}}')


if __name__ == '__main__':
    main()
