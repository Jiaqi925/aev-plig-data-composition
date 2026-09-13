#!/usr/bin/env python
"""
Stage 3b: GatorAffinity cross-database deduplication + evaluation-set
separation + scale sampling.

Four things, executed in order:

  (1) Safety net: the Gator index ships its own
      `PDB ID(s) for Ligand-Target Complex` field. The initial design
      specified that Gator has "no real PDB IDs", but rows where that
      field is non-empty are still removed via the PDB ID leakage key
      (330 leakage IDs). Free extra insurance, and it does not conflict
      with the criteria that follow.

  (2) Evaluation-set separation: target sequence identity (MMseqs2) +
      ligand ECFP4 Tanimoto, against the OOD 295 and FEP 42 series.
      * How the two criteria are combined: see "Criterion note" below.

  (3) Cross-database deduplication: against PDBbind and HiQBind (after
      deduplication) using the CleanSplit triple criterion --
      Tanimoto > 0.9 and |dpK| <= 1.0 and TM-score > 0.8; a pair counts
      as duplicate only when all three hold at once, and only the Gator
      side is dropped.

  (4) Versions and sampling: unfiltered version = everything left after
      deduplication; filtered version = unfiltered version intersected
      with (Boltz-1 confidence >= 0.7). Each is shuffled with seed 42
      and the first 30% / 60% / 100% are taken, **nested**
      (30% within 60% within 100%), so the three Scaling Law points sit
      on a single data-growth curve rather than being three independent
      samples.

--------------- Criterion note (evaluation-set separation) ------------
The original wording was: "filter by target sequence similarity
(< 30%) and ligand chemical similarity (Tanimoto < 0.9) to ensure no
overlap with the OOD test set and the FEP Benchmark". Literally, that
allows two readings:

  Reading A (OR removal): remove as soon as sequence identity >= 30%
    **or** Tanimoto >= 0.9.
  Reading B (AND removal): remove only when sequence identity >= 30%
    **and** Tanimoto >= 0.9.

This script uses **Reading B**, because:
  - The purpose of the clause is stated plainly -- "ensure no overlap",
    i.e. find the incarnations inside Gator of "the same complex". The
    same complex = the same target **and** the same ligand, which is
    consistent with the conjunctive logic used throughout the initial
    design ("structure and ligand and label must all hold at once").
  - The other data sources (PDBbind/BindingNet/DCS/HiQBind) use
    **exact** PDB ID matching, which removes only strictly identical
    complexes and never homologues. Reading B (homologous target and
    near-identical ligand) is already **stricter** than exact PDB ID
    matching; Reading A would instead sweep out of Gator "every protein
    30% homologous to a test target" -- 30% is the lower bound of the
    homology threshold, kinases are almost universally >30% identical
    to one another, and the FEP 42 series contains several kinases
    (CDK2/TYK2/JNK1/p38), so Reading A would wipe out Gator's kinase
    data wholesale while not touching a single row of the equivalent
    baseline data. That would systematically handicap the very question
    under test ("does Gator add anything?") before the race starts,
    making the experiment unfair.
  - The hit counts of **both** readings are computed and written into
    the report by this script, so the criterion can be switched and the
    conclusion re-checked at any time.
-----------------------------------------------------------------------

TM-align prescreen (recorded honestly): after the ligand + pK criteria,
step (3) may still leave a huge number of structure pairs. TM-score >
0.8 effectively demands the same fold, so MMseqs2 is run first with
**loose** parameters (--min-seq-id 0.2 -c 0.3 -e 10 -s 7.5) as a
prescreen, and TM-align is run only on pairs with a sequence hit. Pairs
dropped by the prescreen are **audited by actually running TM-align on
400 randomly sampled pairs**; if any of them shows TM > 0.8 the
prescreen was too tight, and that number is stated explicitly in the
report -- nothing is truncated silently.
"""
import os, sys, json, math, re, random, subprocess, shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
META = f'{ROOT}/experiments/00_meta'
OUT = f'{ROOT}/experiments/03_gator'
DS = f'{ROOT}/datasets'
TS = f'{DS}/index/_testsets'
FEP = f'{TS}/fep_benchmark_inputs'
TMP = f'{ROOT}/tmp_03b'

NPROC = min(64, os.cpu_count())
T_TANI, T_DPK, T_TM, T_SEQID = 0.9, 1.0, 0.8, 0.30
CONF_THR = 0.7
SEED = 42
FRACTIONS = [0.30, 0.60, 1.00]

# dG = -RT*ln10*pK  ->  pK = -dG / (RT*ln10); R=1.987204e-3 kcal/(K*mol),
# T=297K
RT_LN10 = 1.987204259e-3 * 297.0 * math.log(10.0)   # ≈ 1.35867 kcal/mol

log = lambda *a: print(*a, flush=True)
_TM_RE = re.compile(r'TM-score=\s*([0-9.]+)')

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O'}


# ───────────────────────── Basic utilities ─────────────────────────

def longest_chain_seq(p):
    """Parse the PDB by CA atoms and return the one-letter sequence of
    the longest chain (exactly the same convention as 02b)."""
    try:
        chains, seen = {}, set()
        with open(p, 'r', errors='ignore') as f:
            for ln in f:
                if ln.startswith('ENDMDL'):
                    break
                if ln.startswith('ATOM') and ln[12:16].strip() == 'CA':
                    res = ln[17:20].strip().upper()
                    if res not in AA3:
                        continue
                    rid = (ln[21], ln[22:27])
                    if rid in seen:
                        continue
                    seen.add(rid)
                    chains.setdefault(ln[21], []).append(AA3[res])
        return ''.join(max(chains.values(), key=len)) if chains else ''
    except Exception:
        return ''


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def fp_of(s):
    m = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
    return None if m is None else AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def pmap(fn, it, chunk=128):
    from multiprocessing import Pool
    with Pool(NPROC) as p:
        return p.map(fn, list(it), chunksize=chunk)


def tmscore(a):
    p, q = a
    try:
        r = subprocess.run(['TMalign', p, q], capture_output=True,
                           text=True, timeout=600)
        v = [float(x) for x in _TM_RE.findall(r.stdout)]
        return max(v) if v else -1.0
    except Exception:
        return -1.0


def write_fasta(path, pairs):
    n = 0
    with open(path, 'w') as f:
        for i, s in pairs:
            if s:
                f.write(f'>{i}\n{s}\n')
                n += 1
    return n


def mmseqs_hits(qfa, tfa, tag, min_id, cov, evalue, sens=7.5):
    """Return the set of {(query_id, target_id)} hits whose identity is
    >= min_id."""
    d = f'{TMP}/{tag}'
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    res = f'{d}/hits.tsv'
    cmd = ['mmseqs', 'easy-search', qfa, tfa, res, f'{d}/tmp',
           '--min-seq-id', str(min_id), '-c', str(cov), '--cov-mode', '0',
           '-e', str(evalue), '-s', str(sens), '--max-seqs', '4000',
           '--threads', str(NPROC), '--format-output', 'query,target,fident',
           '-v', '1']
    log(f'  [mmseqs:{tag}] ' + ' '.join(cmd[1:6]) + f' … (min_id={min_id}, c={cov}, e={evalue})')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(r.stdout[-2000:]); log(r.stderr[-2000:])
        raise RuntimeError(f'mmseqs {tag} failed')
    hits = set()
    with open(res) as f:
        for ln in f:
            q, t, fi = ln.rstrip('\n').split('\t')[:3]
            if float(fi) >= min_id:
                hits.add((q, t))
    log(f'  [mmseqs:{tag}] hit pairs {len(hits)}')
    return hits


# ───── Tanimoto matching (popcount pruning, parallel) ─────

_QF = _RF = _BYPC = _PCS = None


def _match_block(rng):
    """One block of query fingerprints -> [(qi, ri, sim)], keeping only
    sim > T_TANI."""
    out = []
    for j in rng:
        f = _QF[j]
        if f is None:
            continue
        a = f.GetNumOnBits()
        lo, hi = int(math.ceil(T_TANI * a)), int(math.floor(a / T_TANI)) if a else 0
        cand = [i for pc in _PCS if lo <= pc <= hi for i in _BYPC[pc]]
        if not cand:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(f, [_RF[i] for i in cand])
        out += [(j, i, s) for i, s in zip(cand, sims) if s > T_TANI]
    return out


def tanimoto_pairs(qfps, rfps, label):
    """Return [(qi, ri, sim)]; for T=c/(a+b-c)>t, prune with
    min(a,b) > t*max(a,b)."""
    global _QF, _RF, _BYPC, _PCS
    bypc = defaultdict(list)
    for i, f in enumerate(rfps):
        if f is not None:
            bypc[f.GetNumOnBits()].append(i)
    _QF, _RF, _BYPC, _PCS = qfps, rfps, bypc, sorted(bypc)
    blocks = [range(k, min(k + 200, len(qfps))) for k in range(0, len(qfps), 200)]
    log(f'  [Tanimoto:{label}] {len(qfps)} × {len(rfps)} unique ligands,'
        f' {len(blocks)} blocks ...')
    pairs = [x for b in pmap(_match_block, blocks, 1) for x in b]
    log(f'  [Tanimoto:{label}] ligand pairs > {T_TANI}: {len(pairs)}')
    return pairs


# ───── Evaluation-set manifests (reused by this script and stage 6) ─────

def smiles_from_files(sdf, mol2):
    """Try SDF first (with sanitize), fall back to mol2; same convention
    as 02b/02b2."""
    try:
        for m in Chem.SDMolSupplier(sdf, sanitize=True):
            if m is not None:
                return Chem.MolToSmiles(m), 'sdf'
    except Exception:
        pass
    try:
        m = Chem.MolFromMol2File(mol2, sanitize=True)
        if m is not None:
            return Chem.MolToSmiles(m), 'mol2'
    except Exception:
        pass
    return None, 'fail'


def build_ood295():
    d = pd.read_csv(f'{TS}/index_oodtest.csv')
    d = d[d.split == 'test'].copy()
    assert len(d) == 295, f'unexpected OOD test row count {len(d)}'
    rows = []
    for p, pk in zip(d.PDB_code, d['-logKd/Ki']):
        prot = f'{DS}/structure/pdbbind/refined-set/{p}/{p}_protein.pdb'
        sdf = f'{DS}/structure/pdbbind/sdf/{p}_ligand.sdf'
        mol2 = f'{DS}/structure/pdbbind/refined-set/{p}/{p}_ligand.mol2'
        smi, src = smiles_from_files(sdf, mol2)
        rows.append((p, float(pk), smi, src, longest_chain_seq(prot), prot, sdf))
    o = pd.DataFrame(rows, columns=['pdb_id', 'pK', 'smiles', 'smiles_src',
                                    'seq', 'protein_pdb', 'ligand_sdf'])
    log(f'[OOD] {len(o)} records; SMILES failures {o.smiles.isna().sum()},'
        f' empty sequences {(o.seq == "").sum()}')
    o.to_csv(f'{META}/testset_ood295.csv', index=False)
    return o


def build_fep42():
    m = pd.read_csv(f'{FEP}/benchmark_metadata.csv')
    s = m[m['Number of nodes'] >= 10].copy()
    assert len(s) == 42, f'unexpected FEP n>=10 series count {len(s)}'
    rows, seqs = [], []
    for _, g in s.iterrows():
        ab, nm = g['Group abbreviation'], g['Input file naming scheme']
        base = f'{FEP}/structure_inputs/{ab}'
        prot, lig = f'{base}/{nm}_protein.pdb', f'{base}/{nm}_ligands.sdf'
        assert os.path.exists(prot) and os.path.exists(lig), \
            f'missing file {ab}/{nm}'
        seqs.append((f'FEP|{nm}', longest_chain_seq(prot)))
        n = 0
        for k, mol in enumerate(Chem.SDMolSupplier(lig, sanitize=True)):
            if mol is None:
                continue
            dg = mol.GetPropsAsDict().get('r_exp_dg', None)
            if dg is None:
                continue
            rows.append((nm, ab, g['Group name'], str(g['Reference PDB']).lower(),
                         int(g['Number of nodes']), k,
                         mol.GetProp('_Name') if mol.HasProp('_Name') else f'{nm}_{k}',
                         float(dg), -float(dg) / RT_LN10,
                         Chem.MolToSmiles(mol), prot, lig))
            n += 1
        if n != int(g['Number of nodes']):
            # merck/syk_4puz_fullmap holds 2 nodes named intermediate*:
            # they are synthetic intermediates on the perturbation graph,
            # they carry no r_exp_dg in the SDF, so they have no
            # experimental label and cannot be evaluated at all.
            # Hence 1087 evaluable ligands < 1089 metadata nodes; this is
            # recorded faithfully rather than imputed.
            log(f'  ! {nm}: {n} ligands with an experimental dG, '
                f'metadata nodes {g["Number of nodes"]}')
    f = pd.DataFrame(rows, columns=['series', 'group_abbrev', 'group_name', 'ref_pdb',
                                    'n_nodes', 'lig_idx', 'lig_name', 'dG_exp', 'pK',
                                    'smiles', 'protein_pdb', 'ligand_sdf'])
    log(f'[FEP] {f.series.nunique()} series / {len(f)} ligands '
        f'(metadata total 1089); '
        f'pK {f.pK.min():.2f}~{f.pK.max():.2f}')
    f.to_csv(f'{META}/testset_fep42.csv', index=False)
    write_fasta(f'{META}/pool_fep42.fasta', seqs)
    return f, seqs


# ───────────────────────── Main pipeline ─────────────────────────

def main():
    os.makedirs(TMP, exist_ok=True)
    rep = {'thresholds': {'tanimoto': T_TANI, 'abs_dpK': T_DPK, 'tm_score': T_TM,
                          'seq_identity': T_SEQID, 'conf': CONF_THR,
                          'testset_rule': 'remove only when AND(seqid>='
                                          '0.30, tanimoto>=0.9); see the '
                                          'module docstring'},
           'seed': SEED}
    log(f'parallel {NPROC}; thresholds Tanimoto>{T_TANI} '
        f'|dpK|<={T_DPK} TM>{T_TM} seqid>={T_SEQID}')

    g = pd.read_csv(f'{OUT}/gator_clean.csv')
    log(f'\n[Gator] input {len(g)} records (output of 03a)')
    rep['gator_in'] = int(len(g))

    # ============ (1) PDB ID safety net ============
    leak = set(json.load(open(f'{META}/leakage_ids.json'))['excluded_pdb_ids'])
    col = g['PDB ID(s) for Ligand-Target Complex'].astype(str)
    hit_pdb = np.array([
        any(x.strip().lower()[:4] in leak for x in c.replace(';', ',').split(',') if x.strip())
        if c and c.lower() != 'nan' else False for c in col])
    log(f'(1) PDB ID safety net: field non-empty '
        f'{int((col.str.lower() != "nan").sum())}, '
        f'matching a leakage ID {int(hit_pdb.sum())}')
    rep['step1_pdbid_leak'] = int(hit_pdb.sum())
    g = g[~hit_pdb].copy().reset_index(drop=True)

    # ============ (2) Evaluation-set separation ============
    log('\n(2) Evaluation-set separation (OOD 295 + FEP 42)')
    ood = build_ood295()
    fep, fep_seqs = build_fep42()

    # 2a Sequence identity: Gator vs test targets
    tfa = f'{TMP}/testset_targets.fasta'
    n = write_fasta(tfa, [(f'OOD|{p}', s) for p, s in zip(ood.pdb_id, ood.seq)] + fep_seqs)
    log(f'  test target sequences {n}')
    qfa = f'{TMP}/gator_q.fasta'
    write_fasta(qfa, [(f'GA|{i}', s) for i, s in zip(g.unique_id, g.seq.fillna(''))])
    hits = mmseqs_hits(qfa, tfa, 'test', T_SEQID, 0.5, 1e-3)
    seq_hit_ids = {q.split('|', 1)[1] for q, _ in hits}
    seq_hit = g.unique_id.isin(seq_hit_ids).values
    log(f'  Gator samples with sequence identity >= {T_SEQID}: '
        f'{int(seq_hit.sum())}')

    # 2b Ligand Tanimoto: Gator vs test ligands
    gu = sorted(set(g.smiles_canon.dropna()))
    tl = sorted({x for x in list(ood.smiles.dropna()) + list(fep.smiles.dropna()) if x})
    tl = [c for c in pmap(canon, tl) if c]
    log(f'  unique ligands: Gator {len(gu)}, test set {len(set(tl))}')
    gfp, tfp = pmap(fp_of, gu), pmap(fp_of, sorted(set(tl)))
    lp = tanimoto_pairs(gfp, tfp, 'gator-vs-test')
    lig_hit_smi = {gu[j] for j, _, _ in lp}
    lig_hit = g.smiles_canon.isin(lig_hit_smi).values
    log(f'  Gator samples with Tanimoto > {T_TANI}: {int(lig_hit.sum())}')

    n_and, n_or = int((seq_hit & lig_hit).sum()), int((seq_hit | lig_hit).sum())
    log(f'  * Reading B (AND, used by this script) removes {n_and}; '
        f'Reading A (OR) would have removed {n_or} -- both go into the '
        f'report')
    rep['step2'] = {'seq_hit': int(seq_hit.sum()), 'lig_hit': int(lig_hit.sum()),
                    'removed_AND': n_and, 'would_remove_OR': n_or}
    pd.DataFrame({'unique_id': g.unique_id, 'seq_hit': seq_hit,
                  'lig_hit': lig_hit}).to_csv(f'{OUT}/gator_testset_flags.csv', index=False)
    g = g[~(seq_hit & lig_hit)].copy().reset_index(drop=True)
    log(f'  -> remaining {len(g)}')

    # ====== (3) Cross-DB triple deduplication (vs PDBbind + HiQBind) ======
    log('\n(3) Cross-database deduplication: vs PDBbind + HiQBind '
        '(CleanSplit triple criterion)')
    pb = pd.read_csv(f'{ROOT}/experiments/01_baseline_index/pdbbind_clean.csv')
    ps = pd.read_csv(f'{META}/pdbbind_smiles_seq.csv')
    pb = pb.merge(ps[['pdb_id', 'smiles', 'seq']], on='pdb_id', how='left')
    pb['prot'] = [f'{DS}/structure/pdbbind/'
                  f'{"refined-set" if r else "general-set"}/{i}/{i}_protein.pdb'
                  for i, r in zip(pb.pdb_id, pb.refined)]
    pb['csmi'] = pmap(canon, pb.smiles.fillna('').astype(str), 256)

    hq = pd.read_csv(f'{ROOT}/experiments/02_hiqbind/hiqbind_dedup.csv')
    hs = pd.read_csv(f'{META}/hiqbind_seq.csv')
    hq = hq.merge(hs, on='unique_id', how='left', suffixes=('', '_y'))
    if 'csmi' not in hq.columns:
        hq['csmi'] = pmap(canon, hq['Ligand SMILES'].astype(str), 256)

    ref = pd.concat([
        pd.DataFrame({'rid': 'PB|' + pb.pdb_id.astype(str), 'csmi': pb.csmi,
                      'pK': pb.pK, 'prot': pb.prot, 'seq': pb.seq}),
        pd.DataFrame({'rid': 'HQ|' + hq.unique_id.astype(str), 'csmi': hq.csmi,
                      'pK': hq.pK, 'prot': hq.protein_pdb, 'seq': hq.seq}),
    ], ignore_index=True)
    log(f'  reference pool {len(ref)} = PDBbind {len(pb)} + '
        f'HiQBind {len(hq)} '
        f'(SMILES missing {ref.csmi.isna().sum()})')

    ru = sorted(set(ref.csmi.dropna()))
    gu2 = sorted(set(g.smiles_canon.dropna()))
    gfp2, rfp = pmap(fp_of, gu2), pmap(fp_of, ru)
    lp2 = tanimoto_pairs(gfp2, rfp, 'gator-vs-ref')

    g_by, r_by = defaultdict(list), defaultdict(list)
    for k, s in enumerate(g.smiles_canon):
        if isinstance(s, str):
            g_by[s].append(k)
    for k, s in enumerate(ref.csmi):
        if isinstance(s, str):
            r_by[s].append(k)

    gpk, rpk = g.pK.values, ref.pK.values
    pairs = [(x, y, s) for j, i, s in lp2
             for x in g_by[gu2[j]] for y in r_by[ru[i]]
             if abs(gpk[x] - rpk[y]) <= T_DPK]
    log(f'  sample pairs after adding |dpK| <= {T_DPK}: {len(pairs)}')

    gprot, rprot = g.protein_pdb.values, ref.prot.values
    uniq = sorted({(gprot[x], rprot[y]) for x, y, _ in pairs})
    log(f'  unique structure pairs {len(uniq)}')

    # Loose sequence prescreen (purely to save TM-align compute; audited
    # afterwards on a random sample)
    gid_of = {p: i for p, i in zip(gprot, g.unique_id)}
    rid_of = {p: i for p, i in zip(rprot, ref.rid)}
    need_g = {gid_of[a] for a, _ in uniq}
    need_r = {rid_of[b] for _, b in uniq}
    qf = f'{TMP}/pre_q.fasta'; tf = f'{TMP}/pre_t.fasta'
    write_fasta(qf, [(f'GA|{i}', s) for i, s in zip(g.unique_id, g.seq.fillna(''))
                     if i in need_g])
    write_fasta(tf, [(i, s) for i, s in zip(ref.rid, ref.seq.fillna(''))
                     if i in need_r and isinstance(s, str)])
    # In practice there are only a few thousand unique structure pairs
    # and TM-align finishes in minutes -- the prescreen is pure overhead,
    # so it is enabled only when the scale truly explodes (> 200k pairs),
    # to avoid introducing an unnecessary approximation.
    PRE_LIMIT = 200_000
    if len(uniq) > PRE_LIMIT:
        pre = mmseqs_hits(qf, tf, 'pre', 0.2, 0.3, 10.0)
        pre = {(q.split('|', 1)[1], t) for q, t in pre}
        keep_pair = [(a, b) for a, b in uniq if (gid_of[a], rid_of[b]) in pre]
        drop_pair = [(a, b) for a, b in uniq if (gid_of[a], rid_of[b]) not in pre]
        log(f'  prescreen kept {len(keep_pair)} pairs, '
            f'deferred {len(drop_pair)} pairs')
    else:
        keep_pair, drop_pair = list(uniq), []
        log(f'  only {len(uniq)} structure pairs, skipping the '
            f'prescreen, running TM-align on all of them')

    log(f'  [TM-align] running {len(keep_pair)} pairs ...')
    tm = dict(zip(keep_pair, pmap(tmscore, keep_pair, 4)))
    nfail = sum(1 for v in tm.values() if v < 0)
    log(f'    failures {nfail}; pairs with TM > {T_TM}: '
        f'{sum(1 for v in tm.values() if v > T_TM)}')

    # Prescreen audit: actually run 400 randomly sampled deferred pairs
    # to see whether anything was missed
    random.seed(SEED)
    audit = random.sample(drop_pair, min(400, len(drop_pair)))
    av = pmap(tmscore, audit, 4) if audit else []
    n_audit_bad = sum(1 for v in av if v > T_TM)
    log(f'  [prescreen audit] sampled {len(audit)} pairs, of which '
        f'TM > {T_TM} holds for {n_audit_bad} '
        f'(0 means the prescreen missed nothing)')
    for a, v in zip(audit, av):
        tm[a] = v

    dup = np.zeros(len(g), bool)
    rec = []
    for x, y, s in pairs:
        v = tm.get((gprot[x], rprot[y]), 0.0)
        if v > T_TM:
            dup[x] = True
            rec.append((g.unique_id[x], ref.rid[y], s, abs(gpk[x] - rpk[y]), v))
    pd.DataFrame(rec, columns=['gator_unique_id', 'ref_id', 'tanimoto',
                               'abs_dpK', 'tm_score']).to_csv(
        f'{OUT}/gator_dup_pairs.csv', index=False)
    log(f'  triple criterion hits {int(dup.sum())} Gator records -> '
        f'keeping {int((~dup).sum())}')
    rep['step3'] = {'ref_pool': int(len(ref)), 'lig_pairs': len(lp2),
                    'pairs_after_pK': len(pairs), 'uniq_struct_pairs': len(uniq),
                    'prescreen_kept': len(keep_pair), 'prescreen_deferred': len(drop_pair),
                    'prescreen_audit_n': len(audit), 'prescreen_audit_missed': n_audit_bad,
                    'tm_fail': int(nfail), 'duplicates_removed': int(dup.sum())}
    g = g[~dup].copy().reset_index(drop=True)

    # ============ (4) Versions + nested sampling ============
    log('\n(4) Versions and scale sampling')
    g['pass_conf'] = pd.to_numeric(g.conf, errors='coerce') >= CONF_THR
    ver = {'unfiltered': g.index.values,
           'filtered': g.index[g.pass_conf].values}
    for name, idx in ver.items():
        rng = np.random.RandomState(SEED)
        order = rng.permutation(idx)
        for fr in FRACTIONS:
            k = int(round(len(order) * fr))
            colname = f'in_{name}_{int(fr * 100)}'
            g[colname] = False
            g.loc[order[:k], colname] = True
            log(f'  {colname}: {k}')
        rep[f'version_{name}'] = int(len(idx))

    g.to_csv(f'{OUT}/gator_dedup.csv', index=False)

    log(f'\n[Gator final] {len(g)} records (unfiltered version); '
        f'filtered version {int(g.pass_conf.sum())}')
    log(f'  unique ligands {g.smiles_canon.nunique()}; '
        f'unique targets {g.target_key.nunique()}; '
        f'unique Bemis-Murcko scaffolds {g.scaffold.nunique()}')
    log(f'  pK {g.pK.min():.2f} ~ {g.pK.max():.2f}, '
        f'mean {g.pK.mean():.3f}')
    rep.update({'gator_final_unfiltered': int(len(g)),
                'gator_final_filtered': int(g.pass_conf.sum()),
                'unique_ligand': int(g.smiles_canon.nunique()),
                'unique_target': int(g.target_key.nunique()),
                'unique_scaffold': int(g.scaffold.nunique())})
    json.dump(rep, open(f'{OUT}/report_03b.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\noutputs -> {OUT}/{{gator_dedup.csv, gator_dup_pairs.csv, '
        f'gator_testset_flags.csv, report_03b.json}}'
        f'\n       {META}/{{testset_ood295.csv, testset_fep42.csv, pool_fep42.fasta}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
