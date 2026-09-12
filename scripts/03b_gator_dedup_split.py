#!/usr/bin/env python
"""
阶段 3b：GatorAffinity 跨库去重 + 评测集分离 + 规模采样（PLAN §3.3.1 / §3.4.2）。

四件事，顺序执行：

  (1) 安全网：Gator 索引自带 `PDB ID(s) for Ligand-Target Complex` 字段，
      虽然 PLAN 说 Gator「无真实 PDB ID」，但该字段非空的行仍按 §3.4.2 的
      PDB ID 键剔除（330 个泄漏 ID）。免费的额外保险，不与后续判据冲突。

  (2) 评测集分离（PLAN §3.3.1 末条 / §3.4.2 末段）：
      靶点序列一致性（MMseqs2）+ 配体 ECFP4 Tanimoto，对 OOD 295 与 FEP 42 系列。
      ★ 判据组合方式见下方「口径说明」。

  (3) 跨库去重（PLAN §3.3.1）：与 PDBbind、HiQBind（去重后）比 CleanSplit 三重判据
      —— Tanimoto > 0.9 且 |ΔpK| <= 1.0 且 TM-score > 0.8，三者同时满足才判重复，
      只剔 Gator 侧。

  (4) 版本与采样（PLAN §3.3.2 / §4.1）：
      未筛选版 = 去重后全量；筛选版 = 未筛选版 ∩ (Boltz-1 confidence >= 0.7)。
      各自按 seed 42 打乱后取前 30% / 60% / 100%，**嵌套**（30% ⊂ 60% ⊂ 100%），
      这样 Scaling Law 的三个点是同一条数据增长曲线上的点，而非三份独立样本。

────────────────────────── 口径说明（评测集分离） ──────────────────────────
PLAN 原文：「采用靶点序列相似度（< 30%）和配体化学相似度（Tanimoto < 0.9）过滤，
确保与 OOD 测试集和 FEP Benchmark 无重叠」。字面有两种读法：

  读法 A（OR 剔除）：只要序列一致性 >= 30% **或** Tanimoto >= 0.9 就剔除。
  读法 B（AND 剔除）：序列一致性 >= 30% **且** Tanimoto >= 0.9 才剔除。

本脚本采用**读法 B**，理由：
  · 该条款的目的写得很清楚 —— 「确保无重叠」，即找出「同一个复合物」在 Gator 中的
    化身。同一复合物 = 同一靶点 **且** 同一配体，这与 PLAN 全篇（§3.2、§3.3.1）
    「结构 ∧ 配体 ∧ 标签 三者同时满足」的合取逻辑一致。
  · 其余数据源（PDBbind/BindingNet/DCS/HiQBind）用的是 PDB ID **精确**匹配，
    只剔除完全相同的复合物、不剔同源物。读法 B（同源靶点 ∧ 近乎相同配体）已经
    **严于** PDB ID 精确匹配；读法 A 则会把「任何与测试靶点 30% 同源的蛋白」全部
    清出 Gator —— 30% 是同源性下限阈值，激酶类几乎全域互相 >30%，而 FEP 42 系列
    含 CDK2/TYK2/JNK1/p38 等多个激酶，读法 A 会把 Gator 的激酶数据成片清空，
    却对 baseline 的同类数据一行不动。那会让「Gator 有无增益」这个待检验的问题
    在起跑线上就被系统性压低，实验不公平。
  · 两种读法的命中数本脚本**都会计算并写入报告**，口径可随时切换、结论可复核。
─────────────────────────────────────────────────────────────────────────

TM-align 预筛（诚实记录）：第 (3) 步在配体+pK 判据存活后仍可能有海量结构对。
TM-score > 0.8 实质要求同一折叠，故先用 MMseqs2 以**宽松**参数
（--min-seq-id 0.2 -c 0.3 -e 10 -s 7.5）预筛，只对有序列命中的对跑 TM-align。
被预筛剔掉的对会**随机抽样 400 对实跑 TM-align 做对照**，若其中出现 TM > 0.8
则说明预筛过紧，报告中会明确给出该数字，不做静默截断。
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

# ΔG = -RT·ln10·pK  →  pK = -ΔG / (RT·ln10)；R=1.987204e-3 kcal/(K·mol), T=297K
RT_LN10 = 1.987204259e-3 * 297.0 * math.log(10.0)   # ≈ 1.35867 kcal/mol

log = lambda *a: print(*a, flush=True)
_TM_RE = re.compile(r'TM-score=\s*([0-9.]+)')

AA3 = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
       'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
       'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
       'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O'}


# ──────────────────────────── 基础工具 ────────────────────────────

def longest_chain_seq(p):
    """按 CA 原子解析 PDB，返回最长链的单字母序列（与 02b 完全一致的口径）。"""
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
    """返回 {(query_id, target_id)} 中一致性 >= min_id 的命中集合。"""
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
        raise RuntimeError(f'mmseqs {tag} 失败')
    hits = set()
    with open(res) as f:
        for ln in f:
            q, t, fi = ln.rstrip('\n').split('\t')[:3]
            if float(fi) >= min_id:
                hits.add((q, t))
    log(f'  [mmseqs:{tag}] 命中对 {len(hits)}')
    return hits


# ──────────────── Tanimoto 匹配（popcount 剪枝，并行） ────────────────

_QF = _RF = _BYPC = _PCS = None


def _match_block(rng):
    """一批 query 指纹 → [(qi, ri, sim)]，仅保留 sim > T_TANI。"""
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
    """返回 [(qi, ri, sim)]；对 T=c/(a+b-c)>t 用 min(a,b) > t·max(a,b) 剪枝。"""
    global _QF, _RF, _BYPC, _PCS
    bypc = defaultdict(list)
    for i, f in enumerate(rfps):
        if f is not None:
            bypc[f.GetNumOnBits()].append(i)
    _QF, _RF, _BYPC, _PCS = qfps, rfps, bypc, sorted(bypc)
    blocks = [range(k, min(k + 200, len(qfps))) for k in range(0, len(qfps), 200)]
    log(f'  [Tanimoto:{label}] {len(qfps)} × {len(rfps)} 唯一配体，{len(blocks)} 块 …')
    pairs = [x for b in pmap(_match_block, blocks, 1) for x in b]
    log(f'  [Tanimoto:{label}] > {T_TANI} 的配体对 {len(pairs)}')
    return pairs


# ──────────────────── 评测集清单（供本脚本与阶段 6 复用） ────────────────────

def smiles_from_files(sdf, mol2):
    """先 SDF（sanitize），失败退 mol2；与 02b/02b2 的口径一致。"""
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
    assert len(d) == 295, f'OOD test 行数异常 {len(d)}'
    rows = []
    for p, pk in zip(d.PDB_code, d['-logKd/Ki']):
        prot = f'{DS}/structure/pdbbind/refined-set/{p}/{p}_protein.pdb'
        sdf = f'{DS}/structure/pdbbind/sdf/{p}_ligand.sdf'
        mol2 = f'{DS}/structure/pdbbind/refined-set/{p}/{p}_ligand.mol2'
        smi, src = smiles_from_files(sdf, mol2)
        rows.append((p, float(pk), smi, src, longest_chain_seq(prot), prot, sdf))
    o = pd.DataFrame(rows, columns=['pdb_id', 'pK', 'smiles', 'smiles_src',
                                    'seq', 'protein_pdb', 'ligand_sdf'])
    log(f'[OOD] {len(o)} 条；SMILES 失败 {o.smiles.isna().sum()}，序列为空 {(o.seq == "").sum()}')
    o.to_csv(f'{META}/testset_ood295.csv', index=False)
    return o


def build_fep42():
    m = pd.read_csv(f'{FEP}/benchmark_metadata.csv')
    s = m[m['Number of nodes'] >= 10].copy()
    assert len(s) == 42, f'FEP n>=10 系列数异常 {len(s)}'
    rows, seqs = [], []
    for _, g in s.iterrows():
        ab, nm = g['Group abbreviation'], g['Input file naming scheme']
        base = f'{FEP}/structure_inputs/{ab}'
        prot, lig = f'{base}/{nm}_protein.pdb', f'{base}/{nm}_ligands.sdf'
        assert os.path.exists(prot) and os.path.exists(lig), f'缺文件 {ab}/{nm}'
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
            # merck/syk_4puz_fullmap 含 2 个名为 intermediate* 的节点：它们是扰动图上的
            # 合成中间体，SDF 里没有 r_exp_dg，本就无实验标签、不可评测。
            # 故可评测配体数 1087 < 元数据节点数 1089，如实记录，不做插补。
            log(f'  ! {nm}: 有实验 ΔG 的配体 {n} 个，元数据节点数 {g["Number of nodes"]}')
    f = pd.DataFrame(rows, columns=['series', 'group_abbrev', 'group_name', 'ref_pdb',
                                    'n_nodes', 'lig_idx', 'lig_name', 'dG_exp', 'pK',
                                    'smiles', 'protein_pdb', 'ligand_sdf'])
    log(f'[FEP] {f.series.nunique()} 系列 / {len(f)} 配体（元数据合计 1089）；'
        f'pK {f.pK.min():.2f}~{f.pK.max():.2f}')
    f.to_csv(f'{META}/testset_fep42.csv', index=False)
    write_fasta(f'{META}/pool_fep42.fasta', seqs)
    return f, seqs


# ──────────────────────────────── 主流程 ────────────────────────────────

def main():
    os.makedirs(TMP, exist_ok=True)
    rep = {'thresholds': {'tanimoto': T_TANI, 'abs_dpK': T_DPK, 'tm_score': T_TM,
                          'seq_identity': T_SEQID, 'conf': CONF_THR,
                          'testset_rule': 'AND(seqid>=0.30, tanimoto>=0.9) 才剔除；见脚本文档'},
           'seed': SEED}
    log(f'并行 {NPROC}；阈值 Tanimoto>{T_TANI} |dpK|<={T_DPK} TM>{T_TM} seqid>={T_SEQID}')

    g = pd.read_csv(f'{OUT}/gator_clean.csv')
    log(f'\n[Gator] 输入 {len(g)} 条（03a 产物）')
    rep['gator_in'] = int(len(g))

    # ============ (1) PDB ID 安全网 ============
    leak = set(json.load(open(f'{META}/leakage_ids.json'))['excluded_pdb_ids'])
    col = g['PDB ID(s) for Ligand-Target Complex'].astype(str)
    hit_pdb = np.array([
        any(x.strip().lower()[:4] in leak for x in c.replace(';', ',').split(',') if x.strip())
        if c and c.lower() != 'nan' else False for c in col])
    log(f'(1) PDB ID 安全网：字段非空 {int((col.str.lower() != "nan").sum())}，'
        f'命中泄漏 ID {int(hit_pdb.sum())} 条')
    rep['step1_pdbid_leak'] = int(hit_pdb.sum())
    g = g[~hit_pdb].copy().reset_index(drop=True)

    # ============ (2) 评测集分离 ============
    log('\n(2) 评测集分离（OOD 295 + FEP 42）')
    ood = build_ood295()
    fep, fep_seqs = build_fep42()

    # 2a 序列一致性：Gator vs 测试靶点
    tfa = f'{TMP}/testset_targets.fasta'
    n = write_fasta(tfa, [(f'OOD|{p}', s) for p, s in zip(ood.pdb_id, ood.seq)] + fep_seqs)
    log(f'  测试靶点序列 {n} 条')
    qfa = f'{TMP}/gator_q.fasta'
    write_fasta(qfa, [(f'GA|{i}', s) for i, s in zip(g.unique_id, g.seq.fillna(''))])
    hits = mmseqs_hits(qfa, tfa, 'test', T_SEQID, 0.5, 1e-3)
    seq_hit_ids = {q.split('|', 1)[1] for q, _ in hits}
    seq_hit = g.unique_id.isin(seq_hit_ids).values
    log(f'  序列一致性 >= {T_SEQID} 的 Gator 样本 {int(seq_hit.sum())}')

    # 2b 配体 Tanimoto：Gator vs 测试配体
    gu = sorted(set(g.smiles_canon.dropna()))
    tl = sorted({x for x in list(ood.smiles.dropna()) + list(fep.smiles.dropna()) if x})
    tl = [c for c in pmap(canon, tl) if c]
    log(f'  唯一配体：Gator {len(gu)}，测试集 {len(set(tl))}')
    gfp, tfp = pmap(fp_of, gu), pmap(fp_of, sorted(set(tl)))
    lp = tanimoto_pairs(gfp, tfp, 'gator-vs-test')
    lig_hit_smi = {gu[j] for j, _, _ in lp}
    lig_hit = g.smiles_canon.isin(lig_hit_smi).values
    log(f'  Tanimoto > {T_TANI} 的 Gator 样本 {int(lig_hit.sum())}')

    n_and, n_or = int((seq_hit & lig_hit).sum()), int((seq_hit | lig_hit).sum())
    log(f'  ★ 读法 B（AND，本脚本采用）剔除 {n_and} 条；'
        f'读法 A（OR）本会剔除 {n_or} 条 —— 两者均记入报告')
    rep['step2'] = {'seq_hit': int(seq_hit.sum()), 'lig_hit': int(lig_hit.sum()),
                    'removed_AND': n_and, 'would_remove_OR': n_or}
    pd.DataFrame({'unique_id': g.unique_id, 'seq_hit': seq_hit,
                  'lig_hit': lig_hit}).to_csv(f'{OUT}/gator_testset_flags.csv', index=False)
    g = g[~(seq_hit & lig_hit)].copy().reset_index(drop=True)
    log(f'  → 剩余 {len(g)}')

    # ============ (3) 跨库三重去重（vs PDBbind + HiQBind） ============
    log('\n(3) 跨库去重：vs PDBbind + HiQBind（CleanSplit 三重判据）')
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
    log(f'  参考池 {len(ref)} 条 = PDBbind {len(pb)} + HiQBind {len(hq)}'
        f'（SMILES 缺失 {ref.csmi.isna().sum()}）')

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
    log(f'  叠加 |dpK| <= {T_DPK} 后样本对 {len(pairs)}')

    gprot, rprot = g.protein_pdb.values, ref.prot.values
    uniq = sorted({(gprot[x], rprot[y]) for x, y, _ in pairs})
    log(f'  唯一结构对 {len(uniq)}')

    # 宽松序列预筛（只为省 TM-align 算力，事后抽样对照）
    gid_of = {p: i for p, i in zip(gprot, g.unique_id)}
    rid_of = {p: i for p, i in zip(rprot, ref.rid)}
    need_g = {gid_of[a] for a, _ in uniq}
    need_r = {rid_of[b] for _, b in uniq}
    qf = f'{TMP}/pre_q.fasta'; tf = f'{TMP}/pre_t.fasta'
    write_fasta(qf, [(f'GA|{i}', s) for i, s in zip(g.unique_id, g.seq.fillna(''))
                     if i in need_g])
    write_fasta(tf, [(i, s) for i, s in zip(ref.rid, ref.seq.fillna(''))
                     if i in need_r and isinstance(s, str)])
    # 实测唯一结构对只有几千对，TM-align 几分钟就跑完 —— 预筛纯属多余，
    # 只在规模真的爆炸（> 20 万对）时才启用，以免引入不必要的近似。
    PRE_LIMIT = 200_000
    if len(uniq) > PRE_LIMIT:
        pre = mmseqs_hits(qf, tf, 'pre', 0.2, 0.3, 10.0)
        pre = {(q.split('|', 1)[1], t) for q, t in pre}
        keep_pair = [(a, b) for a, b in uniq if (gid_of[a], rid_of[b]) in pre]
        drop_pair = [(a, b) for a, b in uniq if (gid_of[a], rid_of[b]) not in pre]
        log(f'  预筛保留 {len(keep_pair)} 对，暂缓 {len(drop_pair)} 对')
    else:
        keep_pair, drop_pair = list(uniq), []
        log(f'  结构对仅 {len(uniq)} 对，跳过预筛，全部实跑 TM-align')

    log(f'  [TM-align] 跑 {len(keep_pair)} 对 …')
    tm = dict(zip(keep_pair, pmap(tmscore, keep_pair, 4)))
    nfail = sum(1 for v in tm.values() if v < 0)
    log(f'    失败 {nfail}；TM > {T_TM} 的对 {sum(1 for v in tm.values() if v > T_TM)}')

    # 预筛对照：随机抽 400 对被暂缓的实跑，看是否漏判
    random.seed(SEED)
    audit = random.sample(drop_pair, min(400, len(drop_pair)))
    av = pmap(tmscore, audit, 4) if audit else []
    n_audit_bad = sum(1 for v in av if v > T_TM)
    log(f'  [预筛对照] 抽检 {len(audit)} 对，其中 TM > {T_TM} 的有 {n_audit_bad} 个'
        f'（0 表示预筛未漏判）')
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
    log(f'  三重判据命中 Gator {int(dup.sum())} 条 → 保留 {int((~dup).sum())}')
    rep['step3'] = {'ref_pool': int(len(ref)), 'lig_pairs': len(lp2),
                    'pairs_after_pK': len(pairs), 'uniq_struct_pairs': len(uniq),
                    'prescreen_kept': len(keep_pair), 'prescreen_deferred': len(drop_pair),
                    'prescreen_audit_n': len(audit), 'prescreen_audit_missed': n_audit_bad,
                    'tm_fail': int(nfail), 'duplicates_removed': int(dup.sum())}
    g = g[~dup].copy().reset_index(drop=True)

    # ============ (4) 版本 + 嵌套采样 ============
    log('\n(4) 版本与规模采样')
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

    log(f'\n[Gator 最终] {len(g)} 条（未筛选版）；筛选版 {int(g.pass_conf.sum())}')
    log(f'  唯一配体 {g.smiles_canon.nunique()}；唯一靶点 {g.target_key.nunique()}；'
        f'唯一骨架 {g.scaffold.nunique()}')
    log(f'  pK {g.pK.min():.2f} ~ {g.pK.max():.2f}，均值 {g.pK.mean():.3f}')
    rep.update({'gator_final_unfiltered': int(len(g)),
                'gator_final_filtered': int(g.pass_conf.sum()),
                'unique_ligand': int(g.smiles_canon.nunique()),
                'unique_target': int(g.target_key.nunique()),
                'unique_scaffold': int(g.scaffold.nunique())})
    json.dump(rep, open(f'{OUT}/report_03b.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\n产物 → {OUT}/{{gator_dedup.csv, gator_dup_pairs.csv, '
        f'gator_testset_flags.csv, report_03b.json}}'
        f'\n       {META}/{{testset_ood295.csv, testset_fep42.csv, pool_fep42.fasta}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
