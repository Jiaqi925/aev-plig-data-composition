#!/usr/bin/env python
"""
阶段 4：七源统一图特征生成（AEV + PLIG 节点/边特征）。

七个源：pdbbind / bindingnet / bindingdb_dcs / hiqbind / gatoraffinity（训练侧）
        ood295 / fep42（评测侧）

**严格镜像 AEV-PLIG 参考实现**（generate_pdbbind_graphs.py、generate_bindingnet_graphs.py、
generate_bindingdb_graphs.py）：LoadMolasDF / LoadPDBasDF / atom_features / mol_to_graph
四个函数逐字复制；GetMolAEVs_extended 只做一处工程改动——把 AEVComputer 与
SpeciesConverter 缓存到进程级（参考实现每个复合物重建一次，纯浪费），数值完全一致。

统一 ANI-2x 参数（与参考实现、PLAN §5 一致）：
  RcR=5.1, EtaR=[19.7], RsR=16 档 0.80…4.83, RcA=2.0, Zeta=TsA=EtaA=RsA=[1.0]，
  22 原子类型，**仅保留径向项** → AEV 宽度 22×16 = 352。
  节点特征 = 15 维 PLIG 特征 + 352 维 AEV = 367；边特征 = 键型 one-hot 4 维。

配体读取规则（覆盖参考实现的两种口径，且不引入新口径）：
  · mol2：Chem.MolFromMol2File + AddHs(addCoords=True)   —— 同 pdbbind / bindingdb 参考
  · sdf ：SDMolSupplier(removeHs=False)                   —— 同 bindingnet 参考
          若读到的分子**不含显式氢**，再 AddHs(addCoords=True)。
          原因：total_num_Hs 节点特征数的是氢邻居，无氢则该特征恒为 0，属特征损坏。
          bindingnet 的 sdf 自带氢，故此分支对参考实现的复现无影响。

存储格式（不用 pickle 存 dict——173k 个复合物的 Python 对象开销与加载时间都不可接受）：
  graphs/{source}/*.npy
    x        float32 (总节点数, 367)     所有图的节点特征纵向拼接
    node_ptr int64   (N+1,)              第 i 图节点 = x[node_ptr[i]:node_ptr[i+1]]
    ei       int32   (2, 总边数)         图内局部索引
    ea       uint8   (总边数, 4)         键型 one-hot
    edge_ptr int64   (N+1,)
    ids      <U…     (N,)                unique_id，与索引表对齐
  训练时用 np.load(mmap_mode='r') 惰性读取，切片建 PyG Data，内存占用与 batch 同阶。

用法：
  python 04_generate_graphs.py                 # 全部七源
  python 04_generate_graphs.py --sources fep42 ood295
  python 04_generate_graphs.py --limit 200     # 冒烟测试：每源只跑 200 条
"""
import os, sys, json, argparse, pickle

# 必须在 import numpy / torch 之前限制 BLAS/OpenMP 线程数：
# 本脚本靠多进程并行，每个进程再开 N 个数学库线程会导致严重超订
# （实测 96 进程时单条耗时被放大 11 倍，吞吐从 26 it/s 掉到 2.3 it/s）
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
import torch

# torchani_mod 是 AEV-PLIG 仓库内的改版 TorchANI（提供 aev_computer 的原子对 AEV 输出），
# 未安装到 site-packages，必须先把仓库根目录挂到 sys.path
sys.path.insert(0, os.path.join(os.environ.get('STACEY_ROOT',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'AEV-PLIG'))
import torchani
import torchani_mod
import qcelemental as qcel
from tqdm import tqdm
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
torch.set_num_threads(1)

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AEVDIR = f'{ROOT}/AEV-PLIG'
META = f'{ROOT}/experiments/00_meta'
DS = f'{ROOT}/datasets/structure'
OUT = f'{ROOT}/experiments/04_graphs'
log = lambda *a: print(*a, flush=True)

# ANI-2x 径向参数
RcR = 5.1
EtaR = torch.tensor([19.7])
RsR = torch.tensor([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
                    2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83])
RADIAL_COEFS = [RcR, EtaR, RsR]
N_ATOM_TYPES = 22
AEV_DIM = N_ATOM_TYPES * len(EtaR) * len(RsR)     # 352
NODE_DIM = 15 + AEV_DIM                           # 367

_G = {}          # 进程级缓存


# ══════════════ 以下四个函数逐字复制自 generate_pdbbind_graphs.py ══════════════

def LoadMolasDF(mol):
    m = mol
    atoms = []
    for atom in m.GetAtoms():
        if atom.GetSymbol() != "H":
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = m.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)
    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    return df


def LoadPDBasDF(PDB, atom_keys):
    prot_atoms = []
    f = open(PDB)
    for i in f:
        if i[:4] == "ATOM":
            if (len(i[12:16].replace(" ", "")) < 4 and i[12:16].replace(" ", "")[0] != "H") \
               or (len(i[12:16].replace(" ", "")) == 4 and i[12:16].replace(" ", "")[1] != "H"
                   and i[12:16].replace(" ", "")[0] != "H"):
                prot_atoms.append([int(i[6:11]),
                                   i[17:20] + "-" + i[12:16].replace(" ", ""),
                                   float(i[30:38]), float(i[38:46]), float(i[46:54])])
    f.close()
    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[
        ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]].sort_values(
        by="ATOM_INDEX").reset_index(drop=True)
    return df


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    feature_list = []
    feature_list.extend(one_of_k_encoding(
        atom.GetSymbol(), ['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S']))
    feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors()
                             if x.GetSymbol() != "H"]))
    feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors()
                             if x.GetSymbol() == "H"]))
    feature_list.append(atom.GetExplicitValence())
    feature_list.append(1 if atom.GetIsAromatic() else 0)
    feature_list.append(1 if atom.IsInRing() else 0)
    return np.array(feature_list)


def mol_to_graph(mol, mol_df, aevs):
    features, heavy_atom_index, idx_to_idx, counter = [], [], {}, 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df['ATOM_INDEX'] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            features.append(np.append(atom_features(atom), aevs[aev_idx, :]))
            counter += 1
    edges = []
    for bond in mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            bt = [float(b) for b in one_of_k_encoding(bond.GetBondType(), [1, 12, 2, 3])]
            edges.append([idx_to_idx[idx1], idx_to_idx[idx2]] + bt)
            edges.append([idx_to_idx[idx2], idx_to_idx[idx1]] + bt)
    df = pd.DataFrame(edges, columns=['atom1', 'atom2', 'single', 'aromatic',
                                      'double', 'triple']).sort_values(by=['atom1', 'atom2'])
    return (len(mol_df), features,
            df[['atom1', 'atom2']].to_numpy().tolist(),
            df[['single', 'aromatic', 'double', 'triple']].to_numpy().tolist())


# ══════════ AEV 计算：数值同参考实现，仅把两个 computer 缓存到进程级 ══════════

def _init_worker():
    ak = pd.read_csv(f'{AEVDIR}/data/PDB_Atom_Keys.csv', sep=',')
    am = pd.DataFrame(pd.unique(ak['ATOM_TYPE']))
    am[1] = list(np.arange(len(am)) + 1)
    am = am.rename(columns={0: 'ATOM_TYPE', 1: 'ATOM_NR'})
    syms = [qcel.periodictable.to_symbol(i) for i in range(1, N_ATOM_TYPES + 1)]
    _G['atom_keys'] = ak
    _G['atom_map'] = am
    _G['aevc'] = torchani_mod.AEVComputer(RcR, 2.0, EtaR, RsR,
                                          torch.tensor([1.0]), torch.tensor([1.0]),
                                          torch.tensor([1.0]), torch.tensor([1.0]),
                                          len(syms))
    _G['sc'] = torchani.SpeciesConverter(syms)


def _load_pdb_cached(path):
    """同一蛋白常被同源系列的多个配体复用（FEP 42 蛋白 / 1087 配体、
    BindingNet 5816 蛋白 / 67989 配体），解析一次约 0.5 s，占单条总耗时 14%。
    manifest 已按蛋白路径排序 + chunksize=8，同一 worker 连续命中，单项缓存足够。"""
    c = _G.get('pdb_cache')
    if c is not None and c[0] == path:
        return c[1]
    T = LoadPDBasDF(path, _G['atom_keys'])
    _G['pdb_cache'] = (path, T)
    return T


def GetMolAEVs_extended(protein_path, mol):
    Target = _load_pdb_cached(protein_path)
    Ligand = LoadMolasDF(mol)

    distance_cutoff = RcR + 0.1
    for i in ["X", "Y", "Z"]:
        Target = Target[Target[i] < float(Ligand[i].max()) + distance_cutoff]
        Target = Target[Target[i] > float(Ligand[i].min()) - distance_cutoff]
    Target = Target.merge(_G['atom_map'], on='ATOM_TYPE', how='left')

    mol_len = torch.tensor(len(Ligand))
    atomicnums = torch.tensor(np.append(np.ones(mol_len) * 6, Target["ATOM_NR"]),
                              dtype=torch.int64).unsqueeze(0)
    coordinates = torch.tensor(pd.concat([Ligand[['X', 'Y', 'Z']],
                                          Target[['X', 'Y', 'Z']]]).values).unsqueeze(0)
    sc = _G['sc']((atomicnums, coordinates))
    aev = _G['aevc'].forward((sc.species, sc.coordinates), mol_len)
    return Ligand, aev.aevs.squeeze(0)[:mol_len, :AEV_DIM]


# ────────────────────────────── 配体读取 ──────────────────────────────

def read_ligand(path, fmt, idx=-1):
    if fmt == 'mol2':
        m = Chem.MolFromMol2File(path)
        if m is None:
            # PDBbind 的部分 mol2 有 RDKit 无法解析的原子类型/价态（如 3ip8、2fzc），
            # 官方发布同时提供 .sdf，回退读它 —— 同一构象、同一坐标，不引入任何差异。
            alt = path[:-5] + '.sdf' if path.endswith('.mol2') else None
            if alt and os.path.exists(alt):
                return read_ligand(alt, 'sdf', 0)
            return None
        return Chem.AddHs(m, addCoords=True)
    suppl = Chem.SDMolSupplier(path, removeHs=False)
    m = suppl[idx if idx >= 0 else 0]
    if m is None:
        return None
    if not any(a.GetSymbol() == 'H' for a in m.GetAtoms()):
        m = Chem.AddHs(m, addCoords=True)
    return m


def work(t):
    uid, prot, lig, fmt, idx = t
    try:
        mol = read_ligand(lig, fmt, idx)
        if mol is None:
            return uid, None, 'ligand_unreadable'
        mol_df, aevs = GetMolAEVs_extended(prot, mol)
        n, feats, ei, ea = mol_to_graph(mol, mol_df, aevs)
        if len(feats) == 0 or len(ei) == 0:
            return uid, None, 'empty_graph'
        x = np.asarray(feats, dtype=np.float32)
        if x.shape[1] != NODE_DIM:
            return uid, None, f'bad_dim_{x.shape[1]}'
        return uid, (x, np.asarray(ei, dtype=np.int32).T,
                     np.asarray(ea, dtype=np.uint8)), None
    except Exception as e:
        return uid, None, f'{type(e).__name__}: {e}'[:200]


# ──────────────────────────── 各源清单 ────────────────────────────

def manifest(src, limit=0):
    E = f'{ROOT}/experiments'
    if src == 'pdbbind':
        d = pd.read_csv(f'{E}/01_baseline_index/pdbbind_clean.csv')
        rows = [(p, f'{DS}/pdbbind/{"refined-set" if r else "general-set"}/{p}/{p}_protein.pdb',
                 f'{DS}/pdbbind/{"refined-set" if r else "general-set"}/{p}/{p}_ligand.mol2',
                 'mol2', -1) for p, r in zip(d.pdb_id, d.refined)]
    elif src == 'bindingnet':
        d = pd.read_csv(f'{E}/01_baseline_index/bindingnet_clean.csv')
        B = f'{DS}/bindingnet/bindingnet_database/from_chembl_client'
        rows = [(u, f'{B}/{p}/rec_h_opt.pdb', f'{B}/{p}/target_{t}/{c}/{p}_{t}_{c}.sdf',
                 'sdf', -1)
                for u, t, p, c in zip(d.unique_id, d.target_chembl, d.pdb_id, d.ligand_chembl)]
    elif src == 'bindingdb_dcs':
        d = pd.read_csv(f'{E}/01_baseline_index/bindingdb_dcs_clean.csv')
        rows = [(u, f'{DS}/bindingdb_dcs/{f}/{p}.pdb', f'{DS}/bindingdb_dcs/{f}/{m}',
                 'mol2', -1)
                for u, f, p, m in zip(d.unique_id, d.folder, d.pdb, d.mol2_file)]
    elif src == 'hiqbind':
        d = pd.read_csv(f'{E}/02_hiqbind/hiqbind_dedup.csv')
        rows = list(zip(d.unique_id,
                        [f'{ROOT}/{x}' for x in d.protein_pdb],
                        [f'{ROOT}/{x}' for x in d.ligand_sdf],
                        ['sdf'] * len(d), [-1] * len(d)))
    elif src == 'gatoraffinity':
        d = pd.read_csv(f'{E}/03_gator/gator_dedup.csv')
        rows = list(zip(d.unique_id,
                        [f'{ROOT}/{x}' for x in d.protein_pdb],
                        [f'{ROOT}/{x}' for x in d.ligand_sdf],
                        ['sdf'] * len(d), [-1] * len(d)))
    elif src == 'ood295':
        d = pd.read_csv(f'{META}/testset_ood295.csv')
        rows = [(p, f'{DS}/pdbbind/refined-set/{p}/{p}_protein.pdb',
                 f'{DS}/pdbbind/refined-set/{p}/{p}_ligand.mol2', 'mol2', -1)
                for p in d.pdb_id]
    elif src == 'fep42':
        d = pd.read_csv(f'{META}/testset_fep42.csv')
        rows = [(f'{s}__{i}', f'{ROOT}/{pp}', f'{ROOT}/{ls}', 'sdf', int(i))
                for s, i, pp, ls in zip(d.series, d.lig_idx, d.protein_pdb, d.ligand_sdf)]
    else:
        raise ValueError(src)
    # 同一 unique_id 只算一次（DCS/HiQBind 理论上唯一，防御性去重）
    seen, uniq = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    uniq = uniq[:limit] if limit else uniq
    # 按蛋白路径排序，让同一蛋白的配体落进同一个 chunk，命中 _load_pdb_cached
    uniq.sort(key=lambda r: (r[1], r[0]))
    return uniq


def run_source(src, nproc, limit=0):
    from multiprocessing import Pool
    rows = manifest(src, limit)
    log(f'\n[{src}] {len(rows)} 个复合物')
    res, fails = {}, []
    with Pool(nproc, initializer=_init_worker) as p:
        for uid, g, err in tqdm(p.imap_unordered(work, rows, chunksize=8),
                                total=len(rows), desc=src, ncols=100,
                                mininterval=2.0):
            if g is None:
                fails.append((uid, err))
            else:
                res[uid] = g

    ids = [r[0] for r in rows if r[0] in res]
    xs = [res[i][0] for i in ids]
    eis = [res[i][1] for i in ids]
    eas = [res[i][2] for i in ids]
    node_ptr = np.concatenate([[0], np.cumsum([x.shape[0] for x in xs])]).astype(np.int64)
    edge_ptr = np.concatenate([[0], np.cumsum([e.shape[1] for e in eis])]).astype(np.int64)
    # 存成**独立 .npy**（不是 .npz）—— np.load 的 mmap_mode 对 zip 容器无效，
    # 分开存才能真正内存映射，5 个并发训练进程共享同一份页缓存。
    d = f'{OUT}/{src}'
    os.makedirs(d, exist_ok=True)
    np.save(f'{d}/x.npy', np.concatenate(xs, 0) if xs else np.zeros((0, NODE_DIM), np.float32))
    np.save(f'{d}/ei.npy', np.concatenate(eis, 1) if eis else np.zeros((2, 0), np.int32))
    np.save(f'{d}/ea.npy', np.concatenate(eas, 0) if eas else np.zeros((0, 4), np.uint8))
    np.save(f'{d}/node_ptr.npy', node_ptr)
    np.save(f'{d}/edge_ptr.npy', edge_ptr)
    np.save(f'{d}/ids.npy', np.array(ids))
    pd.DataFrame(fails, columns=['unique_id', 'error']).to_csv(
        f'{d}/failed.csv', index=False)
    log(f'[{src}] 成功 {len(ids)} / {len(rows)}，失败 {len(fails)}'
        f'；节点 {node_ptr[-1]}，边 {edge_ptr[-1]}'
        f'，x 体积 {node_ptr[-1] * NODE_DIM * 4 / 2**30:.2f} GiB')
    if fails:
        from collections import Counter
        log(f'   失败原因 top5 {Counter(e for _, e in fails).most_common(5)}')
    return {'n_in': len(rows), 'n_ok': len(ids), 'n_fail': len(fails),
            'n_nodes': int(node_ptr[-1]), 'n_edges': int(edge_ptr[-1])}


SOURCES = ['ood295', 'fep42', 'pdbbind', 'bindingdb_dcs', 'hiqbind',
           'bindingnet', 'gatoraffinity']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='*', default=SOURCES)
    ap.add_argument('--nproc', type=int, default=min(96, os.cpu_count()))
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log(f'并行 {a.nproc}；节点维 {NODE_DIM} = 15 + {AEV_DIM}')
    rep = {}
    for s in a.sources:
        rep[s] = run_source(s, a.nproc, a.limit)
        json.dump(rep, open(f'{OUT}/report_04.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\n产物 → {OUT}/{{<source>/*.npy, <source>/failed.csv, report_04.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
