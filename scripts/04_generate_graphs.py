#!/usr/bin/env python
"""
Stage 4: unified graph feature generation for all seven sources (AEV + PLIG
node/edge features).

Seven sources: pdbbind / bindingnet / bindingdb_dcs / hiqbind / gatoraffinity
               (training side)
               ood295 / fep42 (evaluation side)

**Strictly mirrors the AEV-PLIG reference implementation**
(generate_pdbbind_graphs.py, generate_bindingnet_graphs.py,
generate_bindingdb_graphs.py): the four functions LoadMolasDF / LoadPDBasDF /
atom_features / mol_to_graph are copied verbatim; GetMolAEVs_extended has a
single engineering change — AEVComputer and SpeciesConverter are cached at
process level (the reference implementation rebuilds them for every complex,
which is pure waste); the numerical results are identical.

Unified ANI-2x parameters (consistent with the reference implementation):
  RcR=5.1, EtaR=[19.7], RsR=16 bins 0.80…4.83, RcA=2.0,
  Zeta=TsA=EtaA=RsA=[1.0], 22 atom types, **radial terms only** → AEV width
  22×16 = 352.
  Node features = 15-dim PLIG features + 352-dim AEV = 367; edge features =
  4-dim bond-type one-hot.

Ligand reading rules (cover both conventions of the reference implementation
without introducing a new one):
  · mol2: Chem.MolFromMol2File + AddHs(addCoords=True)   — as in the pdbbind /
          bindingdb reference
  · sdf : SDMolSupplier(removeHs=False)                   — as in the
          bindingnet reference
          If the molecule read **has no explicit hydrogens**,
          AddHs(addCoords=True) is applied as well.
          Reason: the total_num_Hs node feature counts hydrogen neighbors;
          without hydrogens it is constantly 0, i.e. a corrupted feature.
          BindingNet's sdf files already contain hydrogens, so this branch has
          no effect on reproducing the reference implementation.

Storage format (no pickled dict — for 173k complexes, both the Python object
overhead and the load time are unacceptable):
  graphs/{source}/*.npy
    x        float32 (total nodes, 367)  node features of all graphs, stacked
                                         vertically
    node_ptr int64   (N+1,)              nodes of graph i =
                                         x[node_ptr[i]:node_ptr[i+1]]
    ei       int32   (2, total edges)    local indices within each graph
    ea       uint8   (total edges, 4)    bond-type one-hot
    edge_ptr int64   (N+1,)
    ids      <U…     (N,)                unique_id, aligned with index table
  At training time the arrays are read lazily via np.load(mmap_mode='r') and
  sliced into PyG Data objects, so memory use is of the same order as a batch.

Usage:
  python 04_generate_graphs.py                 # all seven sources
  python 04_generate_graphs.py --sources fep42 ood295
  python 04_generate_graphs.py --limit 200     # smoke test: 200 per source
"""
import os, sys, json, argparse, pickle

# BLAS/OpenMP thread counts must be limited before importing numpy / torch:
# this script relies on multiprocessing, and each process opening N math
# library threads on top causes severe oversubscription
# (measured with 96 processes: per-entry time inflated 11x, throughput dropped
# from 26 it/s to 2.3 it/s)
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
import torch

# torchani_mod is the modified TorchANI inside the AEV-PLIG repo (it provides
# the atom-pair AEV output of aev_computer); it is not installed in
# site-packages, so the repo root must be added to sys.path first
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

# ANI-2x radial parameters
RcR = 5.1
EtaR = torch.tensor([19.7])
RsR = torch.tensor([0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
                    2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83])
RADIAL_COEFS = [RcR, EtaR, RsR]
N_ATOM_TYPES = 22
AEV_DIM = N_ATOM_TYPES * len(EtaR) * len(RsR)     # 352
NODE_DIM = 15 + AEV_DIM                           # 367

_G = {}          # process-level cache


# ══ Next four functions copied verbatim from generate_pdbbind_graphs.py ══

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


# ══ AEV computation: numerically identical to the reference implementation; ══
# ══ only the two computers are cached at process level                      ══

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
    """The same protein is often reused by many ligands of one congeneric
    series (FEP set: 42 protein structures / 1087 ligands; BindingNet: 5816
    proteins / 67989 ligands). Parsing one takes about 0.5 s, 14% of the
    total per-entry time. The manifest is already sorted by protein path
    with chunksize=8, so the same worker gets consecutive hits and a
    single-entry cache suffices."""
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


# ────────────────────────────── Ligand reading ──────────────────────────────

def read_ligand(path, fmt, idx=-1):
    if fmt == 'mol2':
        m = Chem.MolFromMol2File(path)
        if m is None:
            # Some PDBbind mol2 files have atom types/valences RDKit cannot
            # parse (e.g. 3ip8, 2fzc). The official release also ships an
            # .sdf, so fall back to it — same conformer, same coordinates, no
            # difference introduced.
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


# ────────────────────────── Per-source manifests ──────────────────────────

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
    # Count each unique_id only once (DCS/HiQBind should be unique in theory;
    # defensive deduplication)
    seen, uniq = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0]); uniq.append(r)
    uniq = uniq[:limit] if limit else uniq
    # Sort by protein path so ligands of the same protein land in the same
    # chunk and hit _load_pdb_cached
    uniq.sort(key=lambda r: (r[1], r[0]))
    return uniq


def run_source(src, nproc, limit=0):
    from multiprocessing import Pool
    rows = manifest(src, limit)
    log(f'\n[{src}] {len(rows)} complexes')
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
    # Save as **separate .npy files** (not .npz) — np.load's mmap_mode has no
    # effect on zip containers; only separate files can truly be memory-mapped,
    # letting 5 concurrent training processes share the same page cache.
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
    log(f'[{src}] succeeded {len(ids)} / {len(rows)}, failed {len(fails)}'
        f'; nodes {node_ptr[-1]}, edges {edge_ptr[-1]}'
        f', x size {node_ptr[-1] * NODE_DIM * 4 / 2**30:.2f} GiB')
    if fails:
        from collections import Counter
        log(f'   top5 errors {Counter(e for _, e in fails).most_common(5)}')
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
    log(f'processes {a.nproc}; node dim {NODE_DIM} = 15 + {AEV_DIM}')
    rep = {}
    for s in a.sources:
        rep[s] = run_source(s, a.nproc, a.limit)
        json.dump(rep, open(f'{OUT}/report_04.json', 'w'), indent=1, ensure_ascii=False)
    log(f'\nOutputs → {OUT}/{{<source>/*.npy, <source>/failed.csv, '
        f'report_04.json}}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
