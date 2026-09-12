#!/usr/bin/env python
"""
阶段 5 数据层：从 04_graphs 的 .npy 内存映射按需构图（供训练与评测共用）。

设计要点：
  · 不把 17 万个 PyG Data 对象常驻内存 —— 用 mmap 读大数组，__getitem__ 现切现建，
    多个并发训练进程共享同一份操作系统页缓存。
  · y 标准化沿用参考实现（StandardScaler，只用训练集拟合），推理时逆变换回 pK。
"""
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = os.environ.get('STACEY_ROOT',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPHS = f'{ROOT}/experiments/04_graphs'
NODE_DIM = 367
EDGE_DIM_RAW = 4


class GraphStore:
    """按源惰性挂载 04_graphs/<src>/*.npy，提供 (src, uid) → 图。"""

    def __init__(self, sources):
        self.a = {}
        self.idx = {}
        for s in sources:
            d = f'{GRAPHS}/{s}'
            self.a[s] = {k: np.load(f'{d}/{k}.npy', mmap_mode='r')
                         for k in ['x', 'ei', 'ea', 'node_ptr', 'edge_ptr']}
            ids = np.load(f'{d}/ids.npy')
            self.idx[s] = {u: i for i, u in enumerate(ids.tolist())}

    def has(self, src, uid):
        return uid in self.idx.get(src, {})

    def available(self, src):
        return set(self.idx.get(src, {}))

    def get(self, src, uid):
        a, i = self.a[src], self.idx[src][uid]
        n0, n1 = int(a['node_ptr'][i]), int(a['node_ptr'][i + 1])
        e0, e1 = int(a['edge_ptr'][i]), int(a['edge_ptr'][i + 1])
        return (np.asarray(a['x'][n0:n1]),
                np.asarray(a['ei'][:, e0:e1]),
                np.asarray(a['ea'][e0:e1]))


class GraphDataset(torch.utils.data.Dataset):
    """items: DataFrame(source, unique_id, y)；y 可为 NaN（纯推理）。"""

    def __init__(self, store, items, y_scaler=None):
        self.store = store
        self.src = items['source'].tolist()
        self.uid = items['unique_id'].tolist()
        self.y = items['y'].to_numpy(dtype=np.float64)
        self.y_scaler = y_scaler

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, i):
        x, ei, ea = self.store.get(self.src[i], self.uid[i])
        y = self.y[i]
        if self.y_scaler is not None:
            y = (y - self.y_scaler[0]) / self.y_scaler[1]
        return Data(x=torch.from_numpy(x.astype(np.float32)),
                    edge_index=torch.from_numpy(ei.astype(np.int64)),
                    edge_attr=torch.from_numpy(ea.astype(np.float32)),
                    y=torch.tensor([y], dtype=torch.float32),
                    idx=torch.tensor([i], dtype=torch.long))


def fit_y_scaler(y):
    """返回 (mean, std)；std 为 0 时退化为 1，避免除零。"""
    m, s = float(np.mean(y)), float(np.std(y))
    return (m, s if s > 1e-12 else 1.0)


def load_config(cid):
    """读配置索引表 → DataFrame(source, unique_id, y)。"""
    p = f'{ROOT}/experiments/05_configs/config_{cid:02d}.csv'
    d = pd.read_csv(p)
    d = d.rename(columns={'pK': 'y'})
    return d[['source', 'unique_id', 'y']]


def config_meta():
    import json
    return json.load(open(f'{ROOT}/experiments/05_configs/configs.json'))


def load_testset(name, store):
    """name ∈ {ood295, fep42}；返回 DataFrame(source, unique_id, y, ...原始列)。"""
    M = f'{ROOT}/experiments/00_meta'
    if name == 'ood295':
        d = pd.read_csv(f'{M}/testset_ood295.csv')
        d['source'], d['unique_id'], d['y'] = 'ood295', d.pdb_id, d.pK
    elif name == 'fep42':
        d = pd.read_csv(f'{M}/testset_fep42.csv')
        d['source'] = 'fep42'
        d['unique_id'] = d.series.astype(str) + '__' + d.lig_idx.astype(str)
        d['y'] = d.pK
    else:
        raise ValueError(name)
    ok = d.unique_id.isin(store.available(name))
    if (~ok).any():
        print(f'  ! {name}: {int((~ok).sum())} 条缺图，已排除', flush=True)
    return d[ok].reset_index(drop=True)
