#!/usr/bin/env python
"""
Model definition: a verbatim copy of GATv2Net (model_defs.py) and init_weights
(utils.py) from the AEV-PLIG reference implementation.

Upstream source
    Valsson et al., Commun. Chem. 8:41 (2025)
    https://github.com/isakvals/AEV-PLIG
    3-Clause BSD License, Copyright (c) Isak Valsson,
    Oxford Protein Informatics Group, 2024.

Hyperparameters follow the upstream defaults in training.py parse_args()
(hidden_dim=256, head=3, activation_function='leaky_relu'), not any
alternative setting. The resulting architecture is:

    GNN layers        5
    Hidden dim        256
    Attention heads   3
    Activation        LeakyReLU
    Dropout           none
    Normalisation     BatchNorm after every layer, applied post-activation
    Pooling           cat[global_max, global_mean]
    Edge features     4-dim bond-type one-hot, used directly
    Output head       1536 -> 1024 -> 512 -> 256 -> 1,
                      BatchNorm after each layer

init_weights is copied verbatim, including the upstream `if layer.bias is True`
check. That condition evaluates to False for any Tensor, so biases retain
PyTorch default initialisation rather than being zeroed. This is left unchanged
on purpose: altering it would no longer reproduce the reference implementation.

"""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import BatchNorm

NODE_DIM = 367          # 15 维 PLIG + 352 维 AEV
EDGE_DIM = 4            # 键型 one-hot [single, aromatic, double, triple]

# training.py parse_args() 的默认值
DEFAULT_CONFIG = SimpleNamespace(hidden_dim=256, head=3,
                                 activation_function='leaky_relu')

activation_function_dict = {"relu": F.relu,
                            "leaky_relu": F.leaky_relu}


class GATv2Net(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, config):
        super(GATv2Net, self).__init__()

        self.number_GNN_layers = 5
        self.act = config.activation_function
        self.activation = activation_function_dict[self.act]

        self.GNN_layers = nn.ModuleList()
        self.BN_layers = nn.ModuleList()

        input_dim = node_feature_dim
        head = config.head
        hidden_dim = config.hidden_dim

        self.GNN_layers.append(GATv2Conv(input_dim, hidden_dim, heads=head,
                                         edge_dim=edge_feature_dim))
        self.BN_layers.append(BatchNorm(hidden_dim * head))
        for i in range(1, self.number_GNN_layers):
            self.GNN_layers.append(GATv2Conv(hidden_dim * head, hidden_dim,
                                             heads=head, edge_dim=edge_feature_dim))
            self.BN_layers.append(BatchNorm(hidden_dim * head))

        final_dim = hidden_dim * head

        # MLPs
        self.fc1 = nn.Linear(final_dim * 2, 1024)
        self.bn_connect1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_connect2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn_connect3 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (data.x, data.edge_index,
                                           data.edge_attr, data.batch)

        for i, (layer, bn) in enumerate(zip(self.GNN_layers, self.BN_layers)):
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = bn(x)

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # add some dense layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn_connect1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.bn_connect2(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.bn_connect3(x)
        return self.out(x)          # (B, 1) —— 与参考实现一致，调用方负责 view


def init_weights(layer):
    """逐字复制 utils.init_weights（含 `layer.bias is True` 的原样行为）。"""
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is True:
            torch.nn.init.zeros_(layer.bias)


def build_model(node_dim=NODE_DIM, edge_dim=EDGE_DIM, config=None):
    """按参考实现 training.py 的方式建模并初始化。"""
    m = GATv2Net(node_dim, edge_dim, config or DEFAULT_CONFIG)
    m.apply(init_weights)
    return m


AEVPLIG = GATv2Net      # 兼容旧脚本里的名字


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = build_model()
    print(f'GATv2Net(node={NODE_DIM}, edge={EDGE_DIM}, '
          f'hidden={DEFAULT_CONFIG.hidden_dim}, head={DEFAULT_CONFIG.head}, '
          f'layers={m.number_GNN_layers}, act={m.act})')
    print(f'参数量 {n_params(m):,}')
