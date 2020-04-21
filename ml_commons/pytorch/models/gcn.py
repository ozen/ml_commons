import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv = GCNConv(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(True)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        if self.use_bn:
            data.x = self.bn(data.x)
        data.x = self.relu(data.x)
        return data


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, use_bn=True):
        assert isinstance(hidden_dims, list) or isinstance(hidden_dims, tuple)
        assert len(hidden_dims) >= 1
        super().__init__()

        dims = [input_dim] + list(hidden_dims)
        blocks = [GCNBlock(dims[i], dims[i+1], use_bn) for i in range(len(dims) - 1)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, data):
        return self.blocks(data)


class GCNHead(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.conv = GCNConv(dim, n_classes)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        return global_mean_pool(data.x, data.batch)


class GCNNet(nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dim=16, n_blocks=1):
        super().__init__()

        blocks = [GCNBlock(input_dim, hidden_dim)] + \
                 [GCNBlock(hidden_dim, hidden_dim) for i in range(n_blocks - 1)]
        self.blocks = nn.Sequential(*blocks)
        self.conv = GCNConv(hidden_dim, n_classes)

    def forward(self, data):
        data = self.blocks(data)
        data.x = self.conv(data.x, data.edge_index)
        return global_mean_pool(data.x, data.batch)

    def encode(self, data):
        data = self.blocks(data)
        if isinstance(data, Batch):
            batch = data.batch
        else:
            batch = torch.zeros((len(data.x),), dtype=torch.long, device=data.x.device)
        data.x = global_mean_pool(data.x, batch)
        return data
