from math import ceil
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool


class DiffPoolLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 normalize=False, add_loop=False, lin=True):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = []

        self.add_loop = add_loop

        n_channels = [in_channels] + hidden_channels + [out_channels]

        self.expansion = n_channels - 1

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer_i in range(self.expansion):
            self.convs.append(DenseSAGEConv(n_channels[layer_i], n_channels[layer_i+1], normalize))
            self.bns.append(torch.nn.BatchNorm1d(n_channels[layer_i+1]))

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = self.bns[i](x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        xs = []
        for layer_i in range(self.expansion):
            x = self.bn(layer_i, F.relu(self.convs[layer_i](x, adj, mask, self.add_loop)))
            xs.append(x)

        x = torch.cat(xs, dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class DiffPoolNet(nn.Module):
    def __init__(self, input_dim, n_classes, max_nodes, hidden_dims=None, layer_hidden_dims=None, pool_factor=0.5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 64, 64]
        assert len(hidden_dims) == 4

        num_nodes = ceil(pool_factor * max_nodes)
        self.gnn1_pool = DiffPoolLayer(in_channels=input_dim, out_channels=num_nodes,
                                       hidden_channels=layer_hidden_dims, add_loop=True)
        self.gnn1_embed = DiffPoolLayer(in_channels=input_dim, out_channels=hidden_dims[0],
                                        hidden_channels=layer_hidden_dims, add_loop=True, lin=False)

        num_nodes = ceil(pool_factor * num_nodes)
        expansion = self.gnn1_embed.expansion
        self.gnn2_pool = DiffPoolLayer(in_channels=expansion * hidden_dims[0], out_channels=num_nodes,
                                       hidden_channels=layer_hidden_dims)
        self.gnn2_embed = DiffPoolLayer(in_channels=expansion * hidden_dims[0], out_channels=hidden_dims[1],
                                        hidden_channels=layer_hidden_dims, lin=False)

        expansion = self.gnn2_embed.expansion
        self.gnn3_embed = DiffPoolLayer(in_channels=expansion * hidden_dims[1], out_channels=hidden_dims[2],
                                        hidden_channels=layer_hidden_dims, lin=False)

        self.lin1 = torch.nn.Linear(expansion * hidden_dims[2], hidden_dims[3])
        self.lin2 = torch.nn.Linear(hidden_dims[3], n_classes)

    def forward(self, data):
        x = self.encode(data)
        x = self.lin2(x)
        return x

    def encode(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        return x
