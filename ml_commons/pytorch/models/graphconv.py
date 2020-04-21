import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool


class GraphConvNet(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.conv1 = GraphConv(dim, dim)
        self.pool1 = TopKPooling(dim, ratio=0.5)
        self.conv2 = GraphConv(dim, dim)
        self.pool2 = TopKPooling(dim, ratio=0.5)
        self.conv3 = GraphConv(dim, dim)
        self.pool3 = TopKPooling(dim, ratio=0.5)

        self.lin1 = nn.Linear(2*dim, dim)
        self.lin2 = nn.Linear(dim, dim//2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))

        return x
