from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, use_batchnorm=False, dropout_rate=0):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.bn = nn.BatchNorm1d(hidden_size) if use_batchnorm else None
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x, True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

