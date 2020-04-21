import torch
from torch import nn
import torch.nn.functional as F

from . import GatedPixelCNN


class ContrastivePredictiveCoding(nn.Module):
    def __init__(self, in_dim, hidden_dims, lengths, n_directions=4, autoregressor=GatedPixelCNN):
        super().__init__()
        self.lengths = lengths
        self.n_directions = n_directions

        self.autoregressor = autoregressor(in_dim, hidden_dims)

        self.linears = nn.ModuleList([
            nn.ModuleDict({str(k): nn.Linear(hidden_dims[-1], in_dim, bias=False) for k in self.lengths})
            for _ in range(self.n_directions)
        ])

    def _process(self, context, real, direction):
        # BxCxHxW to BxHxWxC
        context = context.permute(0, 2, 3, 1)
        real = real.permute(0, 2, 3, 1)

        loss = 0.

        for length in self.lengths:
            c = context[:, :-length, :, :].reshape(-1, context.shape[-1])
            r = real[:, length:, :, :].reshape(-1, real.shape[-1])
            pred = self.linears[direction][str(length)](c)
            prod = torch.matmul(pred, r.T)
            loss += F.cross_entropy(prod, torch.arange(prod.shape[0], device=prod.device))

        return loss / len(self.lengths)

    def forward(self, encoding):
        loss = 0.

        for direction in range(self.n_directions):
            if direction > 0:
                encoding = encoding.rot90(dims=[2, 3])
            context = self.autoregressor(encoding)
            loss += self._process(context, encoding, direction)

        return loss / self.n_directions