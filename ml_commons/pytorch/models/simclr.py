import torch
from torch import nn
import torch.nn.functional as F

from ..losses import NTXent


class SimCLR(nn.Module):
    def __init__(self, dim, temperature):
        super(SimCLR, self).__init__()

        self.temperature = temperature

        self.projection_head = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(True),
            nn.Linear(dim, dim, bias=False)
        )

        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, og_encodings, aug_encodings):
        """
        :param og_encodings: NxD original encodings
        :param aug_encodings: NxD augmented encodings
        :return: loss
        """
        og_projection = self.projection_head(og_encodings)
        aug_projection = self.projection_head(aug_encodings)

        sim = self.cos(og_projection.unsqueeze(0), aug_projection.unsqueeze(1))
        sim /= self.temperature
        loss = F.cross_entropy(sim, torch.arange(sim.shape[0], device=sim.device))
        return loss


class SimCLRv2(nn.Module):
    def __init__(self, input_dim, projection_dim, temperature, batch_size=None):
        super(SimCLRv2, self).__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(True),
            nn.Linear(input_dim, projection_dim, bias=False)
        )

        self.criterion = NTXent(temperature, batch_size)

    def forward(self, og_encodings, aug_encodings):
        """
        :param og_encodings: NxD original encodings
        :param aug_encodings: NxD augmented encodings
        :return: loss
        """
        og_projection = self.projection_head(og_encodings)
        aug_projection = self.projection_head(aug_encodings)
        loss = self.criterion(og_projection, aug_projection)
        return loss
