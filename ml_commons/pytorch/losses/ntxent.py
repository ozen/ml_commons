import torch
from torch import nn


class NTXent(nn.Module):
    def __init__(self, temperature=1.0, batch_size=None):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.cos = nn.CosineSimilarity(dim=2)

        self.mask = None
        if batch_size is not None:
            self.mask = self._get_negative_mask(batch_size)

    def forward(self, z_i, z_j):
        batch_size = len(z_i)

        concat = torch.cat((z_i, z_j), dim=0)
        sim = self.cos(concat.unsqueeze(1), concat.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        neg_mask = self.mask
        if neg_mask is None:
            neg_mask = self._get_negative_mask(batch_size)
        neg_mask = neg_mask.to(sim.device)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2 * batch_size, 1)
        negative_samples = sim[neg_mask].reshape(2 * batch_size, -1)

        labels = torch.zeros(2 * batch_size).to(sim.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss

    @staticmethod
    def _get_negative_mask(batch_size):
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
