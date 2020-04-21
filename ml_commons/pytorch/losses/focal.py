from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, labels):
        loss_ce = F.cross_entropy(logits, labels)
        p_t = F.softmax(logits, dim=1)[range(labels.shape[0]), labels]
        loss = (1 - p_t) ** self.gamma * loss_ce
        return loss.mean()
