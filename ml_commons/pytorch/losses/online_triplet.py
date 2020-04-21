from itertools import combinations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class OnlineTripletLoss(nn.Module):
    """ Implements semi-hard and hard negative mining strategy """
    def __init__(self, margin, hard=False):
        super().__init__()
        self.margin = margin
        self.hard = hard

    def forward(self, embeddings, labels):
        embeddings_sqr = embeddings.pow(2).sum(dim=1)
        distance_matrix = torch.addmm(1, embeddings_sqr + embeddings_sqr.t(), -2, embeddings, embeddings.t()).cpu()
        labels = labels.cpu().numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]

            if len(label_indices) < 2:
                continue

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positive_pairs = np.array(list(combinations(label_indices, 2)))
            ap_distances = distance_matrix[anchor_positive_pairs[:, 0], anchor_positive_pairs[:, 1]]

            for (anchor_idx, positive_idx), ap_distance in zip(anchor_positive_pairs, ap_distances):
                loss_values = ap_distance - distance_matrix[anchor_idx, negative_indices] + self.margin
                if len(loss_values) > 0:
                    loss_values = loss_values.detach().cpu().numpy()

                    if self.hard:
                        hard_negative_idx = np.argmax(loss_values)
                        if loss_values[hard_negative_idx] > 0:
                            triplets.append([anchor_idx, positive_idx, negative_indices[hard_negative_idx]])
                    else:
                        semihard_negative_indices = np.where(np.logical_and(loss_values < self.margin, loss_values > 0))[0]
                        if len(semihard_negative_indices) > 0:
                            semihard_negative_idx = np.random.choice(semihard_negative_indices)
                            triplets.append([anchor_idx, positive_idx, negative_indices[semihard_negative_idx]])

        if len(triplets) > 0:
            triplets = np.array(triplets)
            return F.triplet_margin_loss(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]],
                                         embeddings[triplets[:, 2]], margin=self.margin)
        else:
            return torch.tensor(0, dtype=torch.float32, device=embeddings.device)
