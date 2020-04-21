import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import *


class Batched:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        return torch.stack([self.transform(item) for item in batch])


class ImageOnly:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(image=x)['image']


class OnBatchTensor:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        batch = batch.permute(0, 2, 3, 1).numpy()  # BxCxHxW to BxHxWxC
        batch = torch.stack([
            torch.from_numpy(self.transform(image=item)['image']) for item in batch
        ])  # apply HxWxC image transformation
        return batch.permute(0, 3, 1, 2)  # BxHxWxC to BxCxHxW




# class Rotate:
#     def __init__(self, dims, p=1):
#         self.dims = dims
#         self.p = p
#
#     def __call__(self, x):
#         if random.random() > self.p:
#             x = torch.rot90(x, dims=self.dims)
#         return x
#
#
# class Flip:
#     def __init__(self, dims, p=1):
#         self.dims = dims
#         self.p = p
#
#     def __call__(self, x):
#         if random.random() > self.p:
#             x = torch.flip(x, dims=self.dims)
#         return x
#
#
# class GaussianNoise:
#     def __init__(self, std):
#         self.std = std
#
#     def __call__(self, x):
#         factor = torch.normal(1.0, self.std, size=x.shape)
#         factor[factor < 1 - 4 * self.std] = 1.0
#         factor[factor > 1 + 4 * self.std] = 1.0
#         return x * factor
#
#
# class GaussianColorJitter:
#     def __init__(self, dim, std):
#         self.dim = dim
#         self.std = std
#
#     def __call__(self, x):
#         size = (x.shape[self.dim],) if self.dim is not None else (1,)
#         x *= torch.normal(1.0, self.std, size=size)
#         return torch.clamp(x, 0, 1)
#
#
# class RandomCrop:
#     """
#     Operates on CxHxW tensor
#     """
#     def __init__(self, pad):
#         self.pad = pad
#
#     def __call__(self, x):
#         size = x.shape[1:]
#         x = F.pad(x, [self.pad] * 4)
#         h = random.randint(0, 2 * self.pad)
#         w = random.randint(0, 2 * self.pad)
#         return x[:, h:h + size[0], w:w + size[1]]
#
#
# class DropChannels:
#     def __init__(self, dim, n_keep):
#         self.dim = dim
#         self.n = n_keep
#
#     def __call__(self, x):
#         high = x.shape[self.dim]
#         n = min(self.n, high)
#         c = np.random.choice(high, n, replace=False)
#         return torch.index_select(x, self.dim, torch.tensor(c, device=x.device))
