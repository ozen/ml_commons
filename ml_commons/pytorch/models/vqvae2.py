from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """
    https://github.com/zalandoresearch/pytorch-vq-vae
    """
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        """
        :param inputs: BxCxHxW tensor
        :return: (indices, quantized, perplexity)
        indices: BxHxW tensor of indices
        quantized: BxCxHxW tensor of quantized version of the inputs
        perplexity: floating point value of perplexity
        """
        # BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        codebook_sqr = torch.sum(self._embedding.weight ** 2, dim=1)
        inputs_sqr = torch.sum(flat_input ** 2, dim=1, keepdim=True)
        distances = torch.addmm(codebook_sqr + inputs_sqr,
                                flat_input, self._embedding.weight.t(),
                                alpha=-2.0, beta=1.0)

        # Encoding
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # NxK matrix, 1 means nth sample belongs to kth cluster
        codebook_matrix = torch.zeros(indices.shape[0], self._num_embeddings, device=indices.device)
        codebook_matrix.scatter_(1, indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._decay * self._ema_cluster_size + \
                                     (1 - self._decay) * torch.sum(codebook_matrix, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(codebook_matrix.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Quantize and unflatten
        quantized = torch.matmul(codebook_matrix, self._embedding.weight).view(input_shape)     # BHWC
        indices = indices.view(input_shape[:-1])      # BHW

        # Stop backpropagation to codebook but not to inputs (I guess)
        quantized = inputs + (quantized - inputs).detach()

        # Calculate perplexity
        avg_probs = torch.mean(codebook_matrix, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return indices, quantized, perplexity

    def lookup(self, indices):
        return self._embedding(indices).detach()

    def get_centroids(self):
        return self._embedding.weight.detach()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


class ResBlock(nn.Module):
    def __init__(self, in_dim, dim, stride=1):
        super().__init__()

        self.in_dim = in_dim
        self.dim = dim
        self.stride = stride

        self.block = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
        )

        self.activation = nn.ReLU(True)

        self.identity = nn.Sequential(
            nn.Conv2d(in_dim, dim, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = self.block(x)

        identity = x
        if self.stride != 1 or self.in_dim != self.dim:
            identity = self.identity(x)

        out += identity
        out = self.activation(out)
        return out


class TransposeResBlock(nn.Module):
    """ Always upsamples by scale of 2 """
    def __init__(self, in_dim, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
        )

        self.activation = nn.ReLU(True)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_dim, dim, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        out = self.block(x)
        identity = self.upsample(x)
        out += identity
        out = self.activation(out)
        return out


class VQVAE2(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_size):
        super().__init__()

        self.levels = 2

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(True),

            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(True),

            ResBlock(hidden_dim // 2, hidden_dim, stride=2),
            ResBlock(hidden_dim, hidden_dim),
        )

        self.encoder2 = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim, stride=2),
            ResBlock(hidden_dim, hidden_dim, stride=2),
        )

        self.vq2 = VectorQuantizerEMA(codebook_size, hidden_dim)

        self.decoder2 = nn.Sequential(
            TransposeResBlock(hidden_dim, hidden_dim),
            TransposeResBlock(hidden_dim, hidden_dim),
        )

        self.combine1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )

        self.vq1 = VectorQuantizerEMA(codebook_size, hidden_dim)

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )

        self.combine2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim),
            TransposeResBlock(hidden_dim, hidden_dim // 2),

            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim // 4, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def encode(self, x: torch.Tensor):
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(encoded1)
        indices2, quantized2, _ = self.vq2(encoded2)
        decoded2 = self.decoder2(quantized2)

        encoded1 = self.combine1(torch.cat([encoded1, decoded2], dim=1))
        indices1, _, _ = self.vq1(encoded1)

        return (encoded1, encoded2), (indices1, indices2)

    def decode(self, indices: Tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(encoded1)
        _, quantized2, _ = self.vq2(encoded2)
        decoded2 = self.decoder2(quantized2)

        encoded1 = self.combine1(torch.cat([encoded1, decoded2], dim=1))
        _, quantized1, _ = self.vq1(encoded1)

        upsampled2 = self.upsample2(quantized2)
        combined = self.combine2(torch.cat([quantized1, upsampled2], dim=1))
        x_recon = self.decoder(combined)

        return x_recon, (encoded1, encoded2), (quantized1, quantized2)


class VQVAE2_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_size):
        super().__init__()

        self.levels = 2

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(True),

            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(True),

            ResBlock(hidden_dim // 2, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
        )

        self.encoder2 = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim, stride=2),
            ResBlock(hidden_dim, hidden_dim),
        )

        self.vq2 = VectorQuantizerEMA(codebook_size, hidden_dim)

        self.decoder2 = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim),
            TransposeResBlock(hidden_dim, hidden_dim),
        )

        self.combine1 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )

        self.vq1 = VectorQuantizerEMA(codebook_size, hidden_dim)

        self.combine2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            ResBlock(hidden_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim // 2),

            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim // 4, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def encode(self, x: torch.Tensor):
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(encoded1)
        indices2, quantized2, _ = self.vq2(encoded2)
        decoded2 = self.decoder2(quantized2)

        encoded1 = self.combine1(torch.cat([encoded1, decoded2], dim=1))
        indices1, _, _ = self.vq1(encoded1)

        return (encoded1, encoded2), (indices1, indices2)

    def decode(self, indices: Tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        encoded1 = self.encoder1(x)

        encoded2 = self.encoder2(encoded1)
        _, quantized2, _ = self.vq2(encoded2)
        decoded2 = self.decoder2(quantized2)

        encoded1 = self.combine1(torch.cat([encoded1, decoded2], dim=1))
        _, quantized1, _ = self.vq1(encoded1)

        upsampled2 = F.interpolate(quantized2, scale_factor=2)
        combined = self.combine2(torch.cat([quantized1, upsampled2], dim=1))
        x_recon = self.decoder(combined)

        return x_recon, (encoded1, encoded2), (quantized1, quantized2)
