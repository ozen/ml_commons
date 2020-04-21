import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class MaskedConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='B', vertical=False, **kwargs):
        kwargs['bias'] = False
        if isinstance(kernel_size, int):
            kwargs['padding'] = kernel_size // 2
        else:
            kwargs['padding'] = (kernel_size[0] // 2, kernel_size[1] // 2)

        super(MaskedConv2D, self).__init__(in_channels, out_channels, kernel_size, **kwargs)

        mask = np.ones_like(self.weight.data.cpu().numpy()).astype(np.float32)
        hc, wc = self.weight.shape[2] // 2, self.weight.shape[3] // 2

        if vertical:
            if mask_type == 'A':
                mask[:, :, hc:, :] = 0.0
            else:
                mask[:, :, hc+1:, :] = 0.0
        else:
            mask[:, :, hc+1:, :] = 0.0
            mask[:, :, hc, wc+1:] = 0.0

        self.register_buffer("mask", torch.from_numpy(mask))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConv2D, self).forward(x)


class GatedPixelCNNLayer(nn.Module):
    def __init__(self, mask_type, in_dim, out_dim, kernel, residual=True):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.residual = residual

        self.v_conv = MaskedConv2D(in_dim, 2 * out_dim, vertical=True,
                                   kernel_size=(kernel, kernel), mask_type=mask_type)

        self.v_to_h_conv = nn.Conv2d(2 * out_dim, 2 * out_dim, kernel_size=1)

        self.h_conv = MaskedConv2D(in_dim, 2 * out_dim, vertical=False,
                                   kernel_size=(1, kernel), mask_type=mask_type)

        self.h_out_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1)

        self.gate = GatedActivation()

        self.downsample = None
        if residual and in_dim != out_dim:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x_v, x_h):
        v = self.v_conv(x_v)
        h = self.h_conv(x_h)
        v_shifted_down = F.pad(v[:, :, :-1, :], [0, 0, 1, 0])
        v_to_h = self.v_to_h_conv(v_shifted_down)

        v = self.gate(v)
        h = self.gate(v_to_h + h)
        h = self.h_out_conv(h)

        if self.residual:
            if self.downsample is not None:
                h += self.downsample(x_h)
            else:
                h += x_h

        return v, h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        assert len(hidden_dims) > 1
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GatedPixelCNNLayer('A', input_dim, hidden_dims[0], kernel=7, residual=False))
        for i in range(1, len(hidden_dims)):
            self.layers.append(GatedPixelCNNLayer('B', hidden_dims[i-1], hidden_dims[i], kernel=3, residual=True))

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                try:
                    nn.init.xavier_uniform_(m.weight.data)
                    m.bias.data.fill_(0)
                except AttributeError:
                    pass

        self.apply(weights_init)

    def forward(self, x):
        x_v, x_h = (x, x)
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h)
        return x_h
