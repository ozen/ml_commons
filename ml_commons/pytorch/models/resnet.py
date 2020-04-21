from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls, Bottleneck, ResNet

from . import StitchedModel


def get_resnet_encoder(in_channels, out_channels=1024, layers=None, pretrained=None, norm_layer=nn.Identity):
    if layers is None:
        layers = [3, 4, 6, 3]
    encoder = ResNet(Bottleneck, layers, norm_layer=norm_layer)

    if pretrained:
        if pretrained not in model_urls:
            raise RuntimeError('No pretrained weights for this model')
        state_dict = load_state_dict_from_url(model_urls[pretrained])
        encoder.load_state_dict(state_dict, strict=False)

    # replace first conv for different number of input channels
    if in_channels != 3:
        encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(encoder.conv1.weight, mode='fan_out', nonlinearity='relu')

    if out_channels == 1024:
        end_layer = -3
    elif out_channels == 2048:
        end_layer = -2
    else:
        raise RuntimeError('Invalid out_channels value')

    return StitchedModel(
        (encoder, 0, end_layer),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1)
    )
