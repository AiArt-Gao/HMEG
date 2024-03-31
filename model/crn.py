import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import get_normalization_2d
from model.layers import get_activation

"""
Cascaded refinement network architecture, as described in:

Qifeng Chen and Vladlen Koltun,
"Photographic Image Synthesis with Cascaded Refinement Networks",
ICCV 2017
"""


class RefinementModule(nn.Module):
    def __init__(self, layout_dim, input_dim, output_dim,
                 normalization='instance', activation='leakyrelu'):
        super(RefinementModule, self).__init__()
        layers = []
        layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                                kernel_size=3, padding=1))
        layers.append(get_normalization_2d(output_dim, normalization))
        layers.append(get_activation(activation))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
        layers.append(get_normalization_2d(output_dim, normalization))
        layers.append(get_activation(activation))
        layers = [layer for layer in layers if layer is not None]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
        self.net = nn.Sequential(*layers)

    def forward(self, layout, feats):
        _, _, HH, WW = layout.size()
        _, _, H, W = feats.size()
        assert HH >= H
        if HH > H:
            factor = round(HH // H)
            assert HH % factor == 0
            assert WW % factor == 0 and WW // factor == W
            layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
        net_input = torch.cat([layout, feats], dim=1)
        out = self.net(net_input)
        return out


class RefinementNetwork(nn.Module):
    def __init__(self, dims, normalization='instance', activation='leakyrelu'):
        super(RefinementNetwork, self).__init__()
        layout_dim = dims[0]
        self.refinement_modules = nn.ModuleList()
        for i in range(1, len(dims)):
            input_dim = 1 if i == 1 else dims[i - 1]
            if i >= len(dims) - 2:
                input_dim += 3
            output_dim = dims[i]
            mod = RefinementModule(layout_dim, input_dim, output_dim,
                                   normalization=normalization, activation=activation)
            self.refinement_modules.append(mod)

        self.output_conv_64 = self._make_output_conv(dims[-3], activation)
        self.output_conv_128 = self._make_output_conv(dims[-2], activation)
        self.output_conv_256 = self._make_output_conv(dims[-1], activation)

    def _make_output_conv(self, dim, activation):
        output_conv_layers = [
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            get_activation(activation),
            nn.Conv2d(dim, 3, kernel_size=1, padding=0)
        ]
        nn.init.kaiming_normal_(output_conv_layers[0].weight)
        nn.init.kaiming_normal_(output_conv_layers[2].weight)
        return nn.Sequential(*output_conv_layers)

    def forward(self, layout):
        """
        Output will have same size as layout
        """
        # H, W = self.output_size
        N, _, H, W = layout.size()
        self.layout = layout

        # Figure out size of input
        input_H, input_W = H, W
        for _ in range(len(self.refinement_modules)):
            input_H //= 2
            input_W //= 2

        assert input_H != 0
        assert input_W != 0

        feats = torch.zeros(N, 1, input_H, input_W).to(layout)
        for mod in self.refinement_modules[:-2]:
            feats = F.upsample(feats, scale_factor=2, mode='nearest')
            feats = mod(layout, feats)

        out_64 = self.output_conv_64(feats)
        feats = torch.cat([feats, out_64], dim=1)
        feats = F.upsample(feats, scale_factor=2, mode='nearest')
        feats = self.refinement_modules[-2](layout, feats)
        out_128 = self.output_conv_128(feats)
        feats = torch.cat([feats, out_128], dim=1)
        feats = F.upsample(feats, scale_factor=2, mode='nearest')
        feats = self.refinement_modules[-1](layout, feats)
        out_256 = self.output_conv_256(feats)
        return out_64, out_128, out_256





