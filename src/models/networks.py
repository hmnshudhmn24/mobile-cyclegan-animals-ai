
import torch
import torch.nn as nn
from typing import List

# -------------------------------
# ResNet Generator (CycleGAN-like)
# -------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim: int, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        p = 0
        if padding_type == 'reflect':
            self.pad1 = nn.ReflectionPad2d(1)
            self.pad2 = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.pad1 = nn.ReplicationPad2d(1)
            self.pad2 = nn.ReplicationPad2d(1)
        else:
            p = 1
            self.pad1 = self.pad2 = nn.Identity()

        block = []
        block += [self.pad1,
                  nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
                  norm_layer(dim),
                  nn.ReLU(True)]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        block += [self.pad2,
                  nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
                  norm_layer(dim)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = x + self.block(x)
        return out


class ResnetGenerator(nn.Module):
    """
    ResNet-9 blocks generator used in CycleGAN.
    c7s1-64, d128, d256, R256 x 9, u128, u64, c7s1-3
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        assert(n_blocks >= 0)
        model: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            norm_layer(ngf),
            nn.ReLU(True)
        ]
        # Downsample
        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]
            mult *= 2
        # ResNet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type='reflect', norm_layer=norm_layer, use_dropout=False)]
        # Upsample
        for i in range(n_downsampling):
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
            mult = int(mult / 2)
        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# -------------------------------
# PatchGAN Discriminator (lite)
# -------------------------------

def _block(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
    if norm:
        layers += [nn.InstanceNorm2d(out_c)]
    layers += [nn.LeakyReLU(0.2, True)]
    return layers

class NLayerDiscriminator(nn.Module):
    """"70x70 PatchGAN (simplified)"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4; padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 1, padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)]  # no sigmoid (use BCEWithLogits)
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
