"""Blur Magnitude Estimator (BME) that takes an RGB frame and an event count map.

Architecture from DADeblur (He et al., ECCV 2024) with the first convolution widened to four input
channels (RGB + event count map C_E), as described in Sec. 3.1 of the EvBS paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _resnet50_backbone(in_channels=4):
    net = torchvision.models.resnet50(weights=None)
    net.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    children = list(net.children())
    div_2 = nn.Sequential(*children[:3])     # conv1, bn1, relu
    div_4 = nn.Sequential(*children[3:5])    # maxpool, layer1
    return div_2, div_4, net.layer2, net.layer3, net.layer4


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            self.activation,
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            self.activation,
        )

    def forward(self, x):
        return self.main(x) + x


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.main = nn.Sequential(*[ResBlock(out_channel, out_channel) for _ in range(5)])

    def forward(self, x):
        return self.main(x)


def _pad_to(x, ref):
    diff_y = ref.size(2) - x.size(2)
    diff_x = ref.size(3) - x.size(3)
    return F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])


class CM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.Up = nn.Sequential(nn.ConvTranspose2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel * 2, out_channel, kernel_size=1), nn.LeakyReLU(0.2, True))

    def forward(self, x, y):
        x = _pad_to(self.Up(x), y)
        return self.conv(torch.cat((x, y), dim=1))


class AFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2048, 64, kernel_size=3, padding=1)

        self.conv0_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv1_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv2_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv3_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv4_1x1 = nn.Conv2d(192, 192, kernel_size=1)

        self.conv0_out = nn.Conv2d(192, 32, kernel_size=3, padding=1)
        self.conv1_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv2_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv3_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv4_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)

    def forward(self, x0, x1, x2, x3, x4):
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x0_out = self.conv0_out(self.conv0_1x1(torch.cat(
            (x0, F.interpolate(x1, scale_factor=2), F.interpolate(x2, scale_factor=4)), dim=1)))
        x1_out = self.conv1_out(self.conv1_1x1(torch.cat(
            (F.interpolate(x0, scale_factor=0.5), x1, F.interpolate(x2, scale_factor=2)), dim=1)))
        x2_out = self.conv2_out(self.conv2_1x1(torch.cat(
            (F.interpolate(x1, scale_factor=0.5), x2, F.interpolate(x3, scale_factor=2)), dim=1)))

        x4_up = F.interpolate(x4, scale_factor=2)
        x2_down = _pad_to(F.interpolate(x2, scale_factor=0.5), x4_up)
        x3 = _pad_to(x3, x4_up)
        x3_out = self.conv3_out(self.conv3_1x1(torch.cat((x2_down, x3, x4_up), dim=1)))

        x2_down = _pad_to(F.interpolate(x2, scale_factor=0.25), x4)
        x3_down = F.interpolate(x3, scale_factor=0.5)
        x4_out = self.conv4_out(self.conv4_1x1(torch.cat((x2_down, x3_down, x4), dim=1)))
        return x0_out, x1_out, x2_out, x3_out, x4_out


class BlurMagnitudeEstimator(nn.Module):
    """Predicts a per-pixel blur magnitude map from cat([RGB / 255, C_E / count_norm])."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = _resnet50_backbone(in_channels)
        self.aff = AFF()

        self.decoder_0 = Decoder(32, 32)
        self.decoder_1 = Decoder(64, 32)
        self.decoder_2 = Decoder(64, 64)
        self.decoder_3 = Decoder(64, 64)
        self.decoder_4 = Decoder(2048, 64)

        self.cm_0 = CM(32, 32)
        self.cm_1 = CM(64, 32)
        self.cm_2 = CM(64, 64)
        self.cm_3 = CM(64, 64)

        self.Up = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        h_pad, w_pad = (32 - h % 32) % 32, (32 - w % 32) % 32
        if h_pad or w_pad:
            x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')

        d2 = self.div_2(x)
        d4 = self.div_4(d2)
        d8 = self.div_8(d4)
        d16 = self.div_16(d8)
        d32 = self.div_32(d16)
        a0, a1, a2, a3, a4 = self.aff(d2, d4, d8, d16, d32)

        y = self.decoder_4(a4)
        y = self.decoder_3(self.cm_3(y, a3))
        y = self.decoder_2(self.cm_2(y, a2))
        y = self.decoder_1(self.cm_1(y, a1))
        y = self.decoder_0(self.cm_0(y, a0))
        out = self.classifier(self.Up(y))
        return out[..., :h, :w]
