import torch
from torch import nn
from PNGAN.layers.blur_pool import BlurPool2d


class FCA(nn.Module):
    def __init__(self, k_size=3):
        super(FCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=(k_size, ), padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x) + x


def up_sample(channels, scale=2, k_size=3, padding=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode='bilinear'),
        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(k_size, k_size), stride=(1, 1), padding=(padding, padding))
    )


class MAB(nn.Module):
    def __init__(self, channels):
        super(MAB, self).__init__()
        self.fca1 = FCA()

        self.down2 = BlurPool2d(channels, stride=2)
        self.fca2 = FCA()
        self.up2 = up_sample(channels, 2)

        self.down3 = BlurPool2d(channels, stride=4)
        self.fca3 = FCA()
        self.up3 = up_sample(channels, 4, k_size=5, padding=0)

        self.conv = nn.Conv2d(in_channels=channels * 3, out_channels=channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x1 = self.fca1(x)
        x2 = self.up2(self.fca2(self.down2(x)))
        x3 = self.up3(self.fca3(self.down3(x)))
        return self.conv(torch.cat((x1, x2, x3), 1)) + x


class SRG(nn.Module):
    def __init__(self, in_c, out_c):
        super(SRG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.mba1 = MAB(out_c)
        self.mba2 = MAB(out_c)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.main = nn.Sequential(
            self.conv1, self.mba1, self.mba2, self.conv2
        )

    def forward(self, x):
        return self.main(x) + x


class Generator(nn.Module):
    def __init__(self, in_c=3, out_c=64):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.srg1 = SRG(out_c, out_c)
        self.srg2 = SRG(out_c, out_c)
        self.srg3 = SRG(out_c, out_c)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=in_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.main = nn.Sequential(
            self.conv1, self.srg1, self.srg2, self.srg3, self.conv2
        )

    def forward(self, x):
        return self.main(x) + x
