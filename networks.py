import torch
import torch.nn as nn

from network_components import FeatureMapBlock, DownSamplingBlock, UpSamplingBlock


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super().__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.downsample1 = DownSamplingBlock(
            hidden_channels, hidden_channels * 2, use_dropout=True
        )
        self.downsample2 = DownSamplingBlock(
            hidden_channels * 2, hidden_channels * 4, use_dropout=True
        )
        self.downsample3 = DownSamplingBlock(
            hidden_channels * 4, hidden_channels * 8, use_dropout=True
        )
        self.downsample4 = DownSamplingBlock(hidden_channels * 8, hidden_channels * 16)
        self.downsample5 = DownSamplingBlock(hidden_channels * 16, hidden_channels * 32)
        self.downsample6 = DownSamplingBlock(hidden_channels * 32, hidden_channels * 64)
        self.downsample7 = DownSamplingBlock(
            hidden_channels * 64, hidden_channels * 128
        )
        self.upsample0 = UpSamplingBlock(hidden_channels * 128, hidden_channels * 64)
        self.upsample1 = UpSamplingBlock(hidden_channels * 64, hidden_channels * 32)
        self.upsample2 = UpSamplingBlock(hidden_channels * 32, hidden_channels * 16)
        self.upsample3 = UpSamplingBlock(hidden_channels * 16, hidden_channels * 8)
        self.upsample4 = UpSamplingBlock(hidden_channels * 8, hidden_channels * 4)
        self.upsample5 = UpSamplingBlock(hidden_channels * 4, hidden_channels * 2)
        self.upsample6 = UpSamplingBlock(hidden_channels * 2, hidden_channels)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.downsample1(x0)
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)
        x4 = self.downsample4(x3)
        x5 = self.downsample5(x4)
        x6 = self.downsample6(x5)
        x7 = self.downsample7(x6)
        x8 = self.upsample0(x7, x6)
        x9 = self.upsample1(x8, x5)
        x10 = self.upsample2(x9, x4)
        x11 = self.upsample3(x10, x3)
        x12 = self.upsample4(x11, x2)
        x13 = self.upsample5(x12, x1)
        x14 = self.upsample6(x13, x0)
        xn = self.downfeature(x14)
        return self.sigmoid(xn)


class Critic(nn.Module):
    def __init__(self, input_channels, num_down_blocks, hidden_channels=32):
        super().__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.downsampling_blocks = nn.Sequential(
            *[
                DownSamplingBlock(
                    hidden_channels * 2**i, hidden_channels * 2 ** (i + 1)
                )
                for i in range(num_down_blocks)
            ]
        )
        self.final = nn.Conv2d(hidden_channels * 2**num_down_blocks, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x = self.upfeature(x)
        x = self.downsampling_blocks(x)
        x = self.final(x)
        return x
