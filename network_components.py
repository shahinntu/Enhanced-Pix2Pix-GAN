import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_dropout=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(output_channels)
        self.batchnorm2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        return x


class DownSamplingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_dropout=False):
        super().__init__()
        self.double_conv = DoubleConvBlock(
            input_channels, output_channels, use_dropout=use_dropout
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.maxpool(x)

        return x


class UpSamplingBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_dropout=False):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(
            input_channels, input_channels // 2, kernel_size=2, stride=2
        )
        self.double_conv = DoubleConvBlock(
            input_channels, output_channels, use_dropout=use_dropout
        )

    def forward(self, x, skip_con_x):
        x = self.trans_conv(x)
        assert (
            x.shape == skip_con_x.shape
        ), "Upsampled and skip connection shapes must match"
        x = torch.cat([x, skip_con_x], dim=1)
        x = self.double_conv(x)

        return x


class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
