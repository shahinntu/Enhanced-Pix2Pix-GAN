import sys
import unittest

import torch

sys.path.append("../")
from network_components import (
    FeatureMapBlock,
    DoubleConvBlock,
    DownSamplingBlock,
    UpSamplingBlock,
)


class TestDoubleConvBlock(unittest.TestCase):
    def setUp(self):
        self.test_samples = 100
        self.test_in_channels = 10
        self.test_out_channels = 20
        self.test_size = 50

        self.test_block = DoubleConvBlock(self.test_in_channels, self.test_out_channels)
        self.test_in = torch.randn(
            self.test_samples, self.test_in_channels, self.test_size, self.test_size
        )

    def tearDown(self):
        self.test_samples = None
        self.test_in_channels = None
        self.test_out_channels = None
        self.test_size = None

        self.test_block = None
        self.test_in = None

    def test_layers(self):
        test_out_conv1 = self.test_block.conv1(self.test_in)
        self.assertEqual(
            test_out_conv1.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size,
                self.test_size,
            ),
        )

        test_out_conv2 = self.test_block.conv2(test_out_conv1)
        self.assertEqual(
            test_out_conv2.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size,
                self.test_size,
            ),
        )

    def test_out(self):
        test_out = self.test_block(self.test_in)
        self.assertEqual(
            test_out.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size,
                self.test_size,
            ),
        )


class TestDownSamplingBlock(unittest.TestCase):
    def setUp(self):
        self.test_samples = 100
        self.test_in_channels = 10
        self.test_out_channels = 20
        self.test_size = 50

        self.test_block = DownSamplingBlock(
            self.test_in_channels, self.test_out_channels
        )
        self.test_in = torch.randn(
            self.test_samples, self.test_in_channels, self.test_size, self.test_size
        )

    def tearDown(self):
        self.test_samples = None
        self.test_in_channels = None
        self.test_out_channels = None
        self.test_size = None

        self.test_block = None
        self.test_in = None

    def test_layers(self):
        test_out_double_conv = self.test_block.double_conv(self.test_in)
        self.assertEqual(
            test_out_double_conv.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size,
                self.test_size,
            ),
        )

        test_maxpool = self.test_block.maxpool(test_out_double_conv)
        self.assertEqual(
            test_maxpool.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size // 2,
                self.test_size // 2,
            ),
        )

    def test_out(self):
        test_out = self.test_block(self.test_in)
        self.assertEqual(
            test_out.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size // 2,
                self.test_size // 2,
            ),
        )


class TestUpSamplingBlock(unittest.TestCase):
    def setUp(self):
        self.test_samples = 100
        self.test_in_channels = 20
        self.test_out_channels = 10
        self.test_size = 50

        self.test_block = UpSamplingBlock(self.test_in_channels, self.test_out_channels)
        self.skip_con_x = torch.randn(
            self.test_samples,
            self.test_in_channels // 2,
            self.test_size * 2,
            self.test_size * 2,
        )
        self.x = torch.randn(
            self.test_samples, self.test_in_channels, self.test_size, self.test_size
        )

    def tearDown(self):
        self.test_samples = None
        self.test_in_channels = None
        self.test_out_channels = None
        self.test_size = None

        self.test_block = None
        self.skip_con_x = None
        self.x = None

    def test_layers(self):
        test_out_trans_conv = self.test_block.trans_conv(self.x)
        self.assertEqual(
            test_out_trans_conv.shape,
            (
                self.test_samples,
                self.test_in_channels // 2,
                self.test_size * 2,
                self.test_size * 2,
            ),
        )

        concat_x = torch.concat([test_out_trans_conv, self.skip_con_x], axis=1)
        test_out_double_conv = self.test_block.double_conv(concat_x)
        self.assertEqual(
            test_out_double_conv.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size * 2,
                self.test_size * 2,
            ),
        )

    def test_out(self):
        test_out = self.test_block(self.x, self.skip_con_x)
        self.assertEqual(
            test_out.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size * 2,
                self.test_size * 2,
            ),
        )


class TestFeatureMapBlock(unittest.TestCase):
    def setUp(self):
        self.test_samples = 10
        self.test_in_channels = 20
        self.test_out_channels = 40
        self.test_size = 50

        self.test_block = FeatureMapBlock(self.test_in_channels, self.test_out_channels)
        self.test_in = torch.randn(
            self.test_samples, self.test_in_channels, self.test_size, self.test_size
        )

    def tearDown(self):
        self.test_samples = None
        self.test_in_channels = None
        self.test_out_channels = None
        self.test_size = None

        self.test_block = None
        self.test_in = None

    def test_out(self):
        test_out = self.test_block(self.test_in)
        self.assertEqual(
            test_out.shape,
            (
                self.test_samples,
                self.test_out_channels,
                self.test_size,
                self.test_size,
            ),
        )


if __name__ == "__main__":
    unittest.main()
