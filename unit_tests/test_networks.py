import sys
import unittest

import torch

sys.path.append("../")
from networks import UNet, Critic


class TestUNet(unittest.TestCase):
    def setUp(self):
        self.test_samples = 20
        self.test_input_channels = 5
        self.test_output_channels = 8
        self.test_hidden_channels = 10
        self.test_size = 256

        self.test_net = UNet(
            self.test_input_channels,
            self.test_output_channels,
            hidden_channels=self.test_hidden_channels,
        )
        self.test_in = torch.randn(
            self.test_samples, self.test_input_channels, self.test_size, self.test_size
        )

    def tearDown(self):
        self.test_samples = None
        self.test_input_channels = None
        self.test_output_channels = None
        self.test_hidden_channels = None
        self.test_size = None

        self.test_net = None
        self.test_in = None

    def test_blocks(self):
        test_upfeature_out = self.test_net.upfeature(self.test_in)
        self.assertEqual(
            test_upfeature_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels,
                self.test_size,
                self.test_size,
            ),
        )

        test_downsample1_out = self.test_net.downsample1(test_upfeature_out)
        self.assertEqual(
            test_downsample1_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 2,
                self.test_size // 2,
                self.test_size // 2,
            ),
        )

        test_downsample2_out = self.test_net.downsample2(test_downsample1_out)
        self.assertEqual(
            test_downsample2_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 4,
                self.test_size // 4,
                self.test_size // 4,
            ),
        )

        test_downsample3_out = self.test_net.downsample3(test_downsample2_out)
        self.assertEqual(
            test_downsample3_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 8,
                self.test_size // 8,
                self.test_size // 8,
            ),
        )

        test_downsample4_out = self.test_net.downsample4(test_downsample3_out)
        self.assertEqual(
            test_downsample4_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 16,
                self.test_size // 16,
                self.test_size // 16,
            ),
        )

        test_downsample5_out = self.test_net.downsample5(test_downsample4_out)
        self.assertEqual(
            test_downsample5_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 32,
                self.test_size // 32,
                self.test_size // 32,
            ),
        )

        test_downsample6_out = self.test_net.downsample6(test_downsample5_out)
        self.assertEqual(
            test_downsample6_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 64,
                self.test_size // 64,
                self.test_size // 64,
            ),
        )

        test_downsample7_out = self.test_net.downsample7(test_downsample6_out)
        self.assertEqual(
            test_downsample7_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 128,
                self.test_size // 128,
                self.test_size // 128,
            ),
        )

        test_upsample0_out = self.test_net.upsample0(
            test_downsample7_out, test_downsample6_out
        )
        self.assertEqual(
            test_upsample0_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 64,
                self.test_size // 64,
                self.test_size // 64,
            ),
        )

        test_upsample1_out = self.test_net.upsample1(
            test_upsample0_out, test_downsample5_out
        )
        self.assertEqual(
            test_upsample1_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 32,
                self.test_size // 32,
                self.test_size // 32,
            ),
        )

        test_upsample2_out = self.test_net.upsample2(
            test_upsample1_out, test_downsample4_out
        )
        self.assertEqual(
            test_upsample2_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 16,
                self.test_size // 16,
                self.test_size // 16,
            ),
        )

        test_upsample3_out = self.test_net.upsample3(
            test_upsample2_out, test_downsample3_out
        )
        self.assertEqual(
            test_upsample3_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 8,
                self.test_size // 8,
                self.test_size // 8,
            ),
        )

        test_upsample4_out = self.test_net.upsample4(
            test_upsample3_out, test_downsample2_out
        )
        self.assertEqual(
            test_upsample4_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 4,
                self.test_size // 4,
                self.test_size // 4,
            ),
        )

        test_upsample5_out = self.test_net.upsample5(
            test_upsample4_out, test_downsample1_out
        )
        self.assertEqual(
            test_upsample5_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 2,
                self.test_size // 2,
                self.test_size // 2,
            ),
        )

        test_upsample6_out = self.test_net.upsample6(
            test_upsample5_out, test_upfeature_out
        )
        self.assertEqual(
            test_upsample6_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels,
                self.test_size,
                self.test_size,
            ),
        )

        test_downfeature_out = self.test_net.downfeature(test_upsample6_out)
        self.assertEqual(
            test_downfeature_out.shape,
            (
                self.test_samples,
                self.test_output_channels,
                self.test_size,
                self.test_size,
            ),
        )

    def test_out(self):
        test_out = self.test_net(self.test_in)
        self.assertEqual(
            test_out.shape,
            (
                self.test_samples,
                self.test_output_channels,
                self.test_size,
                self.test_size,
            ),
        )


class TestCritic(unittest.TestCase):
    def setUp(self):
        self.test_samples = 20
        self.test_input_channels = 5
        self.test_num_down_blocks = 4
        self.test_hidden_channels = 10
        self.test_size = 256

        self.test_net = Critic(
            self.test_input_channels * 2,
            self.test_num_down_blocks,
            self.test_hidden_channels,
        )
        self.test_x = torch.randn(
            self.test_samples, self.test_input_channels, self.test_size, self.test_size
        )
        self.test_y = torch.randn(
            self.test_samples, self.test_input_channels, self.test_size, self.test_size
        )

    def tearDown(self):
        self.test_samples = None
        self.test_input_channels = None
        self.test_num_down_blocks = None
        self.test_hidden_channels = None
        self.test_size = None

        self.test_net = None
        self.test_x = None
        self.test_y = None

    def test_blocks(self):
        concat_x_y = torch.cat([self.test_x, self.test_y], axis=1)

        test_upfeature_out = self.test_net.upfeature(concat_x_y)
        self.assertEqual(
            test_upfeature_out.shape,
            (
                self.test_samples,
                self.test_hidden_channels,
                self.test_size,
                self.test_size,
            ),
        )

        test_downsampling_blocks = self.test_net.downsampling_blocks(test_upfeature_out)
        self.assertEqual(
            test_downsampling_blocks.shape,
            (
                self.test_samples,
                self.test_hidden_channels * 16,
                self.test_size // 16,
                self.test_size // 16,
            ),
        )

        test_final_out = self.test_net.final(test_downsampling_blocks)
        self.assertEqual(
            test_final_out.shape,
            (
                self.test_samples,
                1,
                self.test_size // 16,
                self.test_size // 16,
            ),
        )

    def test_out(self):
        test_out = self.test_net(self.test_x, self.test_y)
        self.assertEqual(
            test_out.shape,
            (
                self.test_samples,
                1,
                self.test_size // 16,
                self.test_size // 16,
            ),
        )


if __name__ == "__main__":
    unittest.main()
