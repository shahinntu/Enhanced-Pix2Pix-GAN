import sys
import unittest
from unittest.mock import patch

import torch

sys.path.append("../")
from gan_loss import GANLoss, GANGradient
from networks import Critic


class TestGANGradient(unittest.TestCase):
    def setUp(self):
        self.test_samples = 256
        self.test_channels = 1
        self.test_size = 28

        self.test_gan_gradient = GANGradient()
        self.test_critic = Critic(self.test_channels * 2, 4)
        self.test_condition = torch.randn(
            self.test_samples, self.test_channels, self.test_size, self.test_size
        )
        self.test_real = (
            torch.randn(
                self.test_samples, self.test_channels, self.test_size, self.test_size
            )
            + 1
        )
        self.test_fake = (
            torch.randn(
                self.test_samples, self.test_channels, self.test_size, self.test_size
            )
            - 1
        )

    def test_get_gradient(self):
        test_gradient = self.test_gan_gradient._get_gradient(
            self.test_critic, self.test_condition, self.test_real, self.test_fake
        )
        self.assertEqual(
            tuple(test_gradient.shape),
            (
                self.test_samples,
                self.test_channels,
                self.test_size,
                self.test_size,
            ),
        )
        self.assertGreater(test_gradient.max(), 0)
        self.assertLess(test_gradient.min(), 0)

    def test_gradient_penalty(self):
        with patch.object(
            self.test_gan_gradient,
            "_get_gradient",
            return_value=torch.zeros(
                (self.test_samples, self.test_channels, self.test_size, self.test_size)
            ),
        ) as mock_get_gradient:
            bad_gradient_penalty = self.test_gan_gradient.gradient_penalty(
                self.test_critic, self.test_condition, self.test_real, self.test_fake
            )
            assert torch.isclose(bad_gradient_penalty, torch.tensor(1.0))
            mock_get_gradient.assert_called()

        with patch.object(
            self.test_gan_gradient,
            "_get_gradient",
            return_value=torch.ones(
                (self.test_samples, self.test_channels, self.test_size, self.test_size)
            )
            / torch.sqrt(
                torch.tensor(self.test_channels * self.test_size * self.test_size)
            ),
        ) as mock_get_gradient:
            good_gradient_penalty = self.test_gan_gradient.gradient_penalty(
                self.test_critic, self.test_condition, self.test_real, self.test_fake
            )
            assert torch.isclose(good_gradient_penalty, torch.tensor(0.0))
            mock_get_gradient.assert_called()


class TestGANLoss(unittest.TestCase):
    def setUp(self):
        self.test_samples = 10
        self.test_channels = 3
        self.test_size = 10
        self.test_condition = torch.ones(
            self.test_samples, self.test_channels, self.test_size, self.test_size
        )
        self.test_gan_gradient = GANGradient()

    def tearDown(self):
        self.test_samples = None
        self.test_channels = None
        self.test_size = None
        self.test_condition = None
        self.test_gan_gradient = None

    def test_gen_loss_with_const_critic(self):
        test_gen = lambda x: torch.zeros_like(x)
        test_recon_criterion = lambda _, __: torch.tensor(0)
        test_lambda_recon = 0

        test_global_critic = lambda x, _: torch.ones(len(x), 1, dtype=torch.float)
        test_local_critic = lambda x, _: torch.ones(len(x), 1, dtype=torch.float)
        gan_loss = GANLoss(
            test_gen,
            test_global_critic,
            test_local_critic,
            test_recon_criterion,
            None,
            test_lambda_recon,
            "cpu",
        )
        loss = gan_loss.get_gen_loss(None, self.test_condition)
        self.assertEqual(loss.item(), -2.0)

        test_global_critic = lambda x, _: torch.zeros(len(x), 1, dtype=torch.float)
        test_local_critic = lambda x, _: torch.zeros(len(x), 1, dtype=torch.float)
        gan_loss = GANLoss(
            test_gen,
            test_global_critic,
            test_local_critic,
            test_recon_criterion,
            None,
            test_lambda_recon,
            "cpu",
        )
        loss = gan_loss.get_gen_loss(None, self.test_condition)
        self.assertEqual(loss.item(), 0)

    def test_gen_loss_for_criterion(self):
        test_recon_criterion = lambda x, y: torch.abs(x - y).max()
        test_real = torch.randn(
            self.test_samples, self.test_channels, self.test_size, self.test_size
        )
        test_gen = lambda _: test_real + 1
        test_global_critic = lambda x, _: torch.zeros(len(x), 1, dtype=torch.float)
        test_local_critic = lambda x, _: torch.zeros(len(x), 1, dtype=torch.float)
        test_lambda_recon = 2

        gan_loss = GANLoss(
            test_gen,
            test_global_critic,
            test_local_critic,
            test_recon_criterion,
            None,
            test_lambda_recon,
            "cpu",
        )
        loss = gan_loss.get_gen_loss(test_real, self.test_condition)
        self.assertAlmostEqual(abs(loss.item()), 2, delta=1e-4)

    def test_critic_loss(self):
        self.test_real = torch.randn(
            self.test_samples, self.test_channels, self.test_size, self.test_size
        )
        self.test_fake = torch.randn(
            self.test_samples, self.test_channels, self.test_size, self.test_size
        )
        test_c_lambda = 0.1
        test_gen = lambda x: torch.zeros_like(x)
        test_critic = (
            lambda x, _: torch.ones(len(x), 1, dtype=torch.float)
            if torch.equal(x, self.test_real)
            else torch.zeros(len(x), 1, dtype=torch.float)
        )

        gan_loss = GANLoss(
            test_gen,
            None,
            None,
            None,
            test_c_lambda,
            None,
            "cpu",
        )
        gan_loss._gan_gradient = self.test_gan_gradient

        with patch.object(
            self.test_gan_gradient,
            "gradient_penalty",
            return_value=torch.tensor(1.0),
        ) as mock_gradient_penalty:
            loss = gan_loss.get_critic_loss(
                test_critic, self.test_real, self.test_fake, self.test_condition
            )
            expected_loss = (0 - 1) + test_c_lambda * 1.0
            self.assertAlmostEqual(loss.item(), expected_loss, delta=1e-4)
            mock_gradient_penalty.assert_called()


if __name__ == "__main__":
    unittest.main()
