import torch


class GANGradient:
    def gradient_penalty(self, critic, condition, real, fake):
        gradient = self._get_gradient(critic, condition, real, fake)
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1)
        penalty = torch.mean((gradient_norm - 1) ** 2)

        return penalty

    def _get_gradient(self, critic, condition, real, fake):
        epsilon = torch.randn(
            len(real), 1, 1, 1, device=real.device, requires_grad=True
        )
        mixed_images = real * epsilon + fake * (1 - epsilon)
        mixed_scores = critic(mixed_images, condition)
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        return gradient


class GANLoss:
    def __init__(
        self,
        gen,
        global_critic,
        local_critic,
        recon_criterion,
        c_lambda,
        lambda_recon,
        device,
    ):
        self._gen = gen
        self._global_critic = global_critic
        self._local_critic = local_critic
        self._recon_criterion = recon_criterion
        self._c_lambda = c_lambda
        self._lambda_recon = lambda_recon
        self._device = device

        self._gan_gradient = GANGradient()

    def get_critic_loss(self, critic, real, fake, condition):
        critic_fake_pred = critic(fake.detach(), condition)
        critic_real_pred = critic(real, condition)
        critic_loss = (
            torch.mean(critic_fake_pred)
            - torch.mean(critic_real_pred)
            + self._c_lambda
            * self._gan_gradient.gradient_penalty(critic, condition, real, fake)
        )

        return critic_loss

    def get_gen_loss(self, real, condition):
        fake = self._gen(condition)
        global_critic_fake_pred = self._global_critic(fake, condition)
        global_gen_adv_loss = -1.0 * torch.mean(global_critic_fake_pred)
        local_critic_fake_pred = self._local_critic(fake, condition)
        local_gen_adv_loss = -1.0 * torch.mean(local_critic_fake_pred)
        gen_recon_loss = self._recon_criterion(real, fake)
        gen_loss = (
            global_gen_adv_loss
            + local_gen_adv_loss
            + self._lambda_recon * gen_recon_loss
        )

        return gen_loss
