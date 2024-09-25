import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision import models, transforms

from utils import denormalize_batch


class FID:
    def __init__(self, device, config, batch_size=32):
        self._device = device
        self._config = config
        self._batch_size = batch_size

        self._inc_model = models.inception_v3(weights="DEFAULT")
        self._inc_model.fc = torch.nn.Identity()
        self._inc_model.eval()
        self._inc_model = self._inc_model.to(self._device)
        self._preprocess = transforms.Compose(
            [
                transforms.Resize(299, antialias=True),
                transforms.CenterCrop(299),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, fakes, reals):
        fakes = denormalize_batch(
            fakes.detach().cpu(), self._config.MEAN, self._config.STD
        )
        reals = denormalize_batch(
            reals.detach().cpu(), self._config.MEAN, self._config.STD
        )

        fake_acts = self._get_activations(self._preprocess(fakes))
        real_acts = self._get_activations(self._preprocess(reals))

        mu1, sigma1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
        mu2, sigma2 = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)

        sigma1 = sigma1 + 1e-6 * np.eye(sigma1.shape[0])
        sigma2 = sigma2 + 1e-6 * np.eye(sigma2.shape[0])

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def _get_activations(self, images):
        activations = []

        for i in range(0, len(images), self._batch_size):
            images_batch = images[i : i + self._batch_size]
            images_batch = images_batch.to(self._device)

            with torch.no_grad():
                pred = self._inc_model(images_batch)

            activations.append(pred.detach().cpu().numpy())

        return np.concatenate(activations)


def psnr(fakes, reals):
    reals = reals.detach().cpu()
    fakes = fakes.detach().cpu()

    mse = torch.mean((reals - fakes) ** 2)

    max_pixel_value = 1.0 if reals.max() <= 1 else 255.0
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))

    return psnr_value.item()
