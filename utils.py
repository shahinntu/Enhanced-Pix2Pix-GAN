import os
import json
import logging
import shutil

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split


class Params:
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def add(self, key, value):
        self.__dict__[key] = value

    @property
    def dict(self):
        return self.__dict__


class RunningAverageDict:
    def __init__(self, keys):
        self.total_dict = {}
        for key in keys:
            self.total_dict[key] = 0
        self.steps = 0

    def update(self, val_dict):
        for key in self.total_dict:
            self.total_dict[key] += val_dict[key]
        self.steps += 1

    def reset(self):
        for key in self.total_dict:
            self.total_dict[key] = 0
        self.steps = 0

    def __call__(self):
        return {
            key: value / float(self.steps) for key, value in self.total_dict.items()
        }


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    return logger


def clear_handlers(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, network_type):
    filepath = os.path.join(checkpoint, f"{network_type}_last.pth")
    if not os.path.exists(checkpoint):
        os.mkdir(checkpoint)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f"{network_type}_best.pth"))


def load_checkpoint(checkpoint, model, optimizer=None, device=None):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist at {checkpoint}")
    if device:
        checkpoint = torch.load(checkpoint, map_location=device)
    else:
        checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])


def show_tensor_images(tensors, num=4):
    tensors = tensors.detach().cpu()
    tensors = tensors.permute(0, 2, 3, 1)
    tensors = tensors.clamp(0, 1)

    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(tensors[i])
        plt.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def denormalize_batch(tensor_batch, mean, std):
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)

    tensor_batch = tensor_batch * std + mean
    return tensor_batch


def train_valid_split(dataset, valid_size):
    data_len = len(dataset)
    valid_len = int(valid_size * data_len)
    return random_split(dataset, [data_len - valid_len, valid_len])
