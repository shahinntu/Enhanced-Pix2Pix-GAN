import sys

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from arg_parse import parse_args, Args
from data_preparation import CustomDataset
from networks import UNet, Critic
from metrics import FID, psnr
from train import Trainer
from utils import Params, weights_init, train_valid_split


def main(args):
    config = Params(args.config_path)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.MEAN, config.STD),
        ]
    )
    dataset = CustomDataset(args.train_data_dir, transform=transform)
    train_dataset, val_dataset = train_valid_split(dataset, config.VAL_SIZE)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, drop_last=True
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gen = UNet(config.INPUT_DIM, config.REAL_DIM)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=config.LR)
    global_critic = Critic(
        config.INPUT_DIM + config.REAL_DIM, config.GLOBAL_CRITIC_NUM_DOWN_BLOCKS
    )
    global_critic_optimizer = torch.optim.Adam(global_critic.parameters(), lr=config.LR)
    local_critic = Critic(
        config.INPUT_DIM + config.REAL_DIM, config.LOCAL_CRITIC_NUM_DOWN_BLOCKS
    )
    local_critic_optimizer = torch.optim.Adam(local_critic.parameters(), lr=config.LR)
    gen = gen.apply(weights_init)
    global_critic = global_critic.apply(weights_init)
    local_critic = local_critic.apply(weights_init)

    recon_criterion = nn.L1Loss()

    metrics = {"fid": FID(device, config), "psnr": psnr}

    trainer = Trainer(
        gen,
        global_critic,
        local_critic,
        gen_optimizer,
        global_critic_optimizer,
        local_critic_optimizer,
        recon_criterion,
        metrics,
        "fid",
        config,
        args.model_log_dir,
        device,
        restore_version=args.restore_version,
    )

    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    if "get_ipython" in globals():
        args = Args()
    else:
        args = parse_args(sys.argv[1:])

    main(args)
