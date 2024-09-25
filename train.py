import os
import time
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from gan_loss import GANLoss
from utils import (
    save_checkpoint,
    load_checkpoint,
    set_logger,
    clear_handlers,
    save_dict_to_json,
    RunningAverageDict,
    denormalize_batch,
    show_tensor_images,
)


class Trainer:
    def __init__(
        self,
        gen,
        global_critic,
        local_critic,
        gen_optimizer,
        global_critic_optimizer,
        local_critic_optimizer,
        recon_crterion,
        metrics,
        objective,
        config,
        model_log_dir,
        device,
        restore_version=None,
    ):
        self._gen = gen
        self._global_critic = global_critic
        self._local_critic = local_critic
        self._gen_optimizer = gen_optimizer
        self._global_critic_optimizer = global_critic_optimizer
        self._local_critic_optimizer = local_critic_optimizer
        self._recon_criterion = recon_crterion
        self._metrics = metrics
        self._objective = objective
        self._config = config
        self._model_log_dir = model_log_dir
        self._device = device
        self._restore_version = restore_version

        self._gen = self._gen.to(self._device)
        self._global_critic = self._global_critic.to(self._device)
        self._local_critic = self._local_critic.to(self._device)
        self._recon_criterion = self._recon_criterion.to(self._device)

        self._gan_loss = GANLoss(
            self._gen,
            self._global_critic,
            self._local_critic,
            self._recon_criterion,
            self._config.C_LAMBDA,
            self._config.LAMBDA_RECON,
            self._device,
        )

        if self._restore_version is not None:
            self._restore_parameters()

    def train(self, train_dataloader, val_dataloader):
        log_dir = self._create_log_dirs()
        train_log_path = os.path.join(log_dir, "train_logs", "train.log")
        train_logger = set_logger(train_log_path)
        self._config.save(os.path.join(log_dir, "hyper_params/params.json"))

        summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb_logs"))

        best_val_obj = float("inf") if self._objective == "fid" else 0.0

        for epoch in range(self._config.EPOCHS):
            logging.info(f"Epoch {epoch+1}/{self._config.EPOCHS}")

            train_metrics_avg_dict = self._train_epoch(train_dataloader)
            val_metrics_avg_dict = self.evaluate(val_dataloader)

            best_val_obj = self._log_model_state(
                epoch,
                log_dir,
                val_metrics_avg_dict,
                best_val_obj,
            )
            self._write_tb_logs(
                summary_writer, epoch, train_metrics_avg_dict, val_metrics_avg_dict
            )

        summary_writer.close()
        clear_handlers(train_logger)

    @torch.no_grad()
    def evaluate(self, val_dataloader):
        self._gen.train()

        val_metrics_avg_dict = RunningAverageDict(
            [metric for metric in list(self._metrics.keys()) if metric != "fid"]
        )

        for val_batch in val_dataloader:
            condition, real = self._unpack_batch(val_batch)

            fake = self._gen(condition)

            val_metrics_dict = {
                metric: self._metrics[metric](fake, real)
                for metric in self._metrics
                if metric != "fid"
            }
            val_metrics_avg_dict.update(val_metrics_dict)

        val_metrics_avg_dict_out = val_metrics_avg_dict()

        if "fid" in self._metrics:
            val_metrics_avg_dict_out["fid"] = self._get_fid(val_dataloader)

        logging.info(
            f"- Validation metrics: {self._format_metrics(val_metrics_avg_dict_out)}"
        )

        return val_metrics_avg_dict_out

    def _train_epoch(self, train_dataloader):
        self._gen.train()
        self._global_critic.train()
        self._local_critic.train()

        train_metrics_avg_dict_obj = RunningAverageDict(
            ["gen_loss", "global_critic_loss", "local_critic_loss"]
            + [metric for metric in list(self._metrics.keys()) if metric != "fid"]
        )

        with tqdm(total=len(train_dataloader)) as t:
            for train_batch in train_dataloader:
                train_metrics_dict, condition, real, fake = self._train_step(
                    train_batch
                )

                train_metrics_avg_dict_obj.update(train_metrics_dict)

                t.set_postfix(
                    metrics=self._format_metrics(train_metrics_avg_dict_obj())
                )
                t.update()

        logging.info(
            f"- Train metrics: {self._format_metrics(train_metrics_avg_dict_obj())}"
        )

        # self._display_images(condition, real, fake)

        return train_metrics_avg_dict_obj()

    def _train_step(self, train_batch):
        condition, real = self._unpack_batch(train_batch)
        with torch.no_grad():
            fake = self._gen(condition)

        global_critic_loss = self._train_critic(
            self._global_critic, self._global_critic_optimizer, real, fake, condition
        )
        local_critic_loss = self._train_critic(
            self._local_critic, self._local_critic_optimizer, real, fake, condition
        )

        self._gen_optimizer.zero_grad()
        gen_loss = self._gan_loss.get_gen_loss(real, condition)
        gen_loss.backward()
        self._gen_optimizer.step()

        train_metrics_dict = {
            metric: self._metrics[metric](fake, real)
            for metric in self._metrics
            if metric != "fid"
        }

        train_metrics_dict["gen_loss"] = gen_loss.item()
        train_metrics_dict["global_critic_loss"] = global_critic_loss
        train_metrics_dict["local_critic_loss"] = local_critic_loss

        return train_metrics_dict, condition, real, fake

    def _train_critic(self, critic, critic_optimizer, real, fake, condition):
        total_iteration_critic_loss = 0
        for _ in range(self._config.CRITIC_REPEATS):
            critic_optimizer.zero_grad()
            critic_loss = self._gan_loss.get_critic_loss(critic, real, fake, condition)
            total_iteration_critic_loss += critic_loss.item()
            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

        return total_iteration_critic_loss / self._config.CRITIC_REPEATS

    def _restore_parameters(self):
        gen_restore_path = os.path.join(
            self._model_log_dir, self._restore_version, "state/gen_best.pth"
        )
        global_critic_restore_path = os.path.join(
            self._model_log_dir, self._restore_version, "state/glob_crit_best.pth"
        )
        local_critic_restore_path = os.path.join(
            self._model_log_dir, self._restore_version, "state/loc_crit_best.pth"
        )
        load_checkpoint(gen_restore_path, self._gen, self._gen_optimizer)
        load_checkpoint(
            global_critic_restore_path,
            self._global_critic,
            self._global_critic_optimizer,
        )
        load_checkpoint(
            local_critic_restore_path,
            self._local_critic,
            self._local_critic_optimizer,
        )
        self._config.add("RESTORE_VERSION", self._restore_version)

    def _get_fid(self, val_dataloader):
        real_list = []
        fake_list = []
        for val_batch in val_dataloader:
            condition, real = self._unpack_batch(val_batch)
            fake = self._gen(condition)
            real_list.append(real)
            fake_list.append(fake)

            if len(real_list) * len(real) >= 5000:
                break

        reals = torch.cat(real_list, dim=0)
        fakes = torch.cat(fake_list, dim=0)

        return self._metrics["fid"](fakes, reals)

    def _unpack_batch(self, batch):
        condition, real = batch
        condition = condition.to(self._device)
        real = real.to(self._device)

        return condition, real

    def _create_log_dirs(self):
        log_dir = os.path.join(
            self._model_log_dir,
            time.strftime("%y%m%d%H%M%S", time.localtime(time.time())),
        )
        os.mkdir(log_dir)
        os.mkdir(os.path.join(log_dir, "state"))
        os.mkdir(os.path.join(log_dir, "tb_logs"))
        os.mkdir(os.path.join(log_dir, "hyper_params"))
        os.mkdir(os.path.join(log_dir, "train_logs"))

        return log_dir

    def _log_model_state(self, epoch, log_dir, metrics_avg_dict, best_val_obj):
        is_best = (
            metrics_avg_dict[self._objective] < best_val_obj
            if self._objective == "fid"
            else metrics_avg_dict[self._objective] > best_val_obj
        )

        log_model_summary = [
            (self._gen, self._gen_optimizer, "gen"),
            (self._global_critic, self._global_critic_optimizer, "glob_crit"),
            (self._local_critic, self._local_critic_optimizer, "loc_crit"),
        ]

        for model, optim, type in log_model_summary:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optim_dict": optim.state_dict(),
                },
                is_best=is_best,
                checkpoint=os.path.join(log_dir, "state"),
                network_type=type,
            )

        if is_best:
            logging.info("- Found new best objective")
            best_val_obj = metrics_avg_dict[self._objective]

            best_json_path = os.path.join(log_dir, "state", "best_metrics.json")
            save_dict_to_json(metrics_avg_dict, best_json_path)

        last_json_path = os.path.join(log_dir, "state", "last_metrics.json")
        save_dict_to_json(metrics_avg_dict, last_json_path)

        return best_val_obj

    def _write_tb_logs(self, writer, step, train_metrics, val_metrics):
        for key in train_metrics:
            writer.add_scalar(f"{key}/train", train_metrics[key], step)
        for key in val_metrics:
            writer.add_scalar(f"{key}/validation", val_metrics[key], step)

    def _format_metrics(self, metrics_avg_dict):
        metrics_string = "; ".join(
            f"{k}: {v:05.3f}" for k, v in metrics_avg_dict.items()
        )
        return metrics_string

    def _display_images(self, *imgs):
        for img in imgs:
            img = img.detach().cpu()
            img = denormalize_batch(img, self._config.MEAN, self._config.STD)
            show_tensor_images(img)
