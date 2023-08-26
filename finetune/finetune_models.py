import os
import time
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import lightning as pl

import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))

from modules import SequenceEncoder
from modules import WarmupCosineFactorLambda


class JESPR_Regression(pl.LightningModule):
    def __init__(
        self,
        seq_encoder: SequenceEncoder,
        optim_args: dict,
        acc_metric_on_train_batch: bool = True,
    ) -> None:
        """JESPR Regression Class. Can be used for both Log Stability Prediction & Log Fluorescence Prediction.

        Args:
            seq_encoder (SequenceEncoder): ESM2 Model
            optim_args (dict): Optimizer args.
        """
        super().__init__()
        self.seq_encoder = seq_encoder
        self.optim_args = optim_args
        self.pred_linear = nn.Linear(
            in_features=self.seq_encoder.joint_embedding_projection.out_features,
            out_features=1,
        )
        self.mse_loss_fn = nn.MSELoss()
        self.acc_metric_on_train_batch = acc_metric_on_train_batch

    def forward(self, x) -> tuple:
        tokens, y = x
        esm2_out = self.seq_encoder(tokens)  # B * Joint_embed_dim
        y_pred = self.pred_linear(esm2_out).squeeze()  # B

        loss = self.mse_loss_fn(y_pred, y)
        return loss, y_pred, y

    @staticmethod
    @torch.no_grad()
    def spearmanr(target: np.array, prediction: np.array) -> float:
        """Spearman R correlation.
        Taken from https://github.com/songlab-cal/tape/blob/master/tape/metrics.py

        Args:
            target (np.array): Target
            prediction (np.array): Prediction

        Returns:
            float: Spearman R correlation
        """
        return scipy.stats.spearmanr(target, prediction).correlation

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        B = batch[0].shape[0]
        loss, y_pred, y = self.forward(batch)
        self.log("metrics/train/loss", loss, batch_size=B)
        self.log(
            "metrics/train/time_per_step",
            time.time() - start_time,
            batch_size=B,
        )
        return {"loss": loss, "y_pred": y_pred, "y": y}

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        B = batch[0].shape[0]
        loss, y_pred, y = self.forward(batch)
        self.log("metrics/val/loss", loss, batch_size=B)
        self.log(
            "metrics/val/time_per_step",
            time.time() - start_time,
            batch_size=B,
        )
        return {"loss": loss, "y_pred": y_pred, "y": y}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Only calculate Spearman R correlation on train batch if specified
        if self.acc_metric_on_train_batch:
            spearman_r = self.spearmanr(
                outputs["y"].detach().cpu().numpy(),
                outputs["y_pred"].detach().cpu().numpy(),
            )
            self.log("metrics/train/spearman_r", spearman_r)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        spearman_r = self.spearmanr(
            outputs["y"].detach().cpu().numpy(),
            outputs["y_pred"].detach().cpu().numpy(),
        )
        self.log("metrics/val/spearman_r", spearman_r)

    def configure_optimizers(self):
        """Return Optimizer

        Returns:
            torch.optim.Adam: Adam Optimizer
        """
        optim_params = self.optim_args["optim_args"]
        scheduler_params = self.optim_args["scheduler"]

        optimizer = Adam(self.parameters(), **optim_params)
        if scheduler_params["type"] == "warmup_cosine_schedule":
            self.scheduler_lamba = WarmupCosineFactorLambda(
                warmup_steps=scheduler_params["warmup_steps"],
                max_steps=scheduler_params["max_steps"],
                max_lr=scheduler_params["max_lr"],
                final_lr=scheduler_params["final_lr"],
                eps=scheduler_params["eps"],
            )
            lr_scheduler = LambdaLR(
                optimizer=optimizer,
                lr_lambda=self.scheduler_lamba.compute_lr_factor,
                verbose=scheduler_params["verbose"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer
