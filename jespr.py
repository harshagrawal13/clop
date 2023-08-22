from os import path
import time
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, CyclicLR

import lightning as pl

from esm.data import Alphabet
from modules import SequenceEncoder, StructureEncoder

DEFAULT_LR = 3e-4


class JESPR(pl.LightningModule):
    def __init__(
        self,
        sequence_encoder: SequenceEncoder,
        structure_encoder: StructureEncoder,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        optim_args: dict,
    ) -> None:
        """
        JESPR Model

        Args:
            esm2 (ESM2): ESM-2 Model
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if (GVPTransformerModel): ESM-IF Model
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            optim_args (dict, optional): Optimizer Arguments. Defaults to {"lr": DEFAULT_LR}.
            temperature (float, optional): Temperature for scaling the cosine similarity score. Defaults to INIT_TEMP.
        """
        super().__init__()

        self.seq_encoder, self.struct_encoder = (
            sequence_encoder,
            structure_encoder,
        )
        self.esm2_alphabet, self.esm_if_alphabet = (
            esm2_alphabet,
            esm_if_alphabet,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.optim_args = optim_args

    def forward(self, x) -> tuple:
        """Foward Function for JESPR

        Args:
            x (tuple): A tuple consisting of output from DataLoader's collate fn

        Returns:
            tuple: Model Embeddings for both ESM2 & ESM-IF
        """

        coords, confidence, strs, tokens, padding_mask = x

        # Get ESM2 & ESM-IF outputs. Shape: Batch_size * Joint_embed_dim
        seq_repr = self.seq_encoder(tokens)
        struct_repr = self.struct_encoder(coords, padding_mask, confidence)

        B, J = seq_repr.shape
        # Calculating the Loss

        logit_scale = self.logit_scale.exp()
        logits_per_structure = logit_scale * seq_repr @ struct_repr.T
        logits_per_seq = logits_per_structure.T

        labels = torch.arange(B, dtype=torch.long, device=logits_per_structure.device)

        loss = (
            torch.nn.functional.cross_entropy(logits_per_structure, labels)
            + torch.nn.functional.cross_entropy(logits_per_seq, labels)
        ) / 2

        return loss, {
            "logits_per_structure": logits_per_structure,
            "logits_per_seq": logits_per_seq,
        }

    @torch.no_grad()
    def calc_argmax_acc(self, logits: dict) -> float:
        """Calculate Argmax Accuracy

        Args:
            logits (dict): Dict of logits for structure and sequence

        Returns:
            float: Argmax Accuracy
        """
        am_logits_per_structure = logits["logits_per_structure"].argmax(0)
        am_logits_per_seq = logits["logits_per_seq"].argmax(0)

        b = am_logits_per_structure.shape[0]  # Batch Size
        truths = torch.arange(b, device=am_logits_per_structure.device)
        acc_str = torch.sum(am_logits_per_structure == truths).item() / b
        acc_seq = torch.sum(am_logits_per_seq == truths).item() / b

        return {"acc_str": acc_str, "acc_seq": acc_seq}

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        loss, logits = self.forward(batch)
        B = batch[0].shape[0]
        self.log("metrics/train/loss", loss, batch_size=B)
        self.log(
            "metrics/train/time_per_step",
            time.time() - start_time,
            batch_size=B,
        )
        return {"loss": loss, "logits": logits}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        argmax_acc = self.calc_argmax_acc(outputs["logits"])
        B = batch[0].shape[0]
        self.log("metrics/train/acc_structure", argmax_acc["acc_str"], batch_size=B)
        self.log("metrics/train/acc_sequence", argmax_acc["acc_seq"], batch_size=B)

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        B = batch[0].shape[0]
        loss, logits = self.forward(batch)
        self.log("metrics/val/loss", loss, batch_size=B)
        self.log(
            "metrics/val/time_per_step",
            time.time() - start_time,
            batch_size=B,
        )
        return {"loss": loss, "logits": logits}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        argmax_acc = self.calc_argmax_acc(outputs["logits"])
        B = batch[0].shape[0]
        self.log("metrics/val/acc_structure", argmax_acc["acc_str"], batch_size=B)
        self.log("metrics/val/acc_sequence", argmax_acc["acc_seq"], batch_size=B)

    def configure_optimizers(self) -> torch.optim.Adam:
        """Return Optimizer

        Returns:
            torch.optim.Adam: Adam Optimizer
        """
        optim_params = self.optim_args["optim_args"]
        scheduler_params = self.optim_args["scheduler"]

        optimizer = Adam(self.parameters(), **optim_params)
        if scheduler_params["type"] == "cyclic_lr":
            scheduler = CyclicLR(
                optimizer,
                base_lr=scheduler_params["base_lr"],
                max_lr=scheduler_params["max_lr"],
                mode=scheduler_params["mode"],
            )

            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    @staticmethod
    def num_params(model: nn.Module) -> int:
        """Calculate the number of parameters

        Args:
            model (nn.Module): PyTorch Model
        Returns:
            int: Total Number of params
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


class JESPR_RH(pl.LightningModule):
    def __init__(
        self,
        seq_encoder: SequenceEncoder,
        optim_args: dict = {
            "lr": DEFAULT_LR,
        },
    ) -> None:
        """_summary_

        Args:
            seq_encoder (SequenceEncoder): ESM2 Model
            optim_args (dict): Optimizer args.
        """
        super().__init__()
        self.seq_encoder = seq_encoder
        self.optim_args = optim_args
        self.pred_linear = nn.Linear(
            in_features=self.seq_encoder.joint_embedding_projection.out_features,
            out_features=1195,
        )

    def forward(self, x) -> tuple:
        tokens, class_labels = x
        esm2_out = self.seq_encoder(tokens)  # B * Joint_embed_dim
        logits = self.pred_linear(esm2_out)  # B * NUM_HOMOLOGY_FOLDS

        loss = torch.nn.functional.cross_entropy(logits, class_labels)
        return loss, logits, class_labels

    @torch.no_grad()
    def calc_argmax_acc(
        self, logits: torch.tensor, class_labels: torch.tensor
    ) -> float:
        """Calculate Argmax Accuracy

        Args:
            logits (torch.tensor): Prediction Probabilities
            class_labels (torch.tensor): Class Labels

        Returns:
            float: Argmax Accuracy
        """
        b = logits.shape[0]  # Batch Size
        argmax_acc = (logits.argmax(1) == class_labels).sum().item() / b
        return argmax_acc

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        B = batch[0].shape[0]
        loss, logits, labels = self.forward(batch)
        self.log("metrics/step/train_loss", loss, batch_size=B)
        self.log(
            "metrics/step/time_per_train_step",
            time.time() - start_time,
            batch_size=B,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        B = batch[0].shape[0]
        loss, logits, labels = self.forward(batch)
        self.log("metrics/step/val_loss", loss, batch_size=B)
        self.log(
            "metrics/step/time_per_val_step",
            time.time() - start_time,
            batch_size=B,
        )
        return (logits, labels)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        argmax_acc = self.calc_argmax_acc(outputs[0], outputs[1])
        self.log("metrics/step/accuracy", argmax_acc)

    def configure_optimizers(self) -> torch.optim.Adam:
        """Return Optimizer

        Returns:
            torch.optim.Adam: Adam Optimizer
            torch.optim.lr_scheduler.ExponentialLR: ExponentialLR
        """
        optimizer = Adam(self.parameters(), **self.optim_args)
        scheduler = ExponentialLR(optimizer, gamma=0.99)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class JESPR_Regression(pl.LightningModule):
    def __init__(
        self,
        seq_encoder: SequenceEncoder,
        optim_args: dict = {
            "lr": DEFAULT_LR,
        },
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
        self.log("metrics/step/train_loss", loss, batch_size=B)
        self.log(
            "metrics/step/time_per_train_step",
            time.time() - start_time,
            batch_size=B,
        )
        return {"loss": loss, "y_pred": y_pred, "y": y}

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        B = batch[0].shape[0]
        loss, y_pred, y = self.forward(batch)
        self.log("metrics/step/val_loss", loss, batch_size=B)
        self.log(
            "metrics/step/time_per_val_step",
            time.time() - start_time,
            batch_size=B,
        )
        return {"loss": loss, "y_pred": y_pred, "y": y}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Only calculate Spearman R correlation on train batch if specified
        if self.acc_metric_on_train_batch:
            spearman_r = self.spearmanr(
                outputs["y"].detach().numpy(),
                outputs["y_pred"].detach().numpy(),
            )
            self.log("metrics/step/train_spearman_r", spearman_r)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        spearman_r = self.spearmanr(
            outputs["y"].detach().numpy(), outputs["y_pred"].detach().numpy()
        )
        self.log("metrics/step/val_spearman_r", spearman_r)

    def configure_optimizers(self):
        """Return Optimizer

        Returns:
            torch.optim.Optimizer
            torch.optim.lr_scheduler (optional)
        """
        optim_params = self.optim_args
        scheduler_params = optim_params.pop("scheduler")

        optimizer = Adam(self.parameters(), **optim_params)
        if scheduler_params["type"] == "cyclic_lr":
            scheduler = CyclicLR(
                optimizer,
                base_lr=scheduler_params["base_lr"],
                max_lr=scheduler_params["max_lr"],
                mode=scheduler_params["mode"],
            )

            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
