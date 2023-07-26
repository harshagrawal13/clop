from os import path
import time
import numpy as np
import torch
import torch.nn as nn

import lightning as pl

from esm.data import Alphabet
from modules import _ESM2, _ESM_IF

DEFAULT_LR = 3e-4
INIT_TEMP = 0.07


class JESPR(pl.LightningModule):
    def __init__(
        self,
        esm2: _ESM2,
        esm2_alphabet: Alphabet,
        esm_if: _ESM_IF,
        esm_if_alphabet: Alphabet,
        optim_args: dict = {"lr": DEFAULT_LR},
        temperature: float = INIT_TEMP,
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

        self.esm2, self.esm2_alphabet = esm2, esm2_alphabet
        self.esm_if, self.esm_if_alphabet = esm_if, esm_if_alphabet

        # For scaling the cosing similarity score
        self.temperature = nn.Parameter(torch.tensor(temperature))

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
        esm2_out = self.esm2(tokens, need_head_weights=False)
        esm_if_out = self.esm_if(coords, padding_mask, confidence)

        B, J = esm2_out.shape
        # Calculating the Loss
        # text = seq, image = structure
        logits_per_structure = self.temperature * esm_if_out @ esm2_out.T
        logits_per_seq = self.temperature * esm2_out @ esm_if_out.T

        labels = torch.arange(
            B, dtype=torch.long, device=logits_per_structure.device
        )

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
        self.log("metrics/step/train_loss", loss, batch_size=self.batch_size)
        self.log(
            "metrics/step/time_per_train_step",
            time.time() - start_time,
            batch_size=self.batch_size,
        )
        return {"loss": loss, "logits": logits}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        argmax_acc = self.calc_argmax_acc(outputs["logits"])
        self.log("metrics/step/argmax_acc_structure", argmax_acc["acc_str"])
        self.log("metrics/step/argmax_acc_sequence", argmax_acc["acc_seq"])

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        loss, _ = self.forward(batch)
        self.log("metrics/step/val_loss", loss, batch_size=self.batch_size)
        self.log(
            "metrics/step/time_per_val_step",
            time.time() - start_time,
            batch_size=self.batch_size,
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        """Return Optimizer

        Returns:
            torch.optim.Adam: Adam Optimizer
        """
        return torch.optim.Adam(self.parameters(), **self.optim_args)
