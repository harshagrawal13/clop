import os
import json
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from esm.inverse_folding import gvp_transformer
from esm import pretrained
from data import ESMDataLoader


class InitEsmModules:
    def __init__(self, **kwargs):
        """
        Keyword Args:
            args_dir (str): Path to args.json file for ESM-IF. Defaults to args.json in current working directory.
            gvp_node_hidden_dim_scalar (int): ESM-IF GVP Scalar Hidden Dim. Defaults to 1024.
            gvp_node_hidden_dim_vector (int): ESM-IF GVP Vector Hidden Dim. Defaults to 256.
            num_del_layers (int): Number of ESM-IF Layers to delete. Defaults to 6.
        """
        self.args_dir = kwargs.get(
            "args_dir", os.path.join(os.getcwd(), "args.json")
        )
        self.gvp_node_hidden_dim_scalar = kwargs.get(
            "gvp_node_hidden_dim_scalar", 1024
        )
        self.gvp_node_hidden_dim_vector = kwargs.get(
            "gvp_node_hidden_dim_vector", 256
        )
        self.num_del_layers = kwargs.get("num_del_layers", 6)

    def __call__(self):
        esm2, _ = pretrained.esm2_t6_8M_UR50D()
        _, alphabet_if = pretrained.esm_if1_gvp4_t16_142M_UR50()

        args_if = json.load(open(self.args_dir, "r"))
        args_if["gvp_node_hidden_dim_scalar"] = self.gvp_node_hidden_dim_scalar
        args_if["gvp_node_hidden_dim_vector"] = self.gvp_node_hidden_dim_vector
        args_if = Namespace(**args_if)

        esm_if = gvp_transformer.GVPTransformerModel(
            args_if,
            alphabet_if,
        )

        del esm_if.decoder
        del esm_if.encoder.layers[1 : self.num_del_layers + 1]

        return esm2, esm_if


class JESPR(pl.LightningModule):
    def __init__(
        self,
        esm_data_loader: ESMDataLoader,
        **kwargs,
    ) -> None:
        """
        JESPR Model

        Args:
            esm_data_loader (ESMDataLoader): ESMDataLoader

        Keyword Args:
            args_dir (str): Path to args.json file for ESM-IF. Defaults to args.json in current working directory.
            num_esm2_layers (int): Number of ESM2 Layers to use (30/33). Defaults to 30.
            esm2_out_size (int): ESM2 Output Size. Defaults to 640.
            esm_if_out_size (int): ESM-IF Output Size. Defaults to 1280.
            final_emb_size (int): Final Embedding Size. Defaults to 512.
            gvp_node_hidden_dim_scalar (int): ESM-IF GVP Scalar Hidden Dim. Defaults to 1024.
            gvp_node_hidden_dim_vector (int): ESM-IF GVP Vector Hidden Dim. Defaults to 256.
            num_del_layers (int): Number of ESM-IF Layers to delete. Defaults to 6.
        """
        super().__init__()

        self.esm_data_loader = esm_data_loader

        self.args_dir = kwargs.get(
            "args_dir", os.path.join(os.getcwd(), "args.json")
        )
        self.gvp_node_hidden_dim_scalar = kwargs.get(
            "gvp_node_hidden_dim_scalar", 1024
        )
        self.gvp_node_hidden_dim_vector = kwargs.get(
            "gvp_node_hidden_dim_vector", 256
        )
        self.num_del_layers = kwargs.get("num_del_layers", 6)

        self.esm2, self.esm_if = InitEsmModules(
            args_dir=self.args_dir,
            gvp_node_hidden_dim_scalar=self.gvp_node_hidden_dim_scalar,
            gvp_node_hidden_dim_vector=self.gvp_node_hidden_dim_vector,
            num_del_layers=self.num_del_layers,
        )()

        # Model params
        self.num_esm2_layers = kwargs.get("num_esm2_layers", 30)
        esm2_out_size = kwargs.get("esm2_out_size", 640)
        esm_if_out_size = kwargs.get("esm_if_out_size", 512)
        final_emb_size = kwargs.get("final_emb_size", 512)

        # Linear projection to 512 dim
        self.structure_emb_linear = nn.Linear(esm_if_out_size, final_emb_size)
        self.seq_emb_linear = nn.Linear(esm2_out_size, final_emb_size)

        # For scaling the cosing similarity score
        self.temperature = torch.tensor(1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x) -> tuple:
        """Foward Function for JESPR

        Args:
            x (tuple): A tuple consisting of output from DataLoader's collate fn

        Returns:
            tuple: Model Embeddings for both ESM2 & ESM-IF
        """

        coords, confidence, strs, tokens, padding_mask = x

        # ESM2 - Sequence Embeddings
        esm2_logits = self.esm2(
            tokens, repr_layers=[self.num_esm2_layers], return_contacts=False
        )["representations"][self.num_esm2_layers]

        # ESM-IF - Structure Embeddings
        esm_if_logits = self.esm_if.encoder.forward(
            coords=coords,
            encoder_padding_mask=padding_mask,
            confidence=confidence,
            return_all_hiddens=False,
        )["encoder_out"][0].swapaxes(0, 1)

        seq_embeddings = self.seq_emb_linear(esm2_logits)
        structure_embeddings = self.structure_emb_linear(esm_if_logits)

        # Batch Size * Residue Length * Embedding Size
        B, _, E = seq_embeddings.shape

        pooled_seq_embeddings = torch.empty(B, E, device=seq_embeddings.device)
        pooled_structure_embeddings = torch.empty_like(pooled_seq_embeddings)

        batch_padding_lens = (
            tokens != self.esm_data_loader.alphabet.padding_idx
        ).sum(1)

        # Average Pooling to get a single sequence
        # and structure embedding for the entire protein
        for i, tokens_len in enumerate(batch_padding_lens):
            pooled_seq_embeddings[i] = seq_embeddings[
                i, 1 : tokens_len - 1
            ].mean(0)

            # Average Pooling for structure
            pooled_structure_embeddings[i] = structure_embeddings[
                i, 1 : tokens_len - 1
            ].mean(0)

        # Normalize the embeddings
        # Taken from https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L362
        pooled_seq_embeddings = (
            pooled_seq_embeddings
            / pooled_seq_embeddings.norm(dim=1, keepdim=True)
        )

        pooled_structure_embeddings = (
            pooled_structure_embeddings
            / pooled_structure_embeddings.norm(dim=1, keepdim=True)
        )

        # Calculating the Loss
        # text = seq, image = structure
        logits = (
            pooled_seq_embeddings @ pooled_structure_embeddings.T
        ) / self.temperature

        loss = self.loss_fn(logits, torch.arange(B, device=logits.device))
        return logits, loss


    @staticmethod
    def cross_entropy(preds, targets, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def training_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
