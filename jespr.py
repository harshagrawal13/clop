import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from esm.inverse_folding import gvp_transformer
from esm import ESM2
from data import ESMDataLoader


class JESPR(pl.LightningModule):
    def __init__(
        self,
        esm_data_loader: ESMDataLoader,
        esm2: ESM2,
        esm_if: gvp_transformer.GVPTransformerModel,
    ) -> None:
        super().__init__()

        self.esm_data_loader = esm_data_loader

        # Protein Sequence Model
        self.esm2 = esm2
        # Protein Structure Model
        self.esm_if = esm_if

        # Linear projection to 512 dim
        self.structure_emb_linear = nn.Linear(512, 512)
        self.seq_emb_linear = nn.Linear(1280, 512)

        # For scaling the cosing similarity score
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temperature = torch.tensor(1)

    def forward(self, x, num_esm2_layers=30) -> tuple:
        """Foward Function for JESPR

        Args:
            x (tuple): A tuple consisting of output from DataLoader's collate fn

        Returns:
            tuple: Model Embeddings for both ESM2 & ESM-IF
        """

        coords, confidence, strs, tokens, padding_mask = x

        # ESM2 - Sequence Embeddings
        esm2_logits = self.esm2(
            tokens, repr_layers=[num_esm2_layers], return_contacts=False
        )["representations"][num_esm2_layers]

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

        pooled_seq_embeddings = torch.empty(B, E)
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

        batch_structure_similarity = (
            pooled_structure_embeddings @ pooled_structure_embeddings.T
        )
        batch_seq_similarity = pooled_seq_embeddings @ pooled_seq_embeddings.T
        targets = F.softmax(
            (batch_seq_similarity + batch_structure_similarity)
            / 2
            * self.temperature,
            dim=-1,
        )
        # TODO: Explore having two individual losses
        loss = self.cross_entropy(logits, targets, reduction="mean")
        return loss, logits

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
