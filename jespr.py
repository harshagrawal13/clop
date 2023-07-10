import torch
import torch.nn as nn

import lightning as pl
from data import ESMDataLoader
from util import load_esm_2, load_esm_if


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
            esm_2_model_type (str): ESM-2 model type - base_8M, base_650M. Defaults to base_8M.
            esm_if_model_type (str): ESM-IF model type - base_7M, base_142M. Defaults to base_7M.
            comb_emb_size (int): Combined Embedding Size. Defaults to 512.
        """
        super().__init__()

        self.esm_data_loader = esm_data_loader
        self.esm2, self.esm2_alphabet = load_esm_2(
            kwargs.get("esm_2_model_type", "base_8M")
        )
        self.esm_if, self.esm_if_alphabet = load_esm_if(
            kwargs.get("esm_if_model_type", "base_7M")
        )

        # Model params
        esm2_out_size = self.esm2.lm_head.dense.out_features
        esm_if_out_size = self.esm_if.encoder.layers[-1].fc2.out_features
        comb_emb_size = kwargs.get("comb_emb_size", 512)

        self.num_esm2_layers = len(self.esm2.layers)

        # Linear projection to 512 dim
        self.structure_emb_linear = nn.Linear(esm_if_out_size, comb_emb_size)
        self.seq_emb_linear = nn.Linear(esm2_out_size, comb_emb_size)

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
            tokens != self.esm_data_loader.esm_if_alphabet.padding_idx
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
        self.log("metrics/epoch/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
