import torch.nn as nn
from esm.inverse_folding import gvp_transformer
from esm import ESM2
from data import ESMDataLoader


class JESPR(nn.Module):
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

    def forward(self, x, num_esm2_layers=30) -> tuple:
        """Foward Function for JESPR

        Args:
            x (tuple): A tuple consisting of output from DataLoader's collate fn

        Returns:
            tuple: Model Embeddings for both ESM2 & ESM-IF
        """

        coords, confidence, strs, tokens, padding_mask = x

        # ESM2 - Sequence Embeddings
        seq_embeddings = self.esm2(
            tokens, repr_layers=[num_esm2_layers], return_contacts=False
        )["representations"][num_esm2_layers][:, 1:-1, :]

        # ESM-IF - Structure Embeddings
        structure_embeddings = self.esm_if.encoder.forward(
            coords=coords,
            encoder_padding_mask=padding_mask,
            confidence=confidence,
            return_all_hiddens=False,
        )["encoder_out"][0][1:-1, 0]

        return seq_embeddings, structure_embeddings
