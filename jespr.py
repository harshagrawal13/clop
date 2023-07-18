from os import path
import time
import numpy as np
import torch
import torch.nn as nn

import lightning as pl

from esm.inverse_folding.gvp_transformer import GVPTransformerModel
from esm.model.esm2 import ESM2
from esm.data import Alphabet

from data import ESMDataLoader
from util import load_esm_2, load_esm_if

ESM2_PADDING_IDX = 0
DEFAULT_COMBINED_EMB_SIZE = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-4
DEFAULT_LOG_EVERY_N_STEPS = 10
DEFAULT_MAX_EPOCHS = 50
DEFAULT_VAL_CHECK_ITVL = 0.25  # default validation step interval (per epoch)
DEFAULT_ESM2_MODEL_TYPE = "base_8M"
DEFAULT_ESM_IF_MODEL_TYPE = "base_7M"
DEFAULT_PROJECT_NAME = "jespr"
DEFAULT_LOGS_DIR = path.join(path.dirname(path.abspath(__file__)), "logs/")
DEFAULT_EMB_NORMALIZATION = True
INIT_TEMP = 0.07


class JESPR(pl.LightningModule):
    def __init__(
        self,
        esm2: ESM2,
        esm2_alphabet: Alphabet,
        esm_if: GVPTransformerModel,
        esm_if_alphabet: Alphabet,
        **kwargs,
    ) -> None:
        """
        JESPR Model

        Args:
            esm2 (ESM2): ESM-2 Model
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if (GVPTransformerModel): ESM-IF Model
            esm_if_alphabet (Alphabet): ESM-IF Alphabet

        Keyword Args:
            comb_emb_size (int): Combined Embedding Size. Defaults to DEFAULT_COMBINED_EMB_SIZE.
            norm_emb (bool): Normalize Embeddings. Defaults to DEFAULT_EMB_NORMALIZATION.
            optim_args (dict): Optimizer Arguments. Defaults to {"lr": DEFAULT_LR}.
        """
        super().__init__()

        self.esm2, self.esm2_alphabet = esm2, esm2_alphabet
        self.esm_if, self.esm_if_alphabet = esm_if, esm_if_alphabet

        # Model params
        esm2_out_size = self.esm2.lm_head.dense.out_features
        esm_if_out_size = self.esm_if.encoder.layers[-1].fc2.out_features
        comb_emb_size = kwargs.get("comb_emb_size", DEFAULT_COMBINED_EMB_SIZE)

        self.num_esm2_layers = len(self.esm2.layers)

        # Linear projection to DEFAULT_COMBINED_EMB_SIZE dim
        self.structure_emb_linear = nn.Linear(esm_if_out_size, comb_emb_size)
        self.seq_emb_linear = nn.Linear(esm2_out_size, comb_emb_size)
        self.seq_layer_norm = nn.LayerNorm(comb_emb_size)
        self.str_layer_norm = nn.LayerNorm(comb_emb_size)
        
        # For scaling the cosing similarity score
        self.temperature = nn.Parameter(torch.tensor(INIT_TEMP))

        self.optim_args = kwargs.get("optim_args", {"lr": DEFAULT_LR})
        self.norm_emb = kwargs.get("norm_emb", DEFAULT_EMB_NORMALIZATION)

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

        # Layer Norm
        seq_embeddings = self.seq_layer_norm(seq_embeddings)
        structure_embeddings = self.str_layer_norm(structure_embeddings)

        # Batch Size * Residue Length * Embedding Size
        B, _, E = seq_embeddings.shape

        # Set batch size for logging
        self.batch_size = B

        pooled_seq_embeddings = torch.empty(B, E, device=seq_embeddings.device)
        pooled_structure_embeddings = torch.empty_like(pooled_seq_embeddings)

        batch_padding_lens = (tokens != ESM2_PADDING_IDX).sum(1)

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

        if self.norm_emb:
            # Normalize the embeddings
            pooled_seq_embeddings = self.normalize_embeddings(
                pooled_seq_embeddings
            )
            pooled_structure_embeddings = self.normalize_embeddings(
                pooled_structure_embeddings
            )

        # Calculating the Loss
        # text = seq, image = structure
        logits_per_structure = (
            self.temperature
            * pooled_structure_embeddings
            @ pooled_seq_embeddings.T
        )
        logits_per_seq = (
            self.temperature
            * pooled_seq_embeddings
            @ pooled_structure_embeddings.T
        )

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

    def normalize_embeddings(self, embeddings: torch.tensor) -> torch.tensor:
        """Normalize Embeddings
         Taken from https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L362

        Args:
            embeddings (torch.tensor): Embedding. (B, E)

        Returns:
            torch.tensor: Normalized Embeddings. (B, E)
        """
        return embeddings / embeddings.norm(dim=1, keepdim=True)

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
        acc_str = torch.sum(am_logits_per_structure == truths).item()/b
        acc_seq = torch.sum(am_logits_per_seq == truths).item()/b

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
        return torch.optim.Adam(
            self.parameters(), **self.optim_args
        )


def train_jespr(**kwargs):
    """
    Train Function for JESPR

    Kwargs:
        batch_size (int): Batch Size. Defaults to DEFAULT_BATCH_SIZE.
        esm2_model_type (str): ESM2 Model Type. Defaults to DEFAULT_ESM2_MODEL_TYPE.
        esm_if_model_type (str): ESM-IF Model Type. Defaults to DEFAULT_ESM_IF_MODEL_TYPE.
        project_name (str): Project Name for Weights & Biases. Defaults to DEFAULT_PROJECT_NAME.
        run_name (str): Run Name for Weights & Biases. Defaults to None.
        lr (float): Learning Rate. Defaults to DEFAULT_LR.
        log_every_n_steps (int): Log every n steps. Defaults to DEFAULT_LOG_EVERY_N_STEPS.
        epochs (int): Number of epochs. Defaults to DEFAULT_MAX_EPOCHS.
    """
    from lightning.pytorch import Trainer
    from pytorch_lightning.loggers import WandbLogger
    from data import ESMDataLightning
    import wandb

    # obtain all kwargs
    batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
    esm2_model_type = kwargs.get("esm2_model_type", DEFAULT_ESM2_MODEL_TYPE)
    esm_if_model_type = kwargs.get(
        "esm_if_model_type", DEFAULT_ESM_IF_MODEL_TYPE
    )
    project_name = kwargs.get("project_name", DEFAULT_PROJECT_NAME)
    run_name = kwargs.get("run_name", None)
    lr = kwargs.get("lr", DEFAULT_LR)
    log_every_n_steps = kwargs.get(
        "log_every_n_steps", DEFAULT_LOG_EVERY_N_STEPS
    )
    epochs = kwargs.get("epochs", DEFAULT_MAX_EPOCHS)
    val_check_interval = kwargs.get(
        "val_check_interval", DEFAULT_VAL_CHECK_ITVL
    )

    # Load ESM Models
    print("Initializing JESPR...")
    esm2, alphabet_2 = load_esm_2(esm2_model_type)
    esm_if, alphabet_if = load_esm_if(esm_if_model_type)

    jespr = JESPR(
        esm2=esm2,
        esm_if=esm_if,
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        lr=lr,
    )

    print("Loading DataModule...")
    esm_data_lightning = ESMDataLightning(
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        batch_size=batch_size,
    )

    wandb_logger = WandbLogger(
        project=project_name, name=run_name, save_dir=DEFAULT_LOGS_DIR
    )

    # add all hyperparams to confid
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["learning_rate"] = lr
    wandb_logger.experiment.config["esm2_model_type"] = esm2_model_type
    wandb_logger.experiment.config["esm_if_model_type"] = esm_if_model_type
    wandb_logger.experiment.config["log_every_n_steps"] = log_every_n_steps
    wandb_logger.experiment.config["val_check_interval"] = val_check_interval
    wandb_logger.experiment.config["epochs"] = epochs

    trainer = Trainer(
        accelerator="auto",
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        logger=wandb_logger,
        max_epochs=epochs,
    )

    print("Starting Training...")
    trainer.fit(jespr, datamodule=esm_data_lightning)
    print("Training Complete!")
    wandb.finish()
