from os import listdir
from os import path
import json
import argparse
from argparse import Namespace
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    LearningRateMonitor,
    EarlyStopping,
)
from esm.data import Alphabet
from lightning.pytorch.loggers import WandbLogger

import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
from finetune_models import JESPR_Regression
from finetune_data import StabilityLightning

sys.path.append(os.path.abspath(os.path.join(__file__, "..")))
from modules import load_sequence_encoder, SequenceEncoder
from trainer import build_callbacks, build_logger


def num_params(model: torch.nn.Module) -> int:
    """Calculate the number of parameters

    Args:
        model (nn.Module): PyTorch Model
    Returns:
        int: Total Number of params
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_seq_encoder(seq_encoder_args: dict) -> SequenceEncoder:
    """Build Seq Encoder from Args

    Args:
        seq_encoder_args (dict): Sequence Encoder Args

    Returns:
        SequenceEncoder: Sequence Encoder
    """
    seq_encoder, alphabet = load_sequence_encoder(seq_encoder_args["config"])

    if seq_encoder_args["pretrained_ckpt"]:
        print("Loading Sequence encoder weights from checkpoint...")

        with open(seq_encoder_args["pretrained_ckpt"], "rb") as f:
            ckpt = torch.load(f)
            seq_encoder.load_state_dict(ckpt["state_dict"])
    return seq_encoder, alphabet


def build_finetune_model(
    seq_encoder: SequenceEncoder, model_type: str, optim_args: dict
):
    """Build Finetuning Model based on Sequence Encoder and Model Type

    Args:
        seq_encoder (SequenceEncoder): Sequence Encoder as returned from build_seq_encoder
        model_type (str): Model Type.
        optim_args (dict): Optimizer Args
    """
    assert model_type in [
        "regression",
        "classification",
    ], f"Invalid Model Type: {model_type}"
    if model_type == "regression":
        model = JESPR_Regression(
            seq_encoder=seq_encoder,
            optim_args=optim_args,
        )
    elif model_type == "classification":
        raise NotImplementedError("Classification not implemented yet")
    return model


def load_data_module(esm2_alphabet: Alphabet, data_args: dict):
    assert data_args["task_name"] in [
        "stability"
    ], f"Invalid Task Name: {data_args['task_name']}"
    if data_args["task_name"] == "stability":
        data_module = StabilityLightning(
            esm2_alphabet=esm2_alphabet, args=Namespace(**data_args)
        )
    return data_module


def main(args, mode="train"):
    """Parse all args and run training loop

    Args:
        args (dict): All arguments from the config file.
        mode (str): mode to run in (train/exp_no_trainer/exp_with_trainer). Defaults to "train".
    """
    assert mode in ["train", "exp_no_trainer", "exp_with_trainer"]

    # ___________ ESM Args __________________________ #
    seq_encoder_args = args["sequence_encoder"]

    # ___________ Data ______________________________ #
    data_args = args["data"]
    batch_size = args["data"]["batch_size"]

    # ___________ optim _____________________________ #
    optim_args = args["optim"]  # Args to be passed to Optimizer

    # ___________ trainer ___________________________ #
    accelerator = args["trainer"]["accelerator"]
    precision = args["trainer"]["precision"]
    devices = args["trainer"]["devices"]
    epochs = args["trainer"]["epochs"]
    log_every_n_steps = args["trainer"]["log_every_n_steps"]  # batch steps not epochs
    enable_progress_bar = args["trainer"]["enable_progress_bar"]
    val_check_interval = args["trainer"]["val_check_interval"]  # epochs
    check_val_every_n_epoch = args["trainer"]["check_val_every_n_epoch"]
    limit_train_batches = args["trainer"]["limit_train_batches"]
    limit_val_batches = args["trainer"]["limit_val_batches"]
    overfit_batches = args["trainer"]["overfit_batches"]
    accumulate_grad_batches = args["trainer"]["accumulate_grad_batches"]
    detect_anomaly = args["trainer"]["detect_anomaly"]
    grad_clip_val = args["trainer"]["grad_clip_val"]
    grad_clip_algorithm = args["trainer"]["grad_clip_algorithm"]
    checkpoint_id = args["trainer"]["checkpoint_id"]
    callback_args = args["trainer"]["callbacks"]
    callbacks = build_callbacks(callback_args)

    # ___________ Logger ______________________________ #
    logger_args = args["logger"]

    seq_encoder, alphabet = build_seq_encoder(seq_encoder_args)

    print("Loading DataModule...")
    datamodule = load_data_module(esm2_alphabet=alphabet, data_args=data_args)

    print(f"Initializing {data_args['task_name']} Model...")
    finetune_model = build_finetune_model(
        seq_encoder=seq_encoder,
        model_type=seq_encoder_args["type"],
        optim_args=optim_args,
    )

    if mode == "exp_no_trainer":
        return (finetune_model, datamodule)

    print("Initializing Wandb Logger...")
    wandb_logger = build_logger(logger_args, checkpoint_id)

    print("Initializing Trainer...")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        detect_anomaly=detect_anomaly,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=wandb_logger,
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm=grad_clip_algorithm,
        overfit_batches=overfit_batches,
    )

    # Wandb object only available to rank 0
    if trainer.global_rank == 0:
        args["num_params"] = num_params(finetune_model)
        # Add all config params
        wandb_logger.experiment.config.update(args)

    if mode == "exp_with_trainer":
        return (finetune_model, datamodule, trainer, wandb_logger)

    print("Starting Training...")

    trainer.fit(
        model=finetune_model,
        datamodule=datamodule,
        # ckpt_path=ckpt_path,
    )
    return None


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-f",
        "--config_path",
        type=str,
        default=path.join(path.dirname(__file__), "config/trainer_config.json"),
        help="Path for the Train Config Json File",
    )

    config_file_path = arg_parser.parse_known_args()[0].config_path
    with open(config_file_path, "r") as f:
        args = json.load(f)
    main(args=args)
