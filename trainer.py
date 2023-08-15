from argparse import Namespace
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

import torch
from data import ESMDataLightning
from jespr import JESPR
from modules import load_sequence_encoder, load_structure_encoder


def main(args, mode="train"):
    """Parse all args and run training loop

    Args:
        args (dict): All arguments from the config file.
        mode (str): mode to run in (train/exp_no_trainer/exp_with_trainer). Defaults to "train".
    """
    assert mode in ["train", "exp_no_trainer", "exp_with_trainer"]
    # ___________ JESPR & Submodules ________________ #
    seq_encoder_args = args["jespr"]["sequence_encoder"]
    struct_encoder_args = args["jespr"]["structure_encoder"]
    joint_embedding_dim = args["jespr"]["joint_embedding_dim"]

    # add joint_embedding_dim & norm_embedding to esm2 and esm-if args
    seq_encoder_args["joint_embedding_dim"] = joint_embedding_dim
    struct_encoder_args["joint_embedding_dim"] = joint_embedding_dim

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
    log_every_n_steps = args["trainer"][
        "log_every_n_steps"
    ]  # batch steps not epochs
    enable_progress_bar = args["trainer"]["enable_progress_bar"]
    val_check_interval = args["trainer"]["val_check_interval"]  # epochs
    check_val_every_n_epoch = args["trainer"]["check_val_every_n_epoch"]
    limit_train_batches = args["trainer"]["limit_train_batches"]
    limit_val_batches = args["trainer"]["limit_val_batches"]
    overfit_batches = args["trainer"]["overfit_batches"]
    accumulate_grad_batches = args["trainer"]["accumulate_grad_batches"]
    stochastic_weight_averaging = args["trainer"][
        "stochastic_weight_averaging"
    ]
    stochastic_weight_averaging_lr = args["trainer"][
        "stochastic_weight_averaging_lr"
    ]
    detect_anomaly = args["trainer"]["detect_anomaly"]
    grad_clip_val = args["trainer"]["grad_clip_val"]
    grad_clip_algorithm = args["trainer"]["grad_clip_algorithm"]

    # ___________ Meta ______________________________ #
    project_name = args["meta"]["project_name"]
    run_name = args["meta"]["run_name"]
    logs_dir = args["meta"]["logs_dir"]
    checkpoint = args["meta"]["checkpoint"]

    seq_encoder, alphabet_2 = load_sequence_encoder(seq_encoder_args)
    struct_encoder, alphabet_if = load_structure_encoder(struct_encoder_args)

    print("Loading DataModule...")
    esm_data_lightning = ESMDataLightning(
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        args=Namespace(**data_args),
    )

    print("Initializing JESPR...")
    jespr = JESPR(
        sequence_encoder=seq_encoder,
        structure_encoder=struct_encoder,
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        optim_args=optim_args,
    )

    if mode == "exp_no_trainer":
        if checkpoint["lightning_checkpoint"]:
            print(
                f"Loading Checkpoint from : {checkpoint['lightning_checkpoint']}..."
            )
            jespr = jespr.load_from_checkpoint(
                checkpoint["lightning_checkpoint"]
            )
        return (jespr, esm_data_lightning)

    print("Initializing Wandb Logger...")

    if checkpoint["wandb_run_id"]:
        print(f"Loading Wandb Run...: {checkpoint['wandb_run_id']}")
        wandb_logger = WandbLogger(
            project=project_name,
            save_dir=logs_dir,
            resume="must",
            id=checkpoint["wandb_run_id"],
        )
    else:
        wandb_logger = WandbLogger(
            project=project_name, save_dir=logs_dir, name=run_name
        )

    callbacks = [
        LearningRateMonitor(logging_interval="step", log_momentum=False)
    ]
    if stochastic_weight_averaging:
        callbacks.append(
            StochasticWeightAveraging(swa_lrs=stochastic_weight_averaging_lr)
        )

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
        # Add all config params
        wandb_logger.experiment.config.update(args)

    if mode == "exp_with_trainer":
        return (jespr, esm_data_lightning, trainer, wandb_logger)

    print("Starting Training...")
    ckpt_path = (
        checkpoint["lightning_checkpoint"]
        if checkpoint["lightning_checkpoint"]
        else None
    )
    trainer.fit(
        model=jespr,
        datamodule=esm_data_lightning,
        ckpt_path=ckpt_path,
    )
    return None


if __name__ == "__main__":
    main()
