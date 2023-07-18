from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger

from data import ESMDataLightning
from jespr import JESPR
from util import load_esm_2, load_esm_if


def main(args):
    """Parse all args and run training loop

    Args:
        args (dict): All arguments from the config file. 
    """
    # ___________ JESPR & Submodules ________________ #
    esm2_args = args["jespr"]["esm2"]
    esm_if_args = args["jespr"]["esm_if"]
    comb_emb_size = args["jespr"]["comb_emb_size"]  # Combined Embedding Size
    norm_emb = args["jespr"]["norm_emb"]  # Normalize Individual seq. & str. Embeddings 

    # ___________ Data ______________________________ #
    data_split_ratio = args["data"]["data_split_ratio"]
    max_seq_len = args["data"]["max_seq_len"]
    data_dir = args["data"]["data_dir"]
    train_shuffle = args["data"]["train_shuffle"]
    val_shuffle= args["data"]["val_shuffle"]
    pin_memory = args["data"]["pin_memory"]
    num_workers = args["data"]["num_workers"]
    
    # ___________ trainer ___________________________ #
    accelerator = args["trainer"]["accelerator"]
    precision = args["trainer"]["precision"]
    devices = args["trainer"]["devices"]
    batch_size = args["trainer"]["batch_size"]
    epochs = args["trainer"]["epochs"]
    log_every_n_steps = args["trainer"]["log_every_n_steps"]  # batch steps not epochs
    enable_progress_bar = args["trainer"]["enable_progress_bar"]
    val_check_interval = args["trainer"]["val_check_interval"]  # epochs
    limit_train_batches = args["trainer"]["limit_train_batches"]
    accumulate_grad_batches = args["trainer"]["accumulate_grad_batches"]
    stochastic_weight_averaging = args["trainer"]["stochastic_weight_averaging"]
    stochastic_weight_averaging_lr = args["trainer"]["stochastic_weight_averaging_lr"]
    detect_anomaly = args["trainer"]["detect_anomaly"]
    grad_clip_val = args["trainer"]["grad_clip_val"]
    grad_clip_algorithm = args["trainer"]["grad_clip_algorithm"]

    # ___________ optim _____________________________ #
    optim_args = args["optim"]  # Args to be passed to Adam
    lr = args["optim"]["lr"]

    # ___________ Meta ______________________________ #
    project_name = args["meta"]["project_name"]
    run_name = args["meta"]["run_name"]
    logs_dir = args["meta"]["logs_dir"]

    print("Initializing JESPR...")
    esm2, alphabet_2 = load_esm_2(esm2_args)
    esm_if, alphabet_if = load_esm_if(esm_if_args)

    jespr = JESPR(
        esm2=esm2,
        esm_if=esm_if,
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        lr=lr,
        norm_emb=norm_emb,
        comb_emb_size=comb_emb_size,
        optim_args=optim_args,
    )

    print("Loading DataModule...")
    esm_data_lightning = ESMDataLightning(
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        batch_size=batch_size,
        data_dir=data_dir,
        split_ratio=data_split_ratio,
        max_seq_len=max_seq_len,
        train_shuffle=train_shuffle,
        val_shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print("Initializing Wandb Logger...")
    wandb_logger = WandbLogger(
        project=project_name, name=run_name, save_dir=logs_dir
    )

    callbacks = []
    if stochastic_weight_averaging:
        callbacks.append(StochasticWeightAveraging(swa_lrs=stochastic_weight_averaging_lr))
    
    print("Initializing Trainer...")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        detect_anomaly=detect_anomaly,
        val_check_interval=val_check_interval,
        logger=wandb_logger,
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm=grad_clip_algorithm,
    )

    # Wandb object only available to rank 0
    if trainer.global_rank == 0:
        # Add all config params
        wandb_logger.experiment.config.update(args)

    print("Starting Training...")
    trainer.fit(
        model=jespr, datamodule=esm_data_lightning
    )

if __name__ == "__main__":
    main()