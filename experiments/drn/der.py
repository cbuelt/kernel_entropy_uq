# Implementation of the DRN deep evidential regression.

import datetime
import os

import lightning as L
import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from data import WeatherBenchModule
from models import EmbeddingMLP, LightningDER


def train(coeff: float = 0.01):
    seed_everything(0)
    d_time = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    identifier = f"mlp_{coeff}"
    ckpt_path = "results/drn/der/checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)
    wandb.init(
        project="kernel-entropy",
        id=identifier + d_time,
    )
    logger = WandbLogger(project="kernel-entropy")

    dm = WeatherBenchModule(train_batch_size=1024, num_workers=6)
    mlp = EmbeddingMLP(n_outputs=4)
    model = LightningDER(mlp, coeff=coeff)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=identifier,
        monitor="val_loss",  # monitor validation loss
        mode="min",
        save_top_k=1,
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=4, mode="min")

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [early_stopping_callback, checkpoint_callback, lr_monitor_callback]

    trainer = L.Trainer(
        max_epochs=30,
        log_every_n_steps=5,
        accelerator="gpu",
        devices="auto",
        enable_progress_bar=False,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    train(coeff=1.0)
