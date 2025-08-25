import os
import tempfile
from functools import partial
import datetime
import wandb
import matplotlib.pyplot as plt
import torch
import pathlib
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from data import WeatherBenchModule
from models import LightningDRN, EmbeddingMLP



def train(id:int = 0):
    seed_everything(id)
    d_time = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    identifier = 'mlp_' + str(id)
    ckpt_path = "results/drn/checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)
    wandb.init(
        project="kernel-entropy",
        id= identifier + d_time,
    )
    logger = WandbLogger(project="kernel-entropy")

    dm = WeatherBenchModule(train_batch_size=1024, num_workers=8)
    mlp = EmbeddingMLP()
    model = LightningDRN(mlp)


    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=identifier,
        monitor="val_loss",  # monitor validation loss
        mode="min",
        save_top_k=1,
    )

    early_stopping_callback = EarlyStopping(
                monitor="val_loss", patience=4, mode="min"
            )

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    callbacks = [early_stopping_callback, checkpoint_callback, lr_monitor_callback]

    trainer = Trainer(
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
    ensembles = [4]#[1,2,3,4,5,6,7,8,9,10]
    for e in ensembles:
        train(e)