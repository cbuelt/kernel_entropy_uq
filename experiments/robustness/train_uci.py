from functools import partial
import os

import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning_uq_box.datamodules import UCIRegressionDatamodule
from lightning_uq_box.datasets import UCIConcrete, UCIEnergy, UCIYacht
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import MVERegression


def train_uci(data_directory, experiment_name, ensemble_id, ckpt_path, train_size, batch_size, lr, epochs, train_distortion = 0):
    # Define dataloader
    dm = UCIRegressionDatamodule(
        dataset_name=experiment_name,
        root=data_directory,
        train_size=train_size,
        batch_size=batch_size,
        train_distortion=train_distortion
    )
    sample = next(iter(dm.train_dataloader()))
    n_input = sample["input"].shape[-1]
    model = MLP(n_inputs = n_input, n_outputs = 2, activation_fn=nn.ReLU(), n_hidden = [256, 512, 1024])
    uq_method = MVERegression(model, optimizer=partial(torch.optim.Adam, lr=lr), burnin_epochs = 100)

    # Callbacks
    if train_distortion > 0:
        filename = f"{experiment_name}_distorted_{train_distortion}"
    else:
        filename = f"{experiment_name}_{ensemble_id}"
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k = 1, filename = filename, dirpath = ckpt_path)
    clbks = [checkpoint_callback]
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
    devices="auto",
    enable_progress_bar=True,
    callbacks=clbks,
)

    trainer.fit(uq_method, dm)


if __name__ == "__main__":
    # Configuration for UCI experiments
    epochs = 10000
    batch_size = 256
    train_size = 0.9
    lr = 0.0001
    data_path = "data/uci/"
    os.makedirs(data_path, exist_ok=True)
    experiments = ["energy", "yacht"]
    ckpt_path = "results/robustness/checkpoints"
    os.makedirs(ckpt_path, exist_ok=True)

    # Ensemble configuration
    n_ensembles = 2

    # Robustness configuration
    distortion = [2]


    for experiment in experiments:
        # General ensemble training
        for ensemble_id in range(n_ensembles):
            print(f"Training {experiment} with ensemble ID {ensemble_id}")
            train_uci(data_path, experiment, ensemble_id, ckpt_path, train_size, batch_size, lr, epochs)
        # Distorted training
        for dist in distortion:
            print(f"Training {experiment} with distortion {dist}")
            train_uci(data_path, experiment, ensemble_id, ckpt_path, train_size, batch_size, lr, epochs, train_distortion=dist)
