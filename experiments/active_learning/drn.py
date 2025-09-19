import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from data import load_dataframes, summary_statistics
from models import (
    NLL,
    GaussianKernelScore,
    NormalCRPS,
    SquaredError,
    StationDRN,
    drop_nans,
    normalize_features,
)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-loss", type=str, default="log")
    args = argparser.parse_args()

    if not os.path.exists("data/EUPPBench/dataframes/final_train"):
        os.makedirs("data/EUPPBench/dataframes/final_train")

    # Use setup 24h
    leadtime = "24h"
    config = {
        "batch_size": 4096,
        "lr": 0.01,
        "hidden_channels": [256],
        "max_epochs": 26,
        "only_summary": "True",
    }

    LOSS = "test"#args.loss
    GAMMA = 1
    N_ENSEMBLES = 10



    dt = time.time()
    base_path = f"results/active_learning/{LOSS}/"
    ckpt_path = f"results/active_learning/{LOSS}/checkpoints/"
    if os.path.exists(base_path):
        os.system("rm -rf " + base_path)
    os.makedirs(base_path)
    os.makedirs(ckpt_path)

    dataframes = load_dataframes(mode="train", leadtime=leadtime)
    dataframes = summary_statistics(dataframes)
    dataframes.pop("stations")

    for X, y in dataframes.values():
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    ################

    train, (test_rf, test_f) = normalize_features(
        training_data=dataframes["train"],
        valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
    )

    # Drop Nans ####
    train = drop_nans(train)
    test = test_f
    test = drop_nans(test)

    y_scaler = StandardScaler(with_std=False)
    y_scaler = y_scaler.fit(train[1][["t2m"]])

    train_dataset = TensorDataset(
        torch.Tensor(train[0].to_numpy()),
        torch.Tensor(y_scaler.transform(train[1][["t2m"]])),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )

    test_dataset = TensorDataset(
        torch.Tensor(test[0].to_numpy()),
        torch.Tensor(y_scaler.transform(test[1][["t2m"]])),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    embed_dim = 20
    in_channels = train[0].shape[1] + embed_dim - 1

    ens_pred = []
    # Iterate over ensembles
    for n in range(N_ENSEMBLES):
        drn = StationDRN(
            in_channels=in_channels,
            hidden_channels=config["hidden_channels"],
            embedding_dim=embed_dim,
            loss=LOSS,
            optimizer_class=AdamW,
            optimizer_params=dict(lr=config["lr"]),
            gamma=GAMMA,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_path,
            filename=f"ens_{n}",
            monitor="train_loss",
            mode="min",
            save_top_k=1,
        )
        trainer = L.Trainer(
            max_epochs=config["max_epochs"],
            log_every_n_steps=1,
            accelerator="gpu",
            enable_progress_bar=True,
            enable_model_summary=True,
            callbacks=checkpoint_callback,
        )
        trainer.fit(model=drn, train_dataloaders=train_loader)

        preds = trainer.predict(model=drn, dataloaders=test_loader)
        preds = torch.cat(preds, dim=0)
        # Reverse transform of the y_scaler (only on the mean)
        preds[:, 0] = torch.Tensor(
            y_scaler.inverse_transform(preds[:, 0].view(-1, 1))
        ).flatten()
        ens_pred.append(preds)

    # Save pred
    stacked = torch.stack(ens_pred, dim=-1)
    np.save(base_path + "pred.npy", ens_pred)

    # Targets #######################################################################
    targets = test[1]
    targets = torch.Tensor(targets.t2m.values)

    final_loss = NormalCRPS()
    # Create ensemble prediction
    mu = stacked[:, 0:1].mean(dim=-1)
    sigma = torch.sqrt(
        stacked[:, 1:2].mean(dim=-1) + torch.var(stacked[:, 0:1], dim=-1)
    )
    test_pred = torch.cat([mu, sigma], dim=1)

    # Save targets
    np.save(base_path + f"targets.npy", targets)

    res = final_loss(test_pred, targets.unsqueeze(1))
    print("#############################################")
    print("#############################################")
    print(f"final crps: {res.item()}")
    print("#############################################")
    print("#############################################")

    # Save Results ##################################################################

    # Create Log File ###############################################################
    log_file = os.path.join(base_path, "f.txt")
    with open(log_file, "w") as f:
        f.write(f"Leadtime: {leadtime}\n")
        f.write(f"Final crps: {res.item()}")
