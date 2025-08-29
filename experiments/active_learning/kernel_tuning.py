# This file contains code adapted from:
# "gnn-post-processing" by hits-mli
# Repository: https://github.com/hits-mli/gnn-post-processing
# Paper: Feik et al. Graph Neural Networks and Spatial Information Learning for Post-Processing Ensemble Weather Forecasts (2024)

import os
import time

import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from data import load_dataframes, summary_statistics
from models import NormalCRPS, StationDRN, drop_nans, normalize_features
from uq import GaussianUQMeasure


def get_data(leadtime):
    dataframes = load_dataframes(mode="train", leadtime=leadtime)
    dataframes = summary_statistics(dataframes)
    dataframes.pop("stations")

    for X, y in dataframes.values():
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    ################

    train, (_, test_f) = normalize_features(
        training_data=dataframes["train"],
        valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
    )

    # Drop Nans ####
    train = drop_nans(train)
    test = test_f
    test = drop_nans(test)

    y_scaler = StandardScaler(with_std=False)
    y_scaler = y_scaler.fit(train[1][["t2m"]])

    train_targets = train[1]
    test_targets = test[1]

    return (
        train[0].to_numpy(),
        test[0].to_numpy(),
        train_targets,
        test_targets,
        y_scaler,
    )


def setup_experiment():
    if not os.path.exists("data/EUPPBench/dataframes/final_train"):
        os.makedirs("data/EUPPBench/dataframes/final_train")

    # Use setup 24h
    leadtime = "24h"
    config = {
        "batch_size": 128,
        "lr": 0.01,
        "hidden_channels": [256],
        "max_epochs": 26,
        "only_summary": "True",
    }

    base_path = f"results/active_learning/kernel_tuning/"
    if os.path.exists(base_path):
        os.system("rm -rf " + base_path)
    os.makedirs(base_path)

    results_path = os.path.join(base_path, f"results/")
    os.makedirs(results_path, exist_ok = True)

    return config, leadtime, base_path, results_path


def get_model(config, train, gamma):
    print("[INFO] Creating model...")
    embed_dim = 20
    in_channels = train.shape[1] + embed_dim - 1
    drn = StationDRN(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        embedding_dim=embed_dim,
        loss="crps",
        optimizer_class=AdamW,
        optimizer_params=dict(lr=config["lr"]),
        gamma=gamma,
    )
    return drn


def get_ensemble_pred(model, model_path, dataloader):
    trainer = L.Trainer(
        log_every_n_steps=1, accelerator="gpu", devices=1, enable_progress_bar=False
    )
    preds_list = []
    for ckpt_path in os.listdir(model_path):
        if ckpt_path.endswith(".ckpt"):
            print(f"[INFO] Loading ensemble member from {ckpt_path}")
            # Load Model from chekcpoint
            checkpoint = torch.load(os.path.join(model_path, ckpt_path))
            model.load_state_dict(checkpoint["state_dict"])
            preds = trainer.predict(model=model, dataloaders=[dataloader])
            preds = torch.cat(preds, dim=0)
            preds_list.append(preds)
    ensemble_pred = torch.stack(preds_list, dim=-1)  # Shape [N, 2, M]
    return ensemble_pred


def train_ensemble_member(model, ckpt_path, ensemble_id, round, dataloader):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(ckpt_path + f"round_{round}/"),
        filename=f"{ensemble_id}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
    )

    # Train Model ###################################################################
    trainer = L.Trainer(
        max_epochs=(round + 1) * EPOCHS,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        callbacks=checkpoint_callback,
    )
    if round > 0:  # Load checkpoint from previous round
        load_ckpt_path = (
            os.path.join(ckpt_path + f"round_{round-1}/{ensemble_id}.ckpt")
        )
        trainer.fit(model=model, train_dataloaders=dataloader, ckpt_path=load_ckpt_path)
    else:
        trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    ROUNDS = 40
    N_MEMBERS = 5
    EPOCHS = 3
    START_SAMPLES = 200
    NEW_SAMPLES = 200
    MEASURE = "kernel"
    GAMMAS = np.arange(0, 2.2, 0.2)
    criterion = NormalCRPS(reduction = None)
    loss = np.empty((ROUNDS, 2))

    # Setup experiment
    config, leadtime, base_path, results_path = setup_experiment()

    # Get data and define test loader
    train_array, test_array, train_targets, test_targets, y_scaler = get_data(leadtime)

    # Setup test loader
    y_test = torch.Tensor(y_scaler.transform(test_targets[["t2m"]]))
    test_dataset = TensorDataset(
        torch.Tensor(test_array),
        y_test,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers = 1
    )
    n_samples = train_array.shape[0]

    # Set seed
    torch.manual_seed(0)
    np.random.seed(0)
    # Configure permutations
    perm = np.random.permutation(n_samples)

    # Iterate over gammas
    for gamma in GAMMAS:
        # Create checkpoint path
        ckpt_base_path = os.path.join(base_path, f"checkpoints/")
        os.makedirs(ckpt_base_path)

        # Get boolean lists
        train_index = perm[0:START_SAMPLES]
        pool_index = perm[START_SAMPLES:]
        x_pool = train_array[pool_index]

        # Take time
        start_time = time.time()

        # Create temp dir
        # Reduce data
        for round in range(ROUNDS):
            # Define train loader with subset
            x_train = train_array[train_index]
            y_train = y_scaler.transform(train_targets[["t2m"]])[train_index]
            y_pool = y_scaler.transform(train_targets[["t2m"]])[pool_index]

            train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
            pool_dataset = TensorDataset(torch.Tensor(x_pool), torch.Tensor(y_pool))

            train_loader = DataLoader(
                train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1
            )
            pool_loader = DataLoader(
                pool_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1
            )

            # Create checkpoint path
            ckpt_path = os.path.join(ckpt_base_path, f"round_{round}")
            os.makedirs(ckpt_path)

            for m in range(N_MEMBERS):
                # Train model for x epochs
                model = get_model(config, train_array, gamma)
                train_ensemble_member(model, ckpt_base_path, m, round, train_loader)

            # Predict test loader
            test_pred = get_ensemble_pred(
                model=model, model_path=ckpt_path, dataloader=test_loader
            )
            # Create ensemble prediction
            mu = test_pred[:,0:1].mean(dim = -1)
            sigma = torch.sqrt(test_pred[:,1:2].mean(dim = -1) + torch.var(test_pred[:,0:1], dim = -1))
            test_pred = torch.cat([mu, sigma], dim = 1)
            # Evaluate metrics on test set and log
            res = criterion(test_pred, y_test)
            loss[round, :] = res.mean(), res.var()

            if round < ROUNDS - 1:
                # Predict train set and get epistemic uncertainty measure
                train_pred = get_ensemble_pred(
                    model=model, model_path=ckpt_path, dataloader=pool_loader
                )
                # EU
                uq_measure = GaussianUQMeasure(train_pred, gamma=gamma)
                _, eu, _ = uq_measure.get_uncertainties(measure=MEASURE)

                # Sort indices and select pool
                indices = np.argsort(eu)
                train_indices = indices[-NEW_SAMPLES:]
                train_index = np.concatenate([train_index, train_indices])
                pool_index = indices[:-NEW_SAMPLES]
                x_pool = x_pool[pool_index]

        # Evaluation time
        end_time = time.time() - start_time
        print(f"Training took {end_time:.4f}s")
        print(loss)

        np.save(f"{results_path}kernel_{gamma}.npy", loss)
        os.system("rm -rf " + ckpt_path)
