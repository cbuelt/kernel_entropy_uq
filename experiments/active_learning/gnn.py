# This file contains code adapted from:
# "gnn-post-processing" by hits-mli
# Repository: https://github.com/hits-mli/gnn-post-processing
# Paper: Feik et al. Graph Neural Networks and Spatial Information Learning for Post-Processing Ensemble Weather Forecasts (2024)

import os
from itertools import compress
import time
import argparse

import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from data import (
    load_dataframes,
    load_distances,
    normalize_features_and_create_graphs,
)
from models import Multigraph, NormalCRPS
from uq import GaussianUQMeasure


def get_data(config, leadtime):
    # Load Data ######################################################################
    dataframes = load_dataframes(mode="train", leadtime=leadtime)
    dist = load_distances(dataframes["stations"])

    # Normalize Features and Create Graphs ###########################################
    graphs_train_rf, tests = normalize_features_and_create_graphs(
        training_data=dataframes["train"],
        valid_test_data=[dataframes["test_rf"], dataframes["test_f"]],
        mat=dist,
        max_dist=config["max_dist"],
    )
    _, graphs_test = tests

    # ! Hacky wack of getting the targets
    train_targets = dataframes["train"][1]
    train_targets = torch.tensor(train_targets.t2m.values) - 273.15
    eval_targets = dataframes["test_f"][1]
    eval_targets = torch.tensor(eval_targets.t2m.values) - 273.15

    return graphs_train_rf, graphs_test, train_targets, eval_targets


def setup_experiment():
    if not os.path.exists("data/EUPPBench/dataframes/final_train"):
        os.makedirs("data/EUPPBench/dataframes/final_train")

    # Use setup 24h
    leadtime = "24h"
    config = {
        "batch_size": 8,
        "gnn_hidden": 265,
        "gnn_layers": 2,
        "heads": 8,
        "lr": 0.0002,
        "max_dist": 100,
    }

    dt = time.time()
    temp_base_path = f"results/active_learning/temp_{dt}/"
    if os.path.exists(temp_base_path):
        os.system("rm -rf "+temp_base_path)
    os.makedirs(temp_base_path)

    return config, leadtime, temp_base_path


def get_model(config, graphs_train_rf):
    print("[INFO] Creating model...")
    emb_dim = 20
    in_channels = graphs_train_rf[0].x.shape[1] + emb_dim - 1
    multigraph = Multigraph(
        embedding_dim=emb_dim,
        in_channels=in_channels,
        hidden_channels_gnn=config["gnn_hidden"],
        out_channels_gnn=config["gnn_hidden"],
        num_layers_gnn=config["gnn_layers"],
        heads=config["heads"],
        hidden_channels_deepset=config["gnn_hidden"],
        optimizer_class=AdamW,
        optimizer_params=dict(lr=config["lr"]),
    )
    return multigraph


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


def train_ensemble_member(model, temp_path, ensemble_id, round, dataloader):
    ckpt_path = os.path.join(temp_path, f"round_{round + 1}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=f"{ensemble_id}",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
    )

    # Train Model ###################################################################
    print(f"[INFO] Training round {round} for member {ensemble_id}")
    trainer = L.Trainer(
        max_epochs=(round+1)*EPOCHS,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        enable_progress_bar=False,
        callbacks=checkpoint_callback,
    )
    if round > 0: # Load checkpoint from previous round
        load_ckpt_path = os.path.join(temp_path, f"round_{round}") + f"/{ensemble_id}.ckpt"
        trainer.fit(model=model, train_dataloaders=dataloader, ckpt_path = load_ckpt_path)
    else:
        trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":
    # Add measure via parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-measure", type=str, default="log")
    args = argparser.parse_args()

    RUNS = 1 #3
    ROUNDS = 20
    N_MEMBERS = 5 #10
    EPOCHS = 3
    START_SAMPLES = 100 #50
    NEW_SAMPLES = 100 #50
    MEASURE = args.measure
    criterion = NormalCRPS(mask = True, ensemble = True, reduction = None)
    loss = np.empty((RUNS, ROUNDS, 2))

    # Setup experiment
    config, leadtime, temp_base_path = setup_experiment()

    # Get data and define test loader
    train_list, graphs_test, train_targets, eval_targets = get_data(config, leadtime)
    test_loader = DataLoader(graphs_test, batch_size=1, shuffle=False, num_workers=4)
    n_samples = len(train_list)

    # Implement for run in run
    for run in range(RUNS):
        # Set seed
        torch.manual_seed(run)
        np.random.seed(run)
        # Configure permutations
        perm = np.random.permutation(n_samples)
        # Get boolean lists
        train_index = perm[0:START_SAMPLES]
        train_bool_index = np.isin(perm, train_index).tolist()
        pool_bool_index = (~np.array(train_bool_index)).tolist()
        pool_index = perm[pool_bool_index]

        # Take time
        start_time = time.time()

        # Create temp dir
        temp_path = f"{temp_base_path}/run{run}/"
        if os.path.exists(temp_path):
            os.system("rm -rf "+temp_path)
        os.makedirs(temp_path)
        # Reduce data
        for round in range(ROUNDS):
            # Define train loader with subset
            x_train = list(compress(train_list, train_bool_index))
            x_pool = list(compress(train_list, pool_bool_index))

            train_loader = DataLoader(
                x_train, batch_size=config["batch_size"], shuffle=True, num_workers=4
            )
            pool_loader = DataLoader(
                x_pool, batch_size=config["batch_size"], shuffle=False, num_workers=4
            )

            # Create checkpoint path
            ckpt_path = os.path.join(temp_path, f"round_{round + 1}")
            os.makedirs(ckpt_path)

            for m in range(N_MEMBERS):
                # Train model for x epochs
                multigraph = get_model(config, train_list)
                train_ensemble_member(multigraph, temp_path, m, round, train_loader)

            # Predict test loader
            test_pred = get_ensemble_pred(
                model=multigraph, model_path=ckpt_path, dataloader=test_loader
            )
            # Evaluate metrics on test set and log
            res = criterion(test_pred, eval_targets)
            loss[run, round,:] = res.mean(), res.var()

            if round < ROUNDS-1:
                # Predict train set and get epistemic uncertainty measure
                train_pred = get_ensemble_pred(
                    model=multigraph, model_path=ckpt_path, dataloader=pool_loader
                )
                # EU
                uq_measure = GaussianUQMeasure(train_pred, gamma = 0.1)
                _, eu, _ = uq_measure.get_uncertainties(measure = MEASURE)
                eu = eu.reshape(-1, 122).mean(dim = -1) # Aggregate over stations

                # Sort indices and select pool
                new_indices = np.argsort(eu)[-NEW_SAMPLES:]
                selected = pool_index[new_indices]
                train_index = np.concatenate([train_index, selected])
                train_bool_index = np.isin(perm, train_index).tolist()
                pool_bool_index = (~np.array(train_bool_index)).tolist()
                pool_index = perm[pool_bool_index]


    # Evaluation time
    end_time = time.time() - start_time
    print(f"Training took {end_time:.4f}s")

    # Setup results dir
    results_dir = "results/active_learning/results/"
    os.makedirs(results_dir, exist_ok = True)
    np.save(f"{results_dir}{MEASURE}.npy", loss)
    #os.system("rm -rf "+temp_base_path)
