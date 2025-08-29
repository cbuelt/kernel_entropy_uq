import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning_uq_box.datamodules import UCIRegressionDatamodule
from lightning_uq_box.datasets import UCIConcrete, UCIEnergy, UCIYacht
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import DeepEnsembleRegression, MVERegression


from uq import GaussianUQMeasure


def train_uci(
    data_directory,
    experiment_name,
    ensemble_id,
    ckpt_path,
    train_size,
    batch_size,
    lr,
    epochs,
    train_distortion=None,
):
    # Define dataloader
    dm = UCIRegressionDatamodule(
        dataset_name=experiment_name,
        root=data_directory,
        train_size=train_size,
        batch_size=batch_size,
        train_distortion=train_distortion,
    )
    sample = next(iter(dm.train_dataloader()))
    n_input = sample["input"].shape[-1]
    model = MLP(
        n_inputs=n_input,
        n_outputs=2,
        activation_fn=nn.ReLU(),
        n_hidden=[256, 512, 1024],
    )
    uq_method = MVERegression(
        model, optimizer=partial(torch.optim.Adam, lr=lr), burnin_epochs=100
    )

    # Callbacks
    if train_distortion is not None:
        filename = f"{experiment_name}_distorted_{train_distortion}"
    else:
        filename = f"{experiment_name}_{ensemble_id}"
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss", save_top_k=1, filename=filename, dirpath=ckpt_path
    )
    clbks = [checkpoint_callback]
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices="auto",
        enable_progress_bar=False,
        callbacks=clbks,
    )

    trainer.fit(uq_method, dm)


def eval_uci(
    data_directory,
    experiment_name,
    results_path,
    measures,
    n_eval_ensembles,
    distortion,
    ckpt_path,
    train_size,
    **kwargs
):
    gamma = kwargs.get("gamma", 1.0)
    # Setup data
    dm = UCIRegressionDatamodule(
        dataset_name=experiment_name,
        root=data_directory,
        train_distortion=0,
        train_size=train_size,
    )
    sample = next(iter(dm.train_dataloader()))
    n_input = sample["input"].shape[-1]

    # Setup model
    mlp = MLP(
        n_inputs=n_input,
        n_outputs=2,
        activation_fn=nn.ReLU(),
        n_hidden=[256, 512, 1024],
    )
    model = MVERegression(
        mlp, optimizer=partial(torch.optim.Adam, lr=1e-2), burnin_epochs=20
    )

    # Create dataframe
    index = pd.MultiIndex.from_product([n_eval_ensembles, measures])
    results = pd.DataFrame(index=index, columns=distortion)

    # Iterate through different number of ensembles
    for n_ensembles in n_eval_ensembles:

        base_ensemble = [
            {"base_model": model, "ckpt_path": f"{ckpt_path}{experiment_name}_{np.round(x,1)}.ckpt"}
            for x in range(n_ensembles)
        ]
        robustness_members = [
            {
                "base_model": model,
                "ckpt_path": f"{ckpt_path}{experiment_name}_distorted_{np.round(x,1)}.ckpt",
            }
            for x in distortion
        ]

        # Base prediction
        deep_ens_nll = DeepEnsembleRegression(base_ensemble)
        input = next(iter(dm.test_dataloader()))
        base_pred = deep_ens_nll.predict_step(input["input"])["samples"].detach()
        log_sigma_2 = base_pred[:, 1]
        eps = torch.ones_like(log_sigma_2) * 1e-6
        base_pred[:, 1] = torch.sqrt(eps + np.exp(log_sigma_2))


        for i in range(len(distortion)):
            # Add distorted member
            base_ensemble.append(robustness_members.pop())
            # Distorted prediction
            deep_ens_nll = DeepEnsembleRegression(base_ensemble)
            input = next(iter(dm.test_dataloader()))
            pred = deep_ens_nll.predict_step(input["input"])["samples"].detach()
            log_sigma_2 = pred[:, 1]
            eps = torch.ones_like(log_sigma_2) * 1e-6
            pred[:, 1] = torch.sqrt(eps + np.exp(log_sigma_2))

            for measure in measures:
                base_uq_measure = GaussianUQMeasure(base_pred, gamma = gamma)
                base_au, _, _ = base_uq_measure.get_uncertainties(measure)

                uq_measure = GaussianUQMeasure(pred, gamma = gamma)
                au, _, _ = uq_measure.get_uncertainties(measure)

                rel_au = torch.abs((au - base_au) / base_au).mean()

                # Append to dataframe
                dist = distortion[len(distortion) - (i + 1)]
                results.loc[(n_ensembles, measure), dist] = rel_au.item()

            # Remove distorted member
            base_ensemble.pop()

    results.to_pickle(f"{results_path}{experiment_name}_au.pkl")


if __name__ == "__main__":
    # Configuration for UCI experiments
    epochs = 10000
    batch_size = 256
    train_size = 0.9
    lr = 0.0001
    data_path = "data/uci/"
    os.makedirs(data_path, exist_ok=True)
    experiments = ["energy", "concrete", "yacht"]
    results_path = "results/robustness/results/ensemble/"
    os.makedirs(results_path, exist_ok=True)

    # Ensemble configuration
    n_ensembles = 25
    n_eval_ensembles = [5,25]

    # Robustness configuration
    distortion = np.concatenate([np.arange(0.0,2.0,0.1),np.arange(2.0, 5.5, 0.5)])
    measures = ["log", "var", "crps", "kernel"]
    gamma_values = {"concrete":0.00095, "energy":0.0033, "yacht":0.0036}

    # Train
    for experiment in experiments:
        ckpt_path = f"results/robustness/checkpoints/{experiment}/"
        os.makedirs(ckpt_path, exist_ok=True)
        # General ensemble training
        for ensemble_id in range(n_ensembles):
            print(f"Training {experiment} with ensemble ID {ensemble_id}")
            train_uci(data_path, experiment, ensemble_id, ckpt_path, train_size, batch_size, lr, epochs)
        # Distorted training
        for dist in distortion:
            print(f"Training {experiment} with distortion {dist}")
            train_uci(data_path, experiment, ensemble_id, ckpt_path, train_size, batch_size, lr, epochs, train_distortion=dist)

        # Evaluate
        eval_uci(
            data_path,
            experiment,
            results_path,
            measures,
            n_eval_ensembles,
            distortion,
            ckpt_path,
            train_size,
            gamma = gamma_values[experiment], 
        )
