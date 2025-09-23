# This file contains code adapted from:
# "gnn-post-processing" by hits-mli
# Repository: https://github.com/hits-mli/gnn-post-processing
# Paper: Feik et al. Graph Neural Networks and Spatial Information Learning for Post-Processing Ensemble Weather Forecasts (2024)


from typing import List, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from models.losses import NLL, DERLoss, GaussianKernelScore, NormalCRPS, SquaredError

EPS = 1e-9


class EmbeddingMLP(nn.Module):
    """Multi-layer perceptron for predictions."""

    def __init__(
        self,
        dropout_p: float = 0.1,
        n_inputs: int = 5,
        emb_dim: int = 5,
        n_hidden: list[int] = [512],
        n_outputs: int = 2,
        activation_fn: nn.Module | None = None,
    ) -> None:
        """Initialize a new instance of MLP.

        Args:
          dropout_p: dropout percentage
          n_inputs: size of input dimension
          n_hidden: list of hidden layer sizes
          n_outputs: number of model outputs
          predict_sigma: whether the model intends to predict sigma term
            when minimizing NLL
          activation_fn: what nonlinearity to include in the network
        """
        super().__init__()
        self.n_outputs = n_outputs
        if activation_fn is None:
            activation_fn = nn.ReLU()
        n_inputs = n_inputs + 2 * emb_dim

        layer_sizes = [n_inputs] + n_hidden
        layers = []
        for idx in range(1, len(layer_sizes)):
            layers += [
                nn.Linear(layer_sizes[idx - 1], layer_sizes[idx]),
                activation_fn,
                nn.Dropout(dropout_p),  # if idx != 1 else nn.Identity(),
            ]
        # add output layer
        layers += [nn.Linear(layer_sizes[-1], n_outputs)]
        self.model = nn.Sequential(*layers)

        # Add Embeddings
        self.lat_embed = nn.Embedding(
            num_embeddings=160, embedding_dim=emb_dim
        )  # 8-dim lat encoding
        self.lon_embed = nn.Embedding(
            num_embeddings=220, embedding_dim=emb_dim
        )  # 8-dim lon encoding

    def forward(self, x, emb) -> Tensor:
        """Forward pass through the neural network.

        Args:
          x: input vector to NN of dimension [batch_size, n_inputs]
          emb: input embeddings of dimension [batch_size, emb_size]

        Returs:
          output from neural net of dimension [batch_size, n_outputs]
        """
        lat_indices, lon_indices = torch.split(emb, 1, dim=-1)
        lat_vec = self.lat_embed(lat_indices)  # shape (..., emb_dim)
        lon_vec = self.lon_embed(lon_indices)  # shape (..., emb_dim)
        coords_vec = torch.cat([lat_vec, lon_vec], dim=-1).squeeze(1)
        input = torch.cat([x, coords_vec], dim=-1)  # shape (..., n_inputs + 2*emb_dim)
        output = self.model(input)
        if self.n_outputs == 2:
            # if predicting mu and sigma, ensure sigma is positive
            mu, sigma = torch.split(output, 1, dim=-1)
            sigma = nn.functional.softplus(sigma) + EPS
            output = torch.cat([mu, sigma], dim=-1)
        elif self.n_outputs == 4:  # Deep evidential regression
            gamma, nu, alpha, beta = torch.split(output, 1, dim=-1)
            nu = nn.functional.softplus(nu)
            alpha = nn.functional.softplus(alpha) + 1.0
            beta = nn.functional.softplus(beta) + EPS
            output = torch.cat([gamma, nu, alpha, beta], dim=-1)
        return output


class LightningDRN(L.LightningModule):
    """Lightning wrapper for the DRN."""

    def __init__(self, model: EmbeddingMLP):
        super().__init__()
        self.model = model
        self.loss = NormalCRPS(reduction="mean")

    def forward(self, x, emb):
        return self.model(x, emb)

    def training_step(self, batch, batch_idx):
        x, emb, y = batch
        pred = self(x, emb)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, emb, y = batch
        pred = self(x, emb)
        loss = self.loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch):
        x, emb, y = batch
        x = x.squeeze(0)
        emb = emb.squeeze(0)
        y = y.squeeze(0)
        pred = self(x, emb)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class LightningDER(L.LightningModule):
    """Lightning wrapper for the DRN with deep evidential regression."""

    def __init__(self, model: EmbeddingMLP, coeff=0.01):
        super().__init__()
        self.model = model
        self.loss = DERLoss(coeff=coeff)

    def forward(self, x, emb):
        return self.model(x, emb)

    def training_step(self, batch, batch_idx):
        x, emb, y = batch
        pred = self(x, emb)
        loss = self.loss(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, emb, y = batch
        pred = self(x, emb)
        loss = self.loss(pred, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch):
        x, emb, y = batch
        x = x.squeeze(0)
        emb = emb.squeeze(0)
        y = y.squeeze(0)
        pred = self(x, emb)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def normalize_features(
    training_data: pd.DataFrame,
    valid_test_data: List[Tuple[pd.DataFrame]],
) -> Tuple[pd.DataFrame, List[Tuple[pd.DataFrame]]]:
    """
    Normalize the features in the training data and validation/test data. Also add the cos_doy and sin_doy features.

    Args:
        training_data (pd.DataFrame): The training data. Each Tuple contains the features and the target.
        valid_test_data (List[Tuple[pd.DataFrame]]): The validation/test data.
        Each Tuple contains the features and the target.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[pd.DataFrame]]]: The normalized training data and validation/test data.
    """

    # Normalize Features ############################################################
    # Select the features to normalize
    print("[INFO] Normalizing features...")
    train_rf = training_data[0]
    features_to_normalize = [
        col for col in train_rf.columns if col not in ["station_id", "time", "number"]
    ]

    # Create a MinMaxScaler object
    scaler = StandardScaler()

    # Fit and transform the selected features
    train_rf.loc[:, features_to_normalize] = scaler.fit_transform(
        train_rf[features_to_normalize]
    ).astype("float32")

    train_rf.loc[:, ["cos_doy"]] = np.cos(
        2 * np.pi * train_rf["time"].dt.dayofyear / 365
    )
    train_rf.loc[:, ["sin_doy"]] = np.sin(
        2 * np.pi * train_rf["time"].dt.dayofyear / 365
    )
    train_rf.drop(columns=["time", "number"], inplace=True)

    for features, _ in valid_test_data:
        features.loc[:, features_to_normalize] = scaler.transform(
            features[features_to_normalize]
        ).astype("float32")
        features.loc[:, ["cos_doy"]] = np.cos(
            2 * np.pi * features["time"].dt.dayofyear / 365
        )
        features.loc[:, ["sin_doy"]] = np.sin(
            2 * np.pi * features["time"].dt.dayofyear / 365
        )
        features.drop(columns=["time", "number"], inplace=True)

    return training_data, valid_test_data


def drop_nans(
    dfs: Tuple[pd.DataFrame, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows with NaN values in the 't2m' column in the target Dataframe.

    Args:
        dfs (Tuple[pd.DataFrame, pd.DataFrame]): A tuple containing two DataFrames (Features and Target).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames with rows containing
            NaN values in the 't2m' column dropped.
    """
    nans = dfs[1]["t2m"].isna().reset_index(drop=True)
    res = (dfs[0][~nans], dfs[1][~nans])
    return res


class EmbedStations(nn.Module):
    """A module to embed station IDs into a feature vector.

    Args:
        num_stations_max (int): The maximum number of stations.
        embedding_dim (int): The dimension of the embedding vector.

    Attributes:
        embed (nn.Embedding): The embedding layer.

    """

    def __init__(self, num_stations_max, embedding_dim):
        super(EmbedStations, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_stations_max, embedding_dim=embedding_dim
        )

    def forward(self, x: torch.Tensor):
        """Forward pass of the EmbedStations module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after embedding the station IDs.

        """
        station_ids = x[..., 0].long()
        emb_station = self.embed(station_ids)
        x = torch.cat(
            (emb_station, x[..., 1:]), dim=-1
        )  # Concatenate embedded station_id to rest of the feature vector
        return x


class StationDRN(L.LightningModule):
    """DRN from Rasp and Lerch - 2018 - Neural Networks for Postprocessing Ensemble Weathe.pdf"""

    def __init__(
        self,
        embedding_dim,
        in_channels,
        hidden_channels,
        loss,
        optimizer_class,
        optimizer_params,
        **kwargs,
    ):
        super(StationDRN, self).__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.gamma = kwargs.get("gamma", 1.0)

        self.embedding = EmbedStations(
            num_stations_max=122, embedding_dim=embedding_dim
        )

        self.linear = nn.ModuleList()
        for hidden_size in self.hidden_channels:
            self.linear.append(
                nn.Linear(in_features=in_channels, out_features=hidden_size)
            )
            in_channels = hidden_size
        self.last_linear_mu = nn.Linear(in_features=in_channels, out_features=1)
        self.last_linear_sigma = nn.Linear(in_features=in_channels, out_features=1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        if loss == "crps":
            self.loss_fn = NormalCRPS()
        elif loss == "se":
            self.loss_fn = SquaredError()
        elif loss == "kernel":
            self.loss_fn = GaussianKernelScore(gamma=self.gamma)
        elif loss == "log":
            self.loss_fn = NLL()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.linear:
            x = layer(x)
            x = self.relu(x)
        mu = self.last_linear_mu(x)  # Last Layer without ReLU
        sigma = self.softplus(self.last_linear_sigma(x)) + EPS
        res = torch.cat([mu, sigma], dim=1)
        return res

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(prediction=y_hat, observation=y)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.optimizer_params)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(prediction=y_hat, observation=y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(prediction=y_hat, observation=y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat


if __name__ == "__main__":
    inp = torch.randn(30, 6)  # Example input tensor
    emb = torch.cat(
        [torch.randint(0, 160, (30, 1)), torch.randint(0, 220, (30, 1))], dim=-1
    )  # Example embedding tensor with lat and lon indices
    model = EmbeddingMLP(n_inputs=6, emb_dim=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = inp.to(device)
    emb = emb.to(device)
    model = model.to(device)
    output = model(inp, emb)
    print(output.shape)
    print(model)
