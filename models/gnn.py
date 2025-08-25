import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
import pytorch_lightning as L
from models.losses import NormalCRPS
from typing import Tuple
from torch_geometric.nn import GATv2Conv
from torch.nn import Linear, ModuleList, ReLU

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
        self.embed = nn.Embedding(num_embeddings=num_stations_max, embedding_dim=embedding_dim)

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


class MakePositive(nn.Module):
    """
    A module that ensures the output tensor has positive values for sigma.

    Args:
        None

    """

    def __init__(self):
        super(MakePositive, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MakePositive module.

        Args:
            x (torch.Tensor): The input tensor containting mu and sigma of shape (S,2).

        Returns:
            torch.Tensor: A tensor with positive values for sigma.

        """
        mu, sigma = torch.split(x, 1, dim=-1)
        sigma = F.softplus(sigma) + 1e-7  # ensure that sigma is positive
        mu_sigma = torch.cat([mu, sigma], dim=-1)
        return mu_sigma


class DeepSetAggregator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DeepSetAggregator, self).__init__()

        self.input = torch.nn.Linear(in_channels, hidden_channels)
        self.hidden1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.hidden2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.output = torch.nn.Linear(hidden_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x, index):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = scatter(x, index, dim=0, reduce="mean")
        self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class ResGnn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int, hidden_channels: int, heads: int):
        super(ResGnn, self).__init__()
        assert num_layers > 0, "num_layers must be > 0."

        # Create Layers
        self.convolutions = ModuleList()
        for _ in range(num_layers):
            self.convolutions.append(
                GATv2Conv(-1, hidden_channels, heads=heads, edge_dim=1, add_self_loops=True, fill_value=0.01)
            )
        self.lin = Linear(hidden_channels * heads, out_channels)
        self.relu = ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = x.float()
        edge_attr = edge_attr.float()
        for i, conv in enumerate(self.convolutions):
            if i == 0:
                # First Layer
                x = conv(x, edge_index, edge_attr)
                x = self.relu(x)
            else:
                x = x + self.relu(conv(x, edge_index, edge_attr))  # Residual Layers

        x = self.lin(x)
        return x

    @torch.no_grad()
    def get_attention(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs a forward Pass for the given graph only though the ResGNN layer.
        NOTE: the data that is given to this method must first pass through the layers before this layer in the Graph

        :param torch.Tensor x: Tensor of Node Features (NxD)
        :param torch.Tensor edge_index: Tensor of Edges (2xE)
        :param torch.Tensor edge_attr: Edge Attributes (ExNum_Attr)
        :return x, edge_index_attention, attention_weights: Tensor of Node Features (NxD), Tensor of Edges with
        self loops (2xE), Tensor of Attention per edge (ExNum_Heads)
        """
        x = x.float()
        edge_attr = edge_attr.float()

        # Pass Data though Layer to get the Attention
        attention_list = []
        # Note: edge_index_attention has to be added since we have self loops now
        edge_index_attention, attention_weights = None, None

        for i, conv in enumerate(
            self.convolutions,
        ):
            if i == 0:
                # First Layer
                x, (edge_index_attention, attention_weights) = conv(
                    x, edge_index, edge_attr, return_attention_weights=True
                )
                attention_list.append(attention_weights)
                x = self.relu(x)
                x = self.norm(x)
            else:
                x_conv, (edge_index_attention, attention_weights) = conv(
                    x, edge_index, edge_attr, return_attention_weights=True
                )
                attention_list.append(attention_weights)
                x = x + self.relu(x_conv)  # Residual Layers
        x = self.lin(x)

        # Attention weights of first layer
        attention_weights = attention_weights.mean(dim=1)

        return x, edge_index_attention, attention_weights, attention_list


class Multigraph(L.LightningModule):
    def __init__(
        self,
        embedding_dim,
        in_channels,
        hidden_channels_gnn,
        out_channels_gnn,
        num_layers_gnn,
        heads,
        hidden_channels_deepset,
        optimizer_class,
        optimizer_params,
    ):
        super(Multigraph, self).__init__()

        self.encoder = EmbedStations(num_stations_max=122, embedding_dim=embedding_dim)

        self.conv = ResGnn(
            in_channels=in_channels,
            hidden_channels=hidden_channels_gnn,
            out_channels=out_channels_gnn,
            num_layers=num_layers_gnn,
            heads=heads,
        )

        self.aggr = DeepSetAggregator(
            in_channels=out_channels_gnn, hidden_channels=hidden_channels_deepset, out_channels=2
        )

        self.postprocess = MakePositive()
        self.loss_fn = NormalCRPS(mask = True)

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

    def forward(self, data):
        x, edge_index, edge_attr, batch_id, node_idx = data.x, data.edge_index, data.edge_attr, data.batch, data.n_idx
        node_idx = node_idx + batch_id * 122  # add batch_id to node_idx to get unique node indices
        x = self.encoder(x)
        x = self.conv(x, edge_index, edge_attr)
        x = self.aggr(x, node_idx)
        x = self.postprocess(x)
        return x

    def training_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.loss_fn(prediction=y_hat, observation=batch.y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=1
        )  # The batch size is not actually 1 but the loss is already averaged over the batch
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.optimizer_params)

    def validation_step(self, batch, batch_idx):
        y_hat = self.forward(batch)
        loss = self.loss_fn(prediction=y_hat, observation=batch.y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        y_hat = self.forward(batch)
        loss = self.loss_fn(prediction=y_hat, observation=batch.y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def initialize(self, dataloader):
        batch = next(iter(dataloader))
        self.validation_step(batch, 0)