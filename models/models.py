import torch.nn as nn
import torch
from torch import Tensor
import lightning as L
from models.losses import NormalCRPS

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
        if activation_fn is None:
            activation_fn = nn.ReLU()
        n_inputs = n_inputs + 2*emb_dim        
        
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
        self.lat_embed = nn.Embedding(num_embeddings=160, embedding_dim=emb_dim)  # 8-dim lat encoding
        self.lon_embed = nn.Embedding(num_embeddings=220, embedding_dim=emb_dim)  # 8-dim lon encoding

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
        lon_vec = self.lon_embed(lon_indices) # shape (..., emb_dim)
        coords_vec = torch.cat([lat_vec, lon_vec], dim=-1).squeeze(1)
        input = torch.cat([x, coords_vec], dim=-1)  # shape (..., n_inputs + 2*emb_dim)
        output = self.model(input)
        mu, sigma = torch.split(output, 1, dim=-1)
        sigma = nn.functional.softplus(sigma) + EPS
        output = torch.cat([mu, sigma], dim=-1)
        return output
    
class LightningDRN(L.LightningModule):
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


if __name__ == "__main__":
    inp = torch.randn(30, 6)  # Example input tensor
    emb = torch.cat([torch.randint(0, 160, (30, 1)), torch.randint(0, 220, (30, 1))], dim=-1)  # Example embedding tensor with lat and lon indices
    model = EmbeddingMLP(n_inputs = 6, emb_dim=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = inp.to(device)
    emb = emb.to(device)
    model = model.to(device)
    output = model(inp, emb)
    print(output.shape)
    print(model)

