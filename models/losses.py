from typing import Optional
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

EPS = 1e-9

class GaussianKernelScore(nn.Module):
    def __init__(
        self,
        gamma:float,
        reduction: Optional[str] = "mean",
        ensemble:bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.ensemble = ensemble
        self.gamma = gamma

    def forward(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:

        mu, sigma = torch.split(prediction, 1, dim=-1)
        if self.ensemble:
            observation = observation.unsqueeze(-1)

        # Use power of sigma
        sigma2 = torch.pow(sigma, 2)
        # Flatten values
        mu = torch.flatten(mu, start_dim=1)
        sigma2 = torch.flatten(sigma2, start_dim=1)
        observation = torch.flatten(observation, start_dim=1)
        gamma = (
            torch.tensor(self.gamma, device=mu.device)
        )
        gamma2 = torch.pow(gamma, 2)
        # Calculate the Gaussian kernel score
        fac1 = (
            1
            / (torch.sqrt(1 + 2 * sigma2 / gamma2))
            * torch.exp(-torch.pow(observation - mu, 2) / (gamma2 + 2 * sigma2))
        )
        fac2 = 1 / (2 * torch.sqrt(1 + 4 * sigma2 / gamma2))
        score = 0.5 - fac1 + fac2

        if self.reduction == "sum":
            return torch.sum(score)
        elif self.reduction == "mean":
            return torch.mean(score)
        else:
            return score

class NLL(nn.Module):
    def __init__(
        self,
        reduction: Optional[str] = "mean",
        ensemble:bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.ensemble = ensemble

    def forward(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:

        mu, sigma = torch.split(prediction, 1, dim=-1)
        if self.ensemble:
            observation = observation.unsqueeze(-1)

        norm = Normal(loc = mu, scale = sigma)
        score = (-1)*norm.log_prob(observation)
        if self.reduction == "sum":
            return torch.sum(score)
        elif self.reduction == "mean":
            return torch.mean(score)
        else:
            return score

class SquaredError(nn.Module):
    def __init__(
        self,
        reduction: Optional[str] = "mean",
        ensemble:bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.ensemble = ensemble

    def forward(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:

        mu, sigma = torch.split(prediction, 1, dim=-1)
        if self.ensemble:
            observation = observation.unsqueeze(-1)

        score = torch.pow(observation - mu, 2)
        if self.reduction == "sum":
            return torch.sum(score)
        elif self.reduction == "mean":
            return torch.mean(score)
        else:
            return score

class NormalCRPS(nn.Module):
    """Computes the continuous ranked probability score (CRPS)
       for a predictive normal distribution and corresponding observations.

    Args:
        observation (torch.Tensor): Observed outcome. Shape = [batch_size, d0, .. dn].
        mu (torch.Tensor): Predicted mu of normal distribution. Shape = [batch_size, d0, .. dn].
        sigma2 (torch.Tensor): Predicted sigma2 of normal distribution. Shape = [batch_size, d0, .. dn].
        reduce (bool, optional): Boolean value indicating whether reducing the loss to one value or to
            a torch.Tensor with shape = `[batch_size]`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``.
    Raises:
        ValueError: If sizes of target mu and sigma don't match.

    Returns:
        CRPS: 1-D float `torch.Tensor` with shape [batch_size] if reduction = True
    """

    def __init__(
        self,
        reduction: Optional[str] = "mean",
        ensemble:bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.ensemble = ensemble

    def forward(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:

        if self.ensemble:
            mu, sigma = torch.split(prediction, 1, dim=1)
            observation = observation.unsqueeze(-1)
        else:
            mu, sigma = torch.split(prediction, 1, dim=-1)
        loc = (observation - mu) / sigma
        cdf = 0.5 * (1 + torch.erf(loc / np.sqrt(2.0)))
        pdf = 1 / (np.sqrt(2.0 * np.pi)) * torch.exp(-torch.pow(loc, 2) / 2.0)
        crps = sigma * (loc * (2.0 * cdf - 1) + 2.0 * pdf - 1 / np.sqrt(np.pi))
        if self.reduction == "sum":
            return torch.sum(crps)
        elif self.reduction == "mean":
            return torch.mean(crps)
        else:
            return crps