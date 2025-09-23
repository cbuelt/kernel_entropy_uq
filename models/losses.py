# Implementation of several loss functions

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

EPS = 1e-9


class GaussianKernelScore(nn.Module):
    """Implementation of the Gaussian kernel score for Gaussian RVs."""

    def __init__(
        self,
        gamma: float,
        reduction: Optional[str] = "mean",
        ensemble: bool = False,
    ) -> None:
        """Initialize class.

        Args:
            gamma (float): Kernel bandwidth
            reduction (Optional[str], optional):  Defaults to "mean".
            ensemble (bool, optional): Whether prediction has an additional dimension. Defaults to False.
        """
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
        gamma = torch.tensor(self.gamma, device=mu.device)
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
    """Implementation of the negative log likelihood for Gaussian RVs."""

    def __init__(
        self,
        reduction: Optional[str] = "mean",
        ensemble: bool = False,
    ) -> None:
        """Initialize class.

        Args:
            reduction (Optional[str], optional):  Defaults to "mean".
            ensemble (bool, optional): Whether prediction has an additional dimension. Defaults to False.
        """
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

        norm = Normal(loc=mu, scale=sigma)
        score = (-1) * norm.log_prob(observation)
        if self.reduction == "sum":
            return torch.sum(score)
        elif self.reduction == "mean":
            return torch.mean(score)
        else:
            return score


class SquaredError(nn.Module):
    """Implementation of the squared error for RVs."""

    def __init__(
        self,
        reduction: Optional[str] = "mean",
        ensemble: bool = False,
    ) -> None:
        """Initialize class.

        Args:
            reduction (Optional[str], optional):  Defaults to "mean".
            ensemble (bool, optional): Whether prediction has an additional dimension. Defaults to False.
        """
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
    """Implementation of the CRPS for Gaussian RVs."""

    def __init__(
        self,
        reduction: Optional[str] = "mean",
        ensemble: bool = False,
    ) -> None:
        """Initialize class.

        Args:
            reduction (Optional[str], optional):  Defaults to "mean".
            ensemble (bool, optional): Whether prediction has an additional dimension. Defaults to False.
        """
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


class DERLoss(nn.Module):
    """Deep Evidential Regression Loss.

    Taken from `here <https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py#L61>`_. # noqa: E501

    This implements the loss corresponding to equation 12
    from the `paper <https://arxiv.org/abs/2205.10060>`_.

    If you use this model in your work, please cite:

    * https://arxiv.org/abs/2205.10060
    """

    def __init__(self, coeff: float = 0.01) -> None:
        """Initialize a new instance of the loss function.

        Args:
          coeff: loss function coefficient
        """
        super().__init__()
        self.coeff = coeff

    def NIG_NLL(self, y, gamma, nu, alpha, beta, reduce=True):
        twoBlambda = 2 * beta * (1 + nu)

        nll = (
            0.5 * torch.log(np.pi / nu)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        return torch.mean(nll) if reduce else nll

    def NIG_Reg(self, y, gamma, nu, alpha, beta, reduce=True):
        error = torch.abs(y - gamma)
        evi = 2 * nu + (alpha)
        reg = error * evi

        return torch.mean(reg) if reduce else reg

    def forward(self, pred, y_true):
        """DER Loss.

        Args:
          logits: predicted tensor from model [batch_size x 4 x other dims]
          y_true: true regression target of shape [batch_size x 1 x other dims]

        Returns:
          DER loss
        """
        assert pred.shape[-1] == 4, (
            "logits should have shape [batch_size x 4 x other dims]"
        )
        gamma, nu, alpha, beta = torch.split(pred, 1, dim=-1)
        loss_nll = self.NIG_NLL(y_true, gamma, nu, alpha, beta)
        loss_reg = self.NIG_Reg(y_true, gamma, nu, alpha, beta)
        return loss_nll + self.coeff * loss_reg
