# Implements the different uncertainty measures for a Gaussian ensemble.

import numpy as np
import torch
from scipy.special import hyp1f1


class GaussianUQMeasure:
    """Uncertainty measure for first-order Gaussian"""

    def __init__(
        self,
        prediction: torch.Tensor,
        variant="pairwise",
        second_order="ensemble",
        **kwargs,
    ):
        """Initialize class

        Args:
            prediction (torch.Tensor): Prediction tensor.
            variant (str, optional): Pairwise or BMA variant. Defaults to "pairwise".
            second_order (str, optional): _description_. Defaults to "ensemble".

        Raises:
            ValueError: second_order method must be "ensemble"
            ValueError: Parediction must have shape [B, d1, ... dN, 2, M].
        """
        self.variant = variant
        self.second_order = second_order
        self.gamma = kwargs.get("gamma", 0.1)

        # Check sizes
        if self.second_order == "ensemble":
            # Prediction should have shape [B, d1, ... dN, 2, M]
            if len(prediction.shape) < 3:
                prediction = prediction.unsqueeze(1)
            if prediction.shape[-2] != 2:
                raise ValueError(
                    "Prediction must have shape [B, d1, ... dN, 2, M] for ensemble second order method"
                )
            self.dimensions = prediction.shape[:-2]
            mu, sigma = torch.split(prediction.detach(), 1, -2)
            self.mu = mu.flatten(start_dim=1, end_dim=-2)
            self.sigma = sigma.flatten(start_dim=1, end_dim=-2)
            self.m = prediction.shape[-1]
            self.corr = self.m * (self.m - 1)
        else:
            raise ValueError("Invalid second order method")

    def get_uncertainties(self, measure: str = "crps") -> tuple:
        """Returns TU,AU,EU for a specific uncertainty measure.

        Args:
            measure (str, optional): Defaults to "crps".

        Raises:
            ValueError: Measure must be one of ["crps", "kernel", "log", "var"]

        Returns:
            tuple: (AU,EU,TU)
        """
        if measure == "crps":
            au, eu = self._get_crps_uncertainty()
        elif measure == "kernel":
            au, eu = self._get_kernel_uncertainty()
        elif measure == "log":
            au, eu = self._get_log_uncertainty()
        elif measure == "var":
            au, eu = self._get_var_uncertainty()
        else:
            raise ValueError("Invalid measure")

        # Reshape to original dimensions
        au = au.reshape(*self.dimensions)
        eu = eu.reshape(*self.dimensions)

        tu = au + eu
        return au, eu, tu

    def _get_crps_uncertainty(self):
        # Aleatoric uncertainty
        au = (self.sigma / np.sqrt(np.pi)).mean(dim=-1)
        # Epistemic uncertainty needs to be on CPU for hyp1f1 in scipy
        device = self.mu.device
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        sigma = self.sigma.unsqueeze(-1)
        tau = self.sigma.unsqueeze(-2)
        mu_diff = mu_diff.cpu()
        sigma = sigma.cpu()
        tau = tau.cpu()
        f1 = hyp1f1(
            -0.5,
            0.5,
            -0.5 * torch.pow(mu_diff, 2) / (torch.pow(sigma, 2) + torch.pow(tau, 2)),
        )
        eu = torch.sqrt(torch.pow(sigma, 2) + torch.pow(tau, 2)) * np.sqrt(
            2 / np.pi
        ) * f1 - (sigma + tau) / np.sqrt(np.pi)
        diag_mask = 1 - torch.eye(self.m, dtype=eu.dtype, device=eu.device)
        eu = eu * diag_mask
        eu = (eu.sum(dim=(-1, -2)) / self.corr).to(device)
        return au, eu

    def _get_kernel_uncertainty(self):
        # Aleatoric uncertainty
        gamma = self.gamma
        au = 0.5 * (
            1 - gamma / (torch.sqrt(gamma**2 + 4 * torch.pow(self.sigma, 2)))
        ).mean(dim=-1)
        # Epistemic uncertainty
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        sigma = self.sigma.unsqueeze(-1)
        tau = self.sigma.unsqueeze(-2)
        eu = (
            0.5 * gamma / torch.sqrt(gamma**2 + 4 * torch.pow(sigma, 2))
            + 0.5 * gamma / torch.sqrt(gamma**2 + 4 * torch.pow(tau, 2))
            - gamma
            / torch.sqrt(gamma**2 + 2 * (torch.pow(sigma, 2) + torch.pow(tau, 2)))
            * torch.exp(
                -torch.pow(mu_diff, 2)
                / (gamma**2 + 2 * (torch.pow(sigma, 2) + torch.pow(tau, 2)))
            )
        )
        eu = eu.sum(dim=(-1, -2)) / self.corr
        return au, eu

    def _get_log_uncertainty(self):
        # Aleatoric uncertainty
        au = 0.5 * torch.log(2 * np.pi * np.e * torch.pow(self.sigma, 2)).mean(dim=-1)
        # Epistemic uncertainty
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        sigma = self.sigma.unsqueeze(-1)
        tau = self.sigma.unsqueeze(-2)
        eu = (
            torch.log(sigma / tau)
            + (torch.pow(sigma, 2) + torch.pow(mu_diff, 2)) / (2 * torch.pow(tau, 2))
            - 0.5
        )
        eu = eu.sum(dim=(-1, -2)) / self.corr
        return au, eu

    def _get_var_uncertainty(self):
        # Aleatoric uncertainty
        au = torch.pow(self.sigma, 2).mean(dim=-1)
        # Epistemic uncertainty
        mu_diff = self.mu.unsqueeze(-1) - self.mu.unsqueeze(-2)
        eu = torch.pow(mu_diff, 2).sum(dim=(-1, -2)) / self.corr
        return au, eu


if __name__ == "__main__":
    pred = torch.rand(10, 5, 6, 2, 3) * 100  # Example mu tensor
    pred2 = torch.rand(10, 2, 3)  # Example mu2 tensor
    uq = GaussianUQMeasure(pred)
    uq2 = GaussianUQMeasure(pred2)
    measures = ["crps", "kernel", "log", "var"]
    for measure in measures:
        au, eu, tu = uq.get_uncertainties(measure=measure)
        au2, eu2, tu2 = uq2.get_uncertainties(measure=measure)
        print(au.mean(), eu.mean(), tu.mean())
        # print(au2.mean(), eu2.mean(), tu2.mean())
