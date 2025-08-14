from typing import Optional
import torch
import torch.nn as nn
import numpy as np

EPS = 1e-9

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
    ) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
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