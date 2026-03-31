import torch
import torch.nn.functional as F

from torch import Tensor


def mae_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()


def rmse_loss(pred: Tensor, target: Tensor) -> Tensor:
    mse = F.mse_loss(pred, target)
    return torch.sqrt(mse)
