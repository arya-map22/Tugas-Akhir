from torch import Tensor


def normalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (data - mean) / std


def denormalize(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return data * std + mean
