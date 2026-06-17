import torch

from torch import Tensor
from torch.distributions import Uniform

uniform_dist = Uniform(0, 1)


def inverse_transform_sampling(x: Tensor, cdf: Tensor) -> Tensor:
    assert x.shape == cdf.shape

    device = cdf.device
    sample_shape = list(x.shape)
    sample_shape[-1] = 1
    U = uniform_dist.sample(sample_shape=sample_shape).to(torch.float32).to(device)
    indices = torch.searchsorted(sorted_sequence=cdf, input=U)

    return torch.gather(input=x, dim=-1, index=indices)