from torch import Tensor
from torch.distributions.constraints import real
from torch.distributions.transforms import Transform


class IdentityTransform(Transform):
    domain = real
    codomain = real
    bijective = True

    def _call(self, x: Tensor) -> Tensor:
        return x

    def _inverse(self, y: Tensor) -> Tensor:
        return y
