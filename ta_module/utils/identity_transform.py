from torch.distributions.constraints import real
from torch.distributions.transforms import Transform
from torch import Tensor


class IdentityTransform(Transform):
    domain = real
    codomain = real
    bijective = True

    def _call(self, x: Tensor) -> Tensor:
        return x

    def _inverse(self, y: Tensor) -> Tensor:
        return y
