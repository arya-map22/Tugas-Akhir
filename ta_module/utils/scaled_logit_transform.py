import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform


class ScaledLogitTransform(Transform):
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, lb: float, ub: float, eps: float = 1e-6):
        super().__init__()
        assert ub > lb, "ub must be greater than lb"
        self.lb = lb
        self.ub = ub
        self.eps = eps
        self.domain = constraints.interval(lb, ub)

    def _inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Inverse link: eta in R -> x in (lb, ub)
        """
        return self.lb + (self.ub - self.lb) * torch.sigmoid(eta)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """
        Link function: x in (lb, ub) -> eta in R
        η = log((x - lb) / (ub - x))
        """
        x = torch.clamp(x, self.lb + self.eps, self.ub - self.eps)
        return torch.log(x - self.lb) - torch.log(self.ub - x)

    def log_abs_det_jacobian(self, eta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        log |dx/deta|
        """
        s = torch.sigmoid(eta)
        return torch.log(self.ub - self.lb) + torch.log(s) + torch.log1p(-s)
