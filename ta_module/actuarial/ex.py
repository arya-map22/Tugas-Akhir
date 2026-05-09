import torch


def compute_ex(kpx: torch.Tensor) -> torch.Tensor:
    """
    kpx: Tensor shape (K)
    return: Tensor shpae (1)
    """
    assert kpx.dim() == 1

    return 0.5 + kpx[1:].sum()
