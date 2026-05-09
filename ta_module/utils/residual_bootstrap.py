from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module


@torch.no_grad()
def recursive_forecast_with_residual_bootstrap(
    model: Module,
    x: Tensor,  # (1, L, W)
    residuals: Tensor,  # (N, 1, W)
    forecast_horizon: int,
    n_sim: int,
) -> Tensor:
    """
    Recursive multi-step forecast dengan wild bootstrap (Mammen).
    Output: Tensor shape (n_sim, H, W).
    """
    device = x.device
    model = model.to(device)

    assert x.dim() == 3 and x.shape[0] == 1
    assert residuals.dim() == 3

    N, _, W = residuals.shape

    # Mammen constants
    _s5 = sqrt(5)
    a = -(_s5 - 1) / 2
    b = (_s5 + 1) / 2
    p_b = (_s5 - 1) / (2 * _s5)

    residuals = residuals.to(device)
    ab = torch.tensor([a, b], device=device)

    x_in = x.repeat(n_sim, 1, 1)
    # Kumpulkan sebagai list, cat sekali di akhir
    # untuk hindari OOM dari pre-alokasi (n_sim, H, W)
    predictions = []

    for i in range(forecast_horizon):
        print(f"Forecasting step {i + 1}/{forecast_horizon}...")
        # Random per-step: hanya (n_sim,) dan (n_sim, 1, 1) — kecil
        idx = torch.randint(0, N, (n_sim,), device=device)
        mask = torch.bernoulli(torch.full((n_sim, 1, 1), p_b, device=device)).bool()

        sampled = residuals[idx]  # (n_sim, 1, W)
        wild_w = ab[mask.long()]  # (n_sim, 1, 1)

        y_t = model(x_in) + sampled * wild_w  # (n_sim, 1, W)
        predictions.append(y_t)

        x_in = torch.cat([x_in[:, 1:, :], y_t], dim=1)

    return torch.cat(predictions, dim=1)  # (n_sim, H, W)
