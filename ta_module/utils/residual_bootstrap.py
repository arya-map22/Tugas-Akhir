from math import sqrt

import torch
from torch import Tensor
from torch.nn import Module

from ta_module.utils import normalize


@torch.no_grad()
def recursive_forecast_with_residual_bootstrap(
    model: Module,
    x: Tensor,  # (1, L, W)
    residuals: Tensor,  # (N, 1, W)
    forecast_horizon: int,
    n_sim: int,
    normalize_mean: Tensor,
    normalize_std: Tensor,
    device: str,
) -> Tensor:
    """
    Output:
        Tensor dengan shape (B, H, W)
        B = n_sim
        H = forecast_horizon
        W = jumlah fitur
    """

    assert x.dim() == 3
    assert residuals.dim() == 3

    # 🔹 repeat initial window
    x_in = x.repeat(n_sim, 1, 1).to(device)  # (B, L, W)

    # 🔹 wild bootstrap multipliers
    v = torch.tensor(
        [(1 + sqrt(5)) / 2, (1 - sqrt(5)) / 2],
        device=device,
    )

    prob_v = torch.tensor(
        [(sqrt(5) - 1) / (2 * sqrt(5)), (sqrt(5) + 1) / (2 * sqrt(5))],
        device=device,
    )

    predictions = []

    N = residuals.shape[0]

    for _ in range(forecast_horizon):
        # 🔹 model prediction
        y_t = model(x_in)  # (B, 1, W)

        # 🔹 sample residual index (GPU)
        idx = torch.randint(
            low=0,
            high=N,
            size=(n_sim,),
            device=device,
        )
        sampled_residuals = residuals[idx]  # (B, 1, W)

        # 🔹 sample wild multiplier (GPU)
        v_idx = torch.multinomial(prob_v, num_samples=n_sim, replacement=True)
        sampled_v = v[v_idx].view(-1, 1, 1)  # (B, 1, 1)

        # 🔹 apply bootstrap
        y_t_res = y_t + sampled_residuals * sampled_v  # (B, 1, W)
        y_t_res = normalize(y_t_res, normalize_mean, normalize_std)

        # 🔹 update rolling window
        x_in = torch.cat([x_in[:, 1:, :], y_t_res], dim=1)  # (B, L, W)

        predictions.append(y_t_res)

    # 🔹 (B, H, W)
    predictions = torch.cat(predictions, dim=1)

    return predictions.to(device)
