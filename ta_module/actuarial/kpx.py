import torch


def compute_kpx_table_from_xstart_dynamic(
    p: torch.Tensor, x_start: int, max_k: int, t0: int = 0
):
    T, X = p.shape
    device = p.device

    # k maksimum dari teori aktuaria
    K = max_k
    X_sub = X - x_start

    # index
    x_idx = torch.arange(x_start, X, device=device).view(X_sub, 1)
    k_idx = torch.arange(K, device=device).view(1, K)

    # diagonal
    t_idx = t0 + k_idx  # shape (1, K)
    x_idx = x_idx + k_idx  # shape (X_sub, K)

    # valid mask
    # shape (X_sub, K)
    valid = (t_idx < T) & (x_idx < X)

    # clamp biar aman CUDA
    t_idx = t_idx.clamp(0, T - 1)
    x_idx = x_idx.clamp(0, X - 1)

    # 🔥 ambil nilai diagonal
    # shape (X_sub, K)
    gathered = p[t_idx, x_idx]

    # invalid → 1
    gathered = torch.where(valid, gathered, torch.ones_like(gathered))

    # cumulative product
    kpx = torch.cumprod(gathered, dim=1)

    kpx = torch.where(valid, kpx, torch.zeros_like(kpx))

    # Tambahkan 0px = 1
    kpx = torch.cat([torch.ones(X_sub, 1, device=device), kpx], dim=1)

    # kpx shape (X_sub, K + 1)
    return kpx.to(torch.float16)


def compute_kpx_table_from_xstart_static(p: torch.Tensor, x_start: int, max_k: int):
    _, X = p.shape
    device = p.device

    # k maksimum dari teori aktuaria
    K = max_k
    X_sub = X - x_start

    p = p.expand(K, X)

    # index
    x_idx = torch.arange(x_start, X, device=device).view(X_sub, 1)
    k_idx = torch.arange(K, device=device).view(1, K)

    # diagonal
    t_idx = 0 + k_idx  # shape (1, K)
    x_idx = x_idx + k_idx  # shape (X_sub, K)

    # valid mask
    # shape (X_sub, K)
    valid = x_idx < X

    # clamp biar aman CUDA
    x_idx = x_idx.clamp(0, X - 1)

    # 🔥 ambil nilai diagonal
    # shape (X_sub, K)
    gathered = p[t_idx, x_idx]

    # invalid → 1
    gathered = torch.where(valid, gathered, torch.ones_like(gathered))

    # cumulative product
    kpx = torch.cumprod(gathered, dim=1)

    kpx = torch.where(valid, kpx, torch.zeros_like(kpx))

    # Tambahkan 0px = 1
    kpx = torch.cat([torch.ones(X_sub, 1, device=device), kpx], dim=1)

    # kpx shape (X_sub, K + 1)
    return kpx.to(torch.float16)


def compute_fractional_m_kpx(kpx: torch.Tensor, s: int, m: int) -> torch.Tensor:
    assert s >= 0
    assert m > 0
    max_k = kpx.shape[-1] - 1
    kpx = kpx.to(torch.float16)

    if s == 0:
        return kpx[..., 0]
    else:
        quotient, remainder = divmod(s, m)

        if quotient > max_k or quotient + 1 > max_k:
            return torch.zeros_like(
                input=kpx[..., 0], device=kpx.device, dtype=kpx.dtype
            )
        else:
            return (1 - remainder / m) * kpx[..., quotient] + remainder / m * kpx[
                ..., quotient + 1
            ]


def create_fractional_m_cdf(kpx: torch.Tensor, m: int) -> torch.Tensor:
    device = kpx.device
    max_k = kpx.shape[-1] - 1
    cdf = []
    for k in range(max_k * m + 1):
        cdf.append(1.0 - compute_fractional_m_kpx(kpx, k + 1, m))

    return torch.stack(cdf, dim=-1).to(device)
