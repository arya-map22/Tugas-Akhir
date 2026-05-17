import torch
from torch import Tensor

from .kpx import compute_fractional_m_kpx


def compute_m_annuity_epv(
    start_age: int,
    end_age: int,
    m: int,
    i: Tensor,
    kpx: Tensor,
    gender_age_prob: Tensor,
) -> Tensor:
    device = kpx.device

    m_annuity_epv = torch.tensor(0.0, device=device, dtype=torch.float32)
    i_m = m * ((1.0 + i).pow(1.0 / m) - 1.0)
    v_m = 1 / (1 + i_m / m)
    d_m = i_m / m * v_m
    for x1 in range(start_age, end_age + 1):
        age_x1_epv = torch.tensor(0.0, device=device, dtype=torch.float32)
        j = 0
        for x2 in range(x1 - start_age, kpx.shape[0]):
            for i in range(m):
                v = v_m.pow(j * m + i + 1)
                kmpx = compute_fractional_m_kpx(kpx=kpx[x2], s=j * m + i, m=m)
                km1px = compute_fractional_m_kpx(kpx=kpx[x2], s=j * m + i + 1, m=m)
                age_x1_epv += v * (kmpx - km1px)
            j += 1

        age_x1_epv = ((1 - age_x1_epv) / d_m) - 1
        m_annuity_epv += gender_age_prob[x1 - start_age] * age_x1_epv

    return m_annuity_epv


def compute_m_annuity_var(
    start_age: int,
    end_age: int,
    m: int,
    i: Tensor,
    kpx: Tensor,
    gender_age_prob: Tensor,
) -> Tensor:
    device = kpx.device

    m_annuity_var = torch.tensor(0.0, device=device, dtype=torch.float32)
    i_m = m * ((1.0 + i).pow(1.0 / m) - 1.0)
    v_m = 1 / (1 + i_m / m)
    d_m = i_m / m * v_m

    for x1 in range(start_age, end_age + 1):
        left_term = torch.tensor(0.0, device=device, dtype=torch.float32)
        right_term = torch.tensor(0.0, device=device, dtype=torch.float32)

        j = 0
        for x2 in range(x1 - start_age, kpx.shape[0]):
            for i in range(m):
                v = v_m.pow(j * m + i + 1)
                kmpx = compute_fractional_m_kpx(kpx=kpx[x2], s=j * m + i, m=m)
                km1px = compute_fractional_m_kpx(kpx=kpx[x2], s=j * m + i + 1, m=m)

                left_term += v.pow(2) * (kmpx - km1px)
                right_term += v * (kmpx - km1px)
            j += 1

        age_x1_var = (left_term - right_term.pow(2)) / d_m.pow(2)
        m_annuity_var += gender_age_prob[x1 - start_age] * age_x1_var

    return m_annuity_var


def compute_m_annuity_epv2(
    start_age: int,
    end_age: int,
    m: int,
    i: Tensor,
    kpx: Tensor,
    gender_age_prob: Tensor,
) -> Tensor:
    device = kpx.device

    m_annuity_epv2 = torch.tensor(0.0, device=device, dtype=torch.float32)
    i_m = m * ((1.0 + i).pow(1.0 / m) - 1.0)
    v_m = 1 / (1 + i_m / m)
    d_m = i_m / m * v_m
    for x1 in range(start_age, end_age + 1):
        age_x1_epv = torch.tensor(0.0, device=device, dtype=torch.float32)
        j = 0
        for x2 in range(x1 - start_age, kpx.shape[0]):
            for i in range(m):
                v = v_m.pow(j * m + i + 1)
                kmpx = compute_fractional_m_kpx(kpx=kpx[x2], s=j * m + i, m=m)
                km1px = compute_fractional_m_kpx(kpx=kpx[x2], s=j * m + i + 1, m=m)
                age_x1_epv += v * (kmpx - km1px)
            j += 1

        age_x1_epv = ((1 - age_x1_epv) / d_m) - 1
        m_annuity_epv2 += gender_age_prob[x1 - start_age] * age_x1_epv.pow(2)

    return m_annuity_epv2
