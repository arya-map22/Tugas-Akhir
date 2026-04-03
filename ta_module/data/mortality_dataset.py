from __future__ import annotations

import numpy as np
import torch

from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset


class MortalityDataset(Dataset):
    """Dataset untuk forecasting mortalitas.

    Memberikan split x, y:
        - x = matriks lookback, dimensi (lookback x n_age)
        - y = matriks forecast, dimensi (horizon x n_age)
    """

    def __init__(
        self,
        mortality_matrix: DataFrame,
        lookback: int,
        horizon: int,
    ):
        self.mortality_matrix = mortality_matrix
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.mortality_matrix) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        n = self.__len__()
        l = self.lookback
        h = self.horizon

        if idx > 0 and idx >= n:
            raise ValueError(f"idx positif hanya valid di [{0}, {n})")

        if idx < 0 and -idx > n:
            raise ValueError(f"idx negatif hanya valid di [{-n}, 0)")

        x_ind = (
            np.arange(idx, idx + l)
            if idx >= 0
            else np.arange(n + idx * l - 1, n + idx * l + h - 1)
        )

        y_ind = (
            np.arange(idx + l, idx + l + h)
            if idx >= 0
            else np.arange(n + (idx + 1) * l - 1, n + (idx + 1) * l + h - 1)
        )

        x = self.mortality_matrix.iloc[x_ind, :].to_numpy(copy=True, dtype=np.float32)
        y = self.mortality_matrix.iloc[y_ind, :].to_numpy(copy=True, dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)

    @classmethod
    def factory(cls, lookback: int, horizon: int):
        def create(mortality_matrix: DataFrame) -> MortalityDataset:
            return cls(
                mortality_matrix=mortality_matrix,
                lookback=lookback,
                horizon=horizon,
            )

        return create
