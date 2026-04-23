from __future__ import annotations

from numpy import arange
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
        mortality_matrix: Tensor,
        lookback: int,
        horizon: int,
    ):
        assert mortality_matrix.dim() == 2

        self.mortality_matrix = mortality_matrix
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        n = self.mortality_matrix.shape[0] - self.lookback - self.horizon + 1
        assert n >= 0

        return n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        n = self.__len__()
        l = self.lookback
        h = self.horizon

        # handle negative index (Python style)
        if idx < 0:
            idx = n + idx

        # validasi index
        if idx < 0 or idx >= n:
            raise IndexError(f"idx hanya valid di [{-n}, {n})")

        # ambil index window
        x_ind = arange(idx, idx + l)
        y_ind = arange(idx + l, idx + l + h)

        x = self.mortality_matrix[x_ind, :]
        y = self.mortality_matrix[y_ind, :]

        return x, y

    @classmethod
    def factory(cls, lookback: int, horizon: int):
        def create(mortality_matrix: Tensor) -> MortalityDataset:
            return cls(
                mortality_matrix=mortality_matrix,
                lookback=lookback,
                horizon=horizon,
            )

        return create
