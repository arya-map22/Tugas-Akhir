from __future__ import annotations

from .mortality_dataset import MortalityDataset
from pandas import DataFrame
from torch import Tensor


class NormalizedMortalityDataset(MortalityDataset):
    def __init__(
        self,
        mortality_matrix: DataFrame,
        lookback: int,
        horizon: int,
        mean: Tensor,
        std: Tensor,
    ):
        super().__init__(
            mortality_matrix=mortality_matrix, lookback=lookback, horizon=horizon
        )

        # mean dan std dihitung terhadap kolom (per usia)
        # Dimensi mean dan std adalan (1, N_AGE)
        self.mean = mean
        self.std = std

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x, y = super().__getitem__(idx)
        x_normalized = (x - self.mean) / self.std
        y_normalized = (y - self.mean) / self.std

        return x_normalized, y_normalized

    @classmethod
    def factory(
        cls,
        lookback: int,
        horizon: int,
        # mean dan std default None agar signature sesuai dengan base class
        mean: Tensor = None,
        std: Tensor = None,
    ):
        def create(mortality_matrix: DataFrame) -> NormalizedMortalityDataset:
            assert mean is not None and std is not None

            return cls(
                mortality_matrix=mortality_matrix,
                lookback=lookback,
                horizon=horizon,
                mean=mean,
                std=std,
            )

        return create
