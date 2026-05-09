from __future__ import annotations

from torch import Tensor
from torch.distributions import Transform

from .mortality_dataset import MortalityDataset


class TransformedYMortalityDataset(MortalityDataset):
    def __init__(
        self,
        mortality_matrix: Tensor,
        lookback: int,
        horizon: int,
        transform_fn: Transform,
    ):
        super().__init__(
            mortality_matrix=mortality_matrix,
            lookback=lookback,
            horizon=horizon,
        )

        self.transform_fn = transform_fn

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x, y = super().__getitem__(idx)
        y_transformed = self.transform_fn(y)

        return x, y_transformed

    @classmethod
    def factory(
        cls,
        lookback: int,
        horizon: int,
        transform_fn: Transform = None,
    ):
        def create(mortality_matrix: Tensor) -> TransformedYMortalityDataset:
            assert transform_fn is not None

            return cls(
                mortality_matrix=mortality_matrix,
                lookback=lookback,
                horizon=horizon,
                transform_fn=transform_fn,
            )

        return create
