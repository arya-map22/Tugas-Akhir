import numpy as np
import torch

from pandas import DataFrame
from pandas.api.types import is_datetime64_any_dtype
from torch.utils.data import Dataset


class MortalityDataset(Dataset):
    """Dataset untuk forecasting mortalitas.

    Memberikan split x, y:
        - x = matriks lookback, dimensi (lookback x n_age)
        - y = matriks forecast, dimensi (horizon x n_age)
    """

    def __init__(
        self,
        df: DataFrame,
        mortality_col: str,
        age_col: str,
        year_col: str,
        lookback: int,
        horizon: int,
    ):
        if lookback <= 0:
            raise ValueError("lookback harus > 0")
        if horizon <= 0:
            raise ValueError("horizon harus > 0")
        if df.isna().any(axis=None):
            raise ValueError("Terdapat missing value pada df")
        if not is_datetime64_any_dtype(df[year_col]):
            raise TypeError("df[year_col] harus tipe datetime-like")

        self.df_pivoted = df.pivot(
            columns=age_col, index=year_col, values=mortality_col
        )

        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):
        return len(self.df_pivoted) - self.lookback - self.horizon + 1

    def __getitem__(self, idx: int):
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

        x = self.df_pivoted.iloc[x_ind, :].to_numpy(copy=True, dtype=np.float32)
        y = self.df_pivoted.iloc[y_ind, :].to_numpy(copy=True, dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)
