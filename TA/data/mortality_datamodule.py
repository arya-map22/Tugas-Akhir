from typing import Collection
import torch
import lightning as L
from torch.utils.data import DataLoader, Subset, ConcatDataset
from .mortality_dataset import MortalityDataset


class MortalityDataModule(L.LightningDataModule):
    def __init__(
        self,
        mortality_datasets: Collection[MortalityDataset],
        batch_sizes: int | tuple[int, int, int] = None,
        splits: tuple[float, float, float] = None,
    ):
        super().__init__()

        self.mortality_datasets = list(mortality_datasets)
        n_datasets = [len(d) for d in self.mortality_datasets]

        assert all(n >= 3 for n in n_datasets), "Each dataset must have >= 3 samples"

        # ========================
        # Batch size handling
        # ========================
        if batch_sizes is None:
            self.train_batch_size = self.val_batch_size = self.test_batch_size = -1
        elif isinstance(batch_sizes, int):
            assert batch_sizes >= 0 or batch_sizes == -1
            self.train_batch_size = self.val_batch_size = self.test_batch_size = (
                batch_sizes
            )
        else:
            assert all(x >= 0 or x == -1 for x in batch_sizes)
            self.train_batch_size, self.val_batch_size, self.test_batch_size = (
                batch_sizes
            )

        # ========================
        # Split handling
        # ========================
        if splits is None:
            self.train_split = 0.8
            self.val_split = 0.1
            self.test_split = 0.1
        else:
            assert all(x >= 0 for x in splits), "splits must be >= 0"
            assert splits[0] > 0, "train_split must be > 0"
            assert sum(splits) == 1, "splits must sum to 1"

            self.train_split = splits[0]
            self.val_split = splits[1]
            self.test_split = splits[2]

        assert (
            self.train_split > self.test_split and self.train_split > self.val_split
        ), "train split must be largest"

        self.train_subsets = None
        self.val_subsets = None
        self.test_subsets = None

    def setup(self, stage: str = None):
        train_sizes, val_sizes, test_sizes = [], [], []

        for dataset in self.mortality_datasets:
            n = len(dataset)
            tr_split = self.train_split
            va_split = self.val_split
            te_split = self.test_split

            tr = self._get_split_size(tr_split, dataset)
            va = self._get_split_size(va_split, dataset)
            te = self._get_split_size(te_split, dataset)

            remainder = n - (tr + va + te)

            # Pastikan val & test tidak nol
            if va == 0 and remainder > 0 and va_split > 0:
                va += 1
                remainder -= 1

            if te == 0 and remainder > 0 and te_split > 0:
                te += 1
                remainder -= 1

            # Sisanya masuk train (logika utama tetap sama secara intent)
            tr += remainder

            train_sizes.append(tr)
            val_sizes.append(va)
            test_sizes.append(te)

        # ========================
        # Build subsets
        # ========================
        self.train_subsets = []
        self.val_subsets = []
        self.test_subsets = []

        for dataset, tr, va, te in zip(
            self.mortality_datasets, train_sizes, val_sizes, test_sizes
        ):
            train_subset = Subset(dataset, range(0, tr))
            val_subset = Subset(dataset, range(tr, tr + va))
            test_subset = Subset(dataset, range(tr + va, tr + va + te))

            self.train_subsets.append(train_subset)
            self.val_subsets.append(val_subset)
            self.test_subsets.append(test_subset)

    # ========================
    # Helper
    # ========================
    @staticmethod
    def _get_split_size(split: float, dataset: MortalityDataset) -> int:
        n = len(dataset)
        return int(split * n)

    @staticmethod
    def _resolve_batch_size(batch_size: int, subset) -> int:
        return batch_size if batch_size > 0 else len(subset)

    def _build_dataloaders(self, subsets, batch_size, shuffle: bool):
        combined_subset = ConcatDataset(subsets)
        return DataLoader(
            dataset=combined_subset,
            batch_size=self._resolve_batch_size(batch_size, combined_subset),
            shuffle=shuffle,
        )

    # ========================
    # Dataloaders
    # ========================
    def train_dataloader(self):
        return self._build_dataloaders(
            self.train_subsets, self.train_batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self._build_dataloaders(
            self.val_subsets, self.val_batch_size, shuffle=False
        )

    def test_dataloader(self):
        return self._build_dataloaders(
            self.test_subsets, self.test_batch_size, shuffle=False
        )

    def train_dataset(self):
        return self.train_subsets

    def val_dataset(self):
        return self.val_subsets

    def test_dataset(self):
        return self.test_subsets
