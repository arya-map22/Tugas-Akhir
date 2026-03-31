from __future__ import annotations

from typing import Callable, Collection, Iterator

import torch
from torch import Tensor, nn
from torch.nn import Parameter


class RegularizationLoss(nn.Module):
    def __init__(
        self,
        # Koefisien untuk penalti regularisasi
        eta: float,
        # alfa = 0 -> ridge loss
        # alfa = 1 -> lasso loss
        alfa: float,
        # Pakai getter agar nilai weights yang digunakan pasti
        # nilai weights terkini dari model yang ingin diregularisasi
        model_weights_getter: Callable[[], Iterator[Parameter]],
        # epsilon digunakan untuk smoothing agar fungsi absolut
        # (dalam penalti l1) pada weight dapat diturunkan ketika = 0
        l1_epsilon: float = 1e-6,
    ):
        super().__init__()
        assert (
            eta > 0 and alfa > 0 and l1_epsilon > 0
        ), "Semua parameter regularisasi harus positif"
        assert 0 <= alfa <= 1, "alfa harus di range [0, 1]"

        self.eta = eta
        self.alfa = alfa
        self.model_weights_getter = model_weights_getter
        self.l1_epsilon = l1_epsilon

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        assert pred.size() == target.size(), "ukuran pred dan target harus sama"

        model_weights = list(self.model_weights_getter())
        l1_penalty = self._l1(model_weights=model_weights, epsilon=self.l1_epsilon)
        l2_penalty = self._l2(model_weights=model_weights)

        # Penalti regularisasi sesuai dengan rumus regularisasi ElasticNet
        regularization_loss = self.eta * (
            (1 - self.alfa) * l2_penalty + self.alfa * l1_penalty
        )

        return regularization_loss

    @staticmethod
    def _l1(model_weights: Collection[Parameter], epsilon: float = 1e-6) -> Tensor:
        return sum(
            (torch.sqrt(beta.pow(2).sum() + epsilon) for beta in model_weights),
            start=torch.tensor(0.0),
        )

    @staticmethod
    def _l2(model_weights: Collection[Parameter]) -> Tensor:
        return sum(
            (beta.pow(2).sum() for beta in model_weights), start=torch.tensor(0.0)
        )

    @classmethod
    def factory(
        cls: RegularizationLoss,
        eta: float,
        alfa: float,
        l1_epsilon: float = 1e-6,
    ) -> Callable[[Callable[[], Iterator[Parameter]]], RegularizationLoss]:
        def create(
            # digunakan untuk membuat RegularizationLoss terhadap parameter model lain
            # dengan parameter regularisasi yang sama
            model_weights_getter: Callable[[], Iterator[Parameter]],
        ) -> RegularizationLoss:
            return cls(
                eta=eta,
                alfa=alfa,
                model_weights_getter=model_weights_getter,
                l1_epsilon=l1_epsilon,
            )

        return create
