from __future__ import annotations

from typing import Callable, Iterator, Collection

import torch
from torch import nn, Tensor
from torch.nn import Parameter


def _l1(model_weights: Collection[Parameter], epsilon: float = 1e-6) -> Tensor:
    n_weights = sum(beta.numel() for beta in model_weights)
    l1_sum = sum(
        (torch.sqrt(beta.pow(2).sum() + epsilon) for beta in model_weights),
        start=torch.tensor(0.0),
    )

    # Nilai dirata-rata terhadap banyaknya weights agar skala konsisten
    # tidak dipengaruhi banyaknya weights
    return l1_sum / n_weights


def _l2(model_weights: Collection[Parameter]) -> Tensor:
    n_weights = sum(beta.numel() for beta in model_weights)
    l2_sum = sum((beta.pow(2).sum() for beta in model_weights), start=torch.tensor(0.0))

    # Nilai dirata-rata terhadap banyaknya weights agar skala konsisten
    # tidak dipengaruhi banyaknya weights
    return l2_sum / n_weights


class RegularizationLoss(nn.Module):
    def __init__(
        self,
        eta: float,
        alfa: float,
        # Pakai getter agar nilai weights yang digunakan pasti nilai weights terkini dari model yang ingin diregularisasi
        model_weights_getter: Callable[[], Iterator[Parameter]],
        l1_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.eta = eta
        self.alfa = alfa
        self.model_weights_getter = model_weights_getter
        self.l1_epsilon = l1_epsilon

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        assert pred.size() == target.size()

        model_weights = list(self.model_weights_getter())
        l1_penalty = _l1(model_weights=model_weights, epsilon=self.l1_epsilon)
        l2_penalty = _l2(model_weights=model_weights)

        regularization_loss = self.eta * (
            (1 - self.alfa) * l2_penalty + self.alfa * l1_penalty
        )

        return regularization_loss

    @classmethod
    def factory(
        cls: RegularizationLoss,
        eta: float,
        alfa: float,
        l1_epsilon: float = 1e-6,
    ) -> Callable[[Callable[[], Iterator[Parameter]]], RegularizationLoss]:
        def create(
            model_weights_getter: Callable[[], Iterator[Parameter]],
        ) -> RegularizationLoss:
            return cls(
                eta=eta,
                alfa=alfa,
                model_weights_getter=model_weights_getter,
                l1_epsilon=l1_epsilon,
            )

        return create
