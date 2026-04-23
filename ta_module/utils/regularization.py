from __future__ import annotations

from typing import Callable, Collection, Iterator

import torch
from torch import Tensor, nn
from torch.nn import Parameter


class ElasticNetRegularizationTerm(nn.Module):
    def __init__(
        self,
        # Koefisien untuk penalti regularisasi
        reg_coef: float,
        # alpha = 0 -> ridge loss
        # alpha = 1 -> lasso loss
        alpha: float,
        # Pakai getter agar nilai weights yang digunakan pasti
        # nilai weights terkini dari model yang ingin diregularisasi
        model_weights_getter: Callable[[], Iterator[Parameter]],
        # epsilon digunakan untuk smoothing agar fungsi absolut
        # (dalam penalti l1) pada weight dapat diturunkan ketika = 0
        l1_epsilon: float = 1e-6,
    ):
        super().__init__()
        assert reg_coef >= 0 and alpha >= 0 and l1_epsilon >= 0
        assert 0 <= alpha <= 1, "alpha harus di range [0, 1]"

        self.reg_coef = reg_coef
        self.alpha = alpha
        self.model_weights_getter = model_weights_getter
        self.l1_epsilon = l1_epsilon

    def forward(self) -> Tensor:
        model_weights = tuple(self.model_weights_getter())
        l1_penalty = self._smooth_l1(
            model_weights=model_weights, epsilon=self.l1_epsilon
        )
        l2_penalty = self._l2(model_weights=model_weights)

        # Penalti regularisasi sesuai dengan rumus regularisasi ElasticNet
        regularization_loss = self.reg_coef * (
            (1 - self.alpha) * l2_penalty + self.alpha * l1_penalty
        )

        return regularization_loss

    @staticmethod
    def _smooth_l1(model_weights: Collection[Parameter], epsilon: float) -> Tensor:
        return torch.stack(
            [torch.sqrt(w.pow(2) + epsilon).sum() for w in model_weights]
        ).sum()

    @staticmethod
    def _l2(model_weights: Collection[Parameter]) -> Tensor:
        return torch.stack([w.pow(2).sum() for w in model_weights]).sum()

    @classmethod
    def factory(
        cls: ElasticNetRegularizationTerm,
        reg_coef: float,
        alpha: float,
        l1_epsilon: float = 1e-8,
    ) -> Callable[[Callable[[], Iterator[Parameter]]], ElasticNetRegularizationTerm]:
        def create(
            # digunakan untuk membuat RegularizationLoss terhadap parameter model lain
            # dengan parameter regularisasi yang sama
            model_weights_getter: Callable[[], Iterator[Parameter]],
        ) -> ElasticNetRegularizationTerm:
            return cls(
                reg_coef=reg_coef,
                alpha=alpha,
                model_weights_getter=model_weights_getter,
                l1_epsilon=l1_epsilon,
            )

        return create
