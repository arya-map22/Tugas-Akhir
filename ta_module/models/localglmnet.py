from __future__ import annotations

from typing import Callable, Collection

import torch
from torch import Tensor, nn, zeros
from torch.distributions import Transform


class LocalGLMnet(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        regression_attention_model: nn.Module,
        link_fn: Transform,
        bias: bool = True,
    ):
        super().__init__()
        # Hyperparameter (statis)
        self.input_size = input_size
        self.output_size = input_size[1]
        self.link_fn = link_fn

        # Hyperparameter (dinamis) -> nilai parameter di dalamnya akan berubah-ubah ketika ditrain
        self.regression_attention_model = regression_attention_model

        # Parameter model (dinamis)
        if bias:
            self.bias = nn.Parameter(data=zeros(size=(self.output_size,)))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # x harus batched
        assert x.dim() == 3

        if x.size()[1:] != self.input_size:
            raise ValueError(
                f"Size x={x.size()[1:]} tidak sama dengan input_size={self.input_size}: Ganti nilai x yang punya size={self.input_size}"
            )

        regression_attention: Tensor = self.regression_attention_model(x)
        if regression_attention.size()[1:] != self.input_size:
            raise AttributeError(
                f"Size dari regression_atention_model(x) {self.regression_attention.size()[1:]} tidak sama dengan input_size={self.input_size}: Definisikan ulang LocalGLMnet dengan regression_attention_model yang menghasilkan output_size sama dengan input_size-nya"
            )

        # regression_attention punya dimensi (N, H, W)
        # x punya dimensi (N, H, W)

        # Hadamard product untuk regression_attention dan x, punya dimensi (N, H, W)
        w_hadamard_x = regression_attention * x

        # self.bias punya dimensi (W)
        # self.bias akan dibroadcast menjadi (N, W)
        # y punya dimensi (N, W)
        if self.bias is not None:
            y: Tensor = w_hadamard_x.sum(dim=1) + self.bias
        else:
            y: Tensor = w_hadamard_x.sum(dim=1)

        # Ubah dimensi y menjadi (N, 1, W) agar konsisten dengan MortalityDataset
        y = y.unsqueeze(1)

        return self.link_fn.inv(y)

    def get_regression_attention(self, x: Tensor) -> Tensor:
        return self.regression_attention_model(x).detach()

    @classmethod
    def factory(
        cls: LocalGLMnet,
        input_size: tuple[int, int],
        link_fn: Transform,
        bias: bool = True,
    ) -> Callable[[nn.Module], LocalGLMnet]:
        def create(
            # Digunakan untuk membuat model LocalGLMnet dengan regression_attention_model yang berbeda
            # namun dengan parameter lain sama
            regression_attention_model: nn.Module,
        ) -> LocalGLMnet:
            return cls(
                input_size=input_size,
                regression_attention_model=regression_attention_model,
                link_fn=link_fn,
                bias=bias,
            )

        return create


class EnsembleLocalGLMNet(nn.Module):
    def __init__(
        self,
        models: Collection[LocalGLMnet],
        weight_per_model: Collection[float] | None = None,
    ) -> None:
        assert all(isinstance(m, LocalGLMnet) for m in models)

        if weight_per_model is not None:
            assert sum(weight_per_model) == 1.0
            assert len(models) == len(weight_per_model)

            self.weight_per_model = weight_per_model
        else:
            self.weight_per_model = [1.0 / len(models) for _ in range(len(models))]

        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3

        y = [
            self.weight_per_model[i] * self.models[i](x)
            for i in range(len(self.models))
        ]
        y = torch.stack(y, dim=0)
        y = torch.sum(y, dim=0)

        return y
