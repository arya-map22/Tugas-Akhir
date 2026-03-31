from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.distributions.transforms import Transform


class LocalGLMnet(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        link_fn: Transform,
        regression_attention_model: nn.Module,
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
            self.bias = nn.Parameter(data=torch.rand(self.output_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # Jika x unbatches (2D) ubah ke 3D dengan dimensi (N = 1, H, W)
        unbatches = x.dim() == 2
        if unbatches:
            x = x.unsqueeze(0)

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
        y: Tensor = self.link_fn.inv(w_hadamard_x.sum(dim=1) + self.bias)

        # Ubah dimensi y menjadi (N, 1, W) agar konsisten dengan MortalityDataset
        y = y.unsqueeze(1)

        # Jika x unbactches (2D), kembalikan y ke 2D dengan dimensi (1, W)
        if unbatches:
            y = y.squeeze(0)

        return y

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
                link_fn=link_fn,
                regression_attention_model=regression_attention_model,
                bias=bias,
            )

        return create
