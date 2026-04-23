from __future__ import annotations

from typing import Callable, Collection

import torch
from torch import Tensor, nn, zeros


class LocalGLMnet(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        regression_attention_model: nn.Module,
        device: str,
        bias: bool = True,
    ):
        super().__init__()

        # Hyperparameter (statis)
        self.input_size = input_size
        self.output_size = input_size[1]
        self.device = device

        # Hyperparameter (dinamis) -> nilai parameter di dalamnya akan berubah-ubah ketika ditrain
        self.regression_attention_model = regression_attention_model

        # Parameter model (dinamis)
        if bias:
            self.bias = nn.Parameter(data=zeros(size=(self.output_size,))).to(
                self.device
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        # x harus batched
        assert x.dim() == 3
        x = x.to(self.device)

        if x.size()[1:] != self.input_size:
            raise ValueError(
                f"Size x={x.size()[1:]} tidak sama dengan input_size={self.input_size}: Ganti nilai x yang punya size={self.input_size}"
            )

        regression_attention: Tensor = self.regression_attention_model(x).to(
            self.device
        )
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
        y: Tensor = w_hadamard_x.sum(dim=1) + self.bias

        # Ubah dimensi y menjadi (N, 1, W) agar konsisten dengan MortalityDataset
        y = y.unsqueeze(1)

        return y.to(self.device)

    def get_regression_attention(self, x: Tensor) -> Tensor:
        return self.regression_attention_model(x).detach()

    @classmethod
    def factory(
        cls: LocalGLMnet,
        input_size: tuple[int, int],
        device: str,
        bias: bool = True,
    ) -> Callable[[nn.Module], LocalGLMnet]:
        def create(
            # Digunakan untuk membuat model LocalGLMnet dengan regression_attention_model yang berbeda
            # namun dengan parameter lain sama
            regression_attention_model: nn.Module,
        ) -> LocalGLMnet:
            return cls(
                device=device,
                input_size=input_size,
                regression_attention_model=regression_attention_model,
                bias=bias,
            )

        return create


class EnsembleLocalGLMNet(nn.Module):
    def __init__(
        self, models: Collection[LocalGLMnet] | nn.ModuleList, device: str
    ) -> None:
        assert all(isinstance(m, LocalGLMnet) for m in models)

        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3
        x = x.to(self.device)
        # y adalah tensor baru sehingga perlu diubah ke device
        y = torch.stack([model(x) for model in self.models]).mean(dim=0)

        return y.to(self.device)
