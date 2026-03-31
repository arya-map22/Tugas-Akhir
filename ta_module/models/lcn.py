from __future__ import annotations

from math import ceil, floor
from typing import Callable

import torch
from torch import nn, Tensor


class LocallyConnected2D(nn.Module):
    def __init__(
        self,
        input_size: int | tuple[int, int],
        kernel_size: int,
        activation_fn: nn.Module,
        stride: int = 1,
        dilation: int = 1,
        zero_padding: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.zero_padding = zero_padding
        self.activation_fn = activation_fn

        if isinstance(input_size, int):
            self.H_in, self.W_in = (input_size, input_size)
        else:
            self.H_in, self.W_in = input_size

        self.pad = (kernel_size - 1) / 2 if zero_padding else 0

        self.H_out: int = floor(
            (self.H_in + 2 * self.pad - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        self.W_out: int = floor(
            (self.W_in + 2 * self.pad - dilation * (kernel_size - 1) - 1) / stride + 1
        )

        self.weight = nn.Parameter(
            data=torch.rand(
                self.H_out, self.W_out, kernel_size, kernel_size, dtype=torch.float32
            )
        )

        if bias:
            self.bias = nn.Parameter(
                data=torch.rand(self.H_out, self.W_out, dtype=torch.float32)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor):
        unbatches = x.dim() == 2
        if unbatches:
            x = x.unsqueeze(0)

        N = x.size(0)
        k = self.kernel_size

        # Ubah dimensi x menjadi (N, 1, x_dim0, x_dim1): menambahkan dimensi untuk channel (C)
        x = x.unsqueeze(1)

        # Tambahkan padding 0 di sisi luar x
        if self.pad > 0:
            x = nn.functional.pad(
                x, [ceil(self.pad), floor(self.pad), ceil(self.pad), floor(self.pad)]
            )

        # Membuat tensor untuk diproses oleh kernel
        # Dimensi tensor (N, k*k, H_out*W_out)
        patches = nn.functional.unfold(x, kernel_size=k)

        # Ubah dimensi patches menjadi (N, k, k, H_out, W_out)
        patches = patches.view(N, k, k, self.H_out, self.W_out)

        # Dimensi patches (N, k, k, H_out, W_out): indeks (n, k, l, h, w)
        # Dimensi weight (H_out, W_out, k, k): indeks (h, w, k, l)
        # Sum pada indeks k dan l
        # Output tensor punya dimensi (N, H_out, W_out)
        y = torch.einsum("nklhw, hwkl -> nhw", patches, self.weight)

        # Tambahkan bias
        if self.bias is not None:
            # Ubah dimensi bias menjadi (N, H_out, W_out) sebelum dijumlahkan
            y = y + self.bias.unsqueeze(0).expand(N, -1, -1)

        if unbatches:
            y = y.squeeze(0)

        return self.activation_fn(y)

    @classmethod
    def factory(
        cls: LocallyConnected2D,
        input_size: int | tuple[int, int],
        kernel_size: int,
        activation_fn: nn.Module,
        stride: int = 1,
        dilation: int = 1,
        zero_padding: bool = False,
        bias: bool = True,
    ) -> Callable[[], LocallyConnected2D]:
        def create() -> LocallyConnected2D:
            return cls(
                input_size,
                kernel_size,
                activation_fn,
                stride,
                dilation,
                zero_padding,
                bias,
            )

        return create
