import torch
import torch.nn as nn

from typing import Callable, Collection
from torch import Tensor
from lightning import LightningModule


class EnsembleLocalGLMNet(nn.Module):
    def __init__(
        self,
        models: Collection[nn.Module] | nn.ModuleList,
        forecast_horizon: int,
    ) -> None:
        super().__init__()

        self.models = nn.ModuleList(models)
        self.forecast_horizon = forecast_horizon

    def forward(self, x: Tensor) -> Tensor:
        y = torch.stack([model(x) for model in self.models]).mean(dim=0)

        return y

    @torch.no_grad()
    def forecast(self, x: Tensor) -> Tensor:
        assert x.dim() == 3

        x_in = x
        predictions = []

        for _ in range(self.forecast_horizon):
            y_hat = self.forward(x_in)
            predictions.append(y_hat)
            # Drop elemen pertama x_in dan tambahkan y_hat di akhir untuk membuat input baru prediksi berikutnya
            x_in = torch.cat([x_in[:, 1:, :], y_hat], dim=1)

        # return semua prediksi setiap step
        return torch.stack(predictions)


class EnsembleLocalGLMnetLightning(LightningModule):
    def __init__(
        self,
        model: EnsembleLocalGLMNet,
        eval_loss: nn.Module | Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.model = model
        self.eval_loss = eval_loss

    def training_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        return None

    def configure_optimizers(self) -> None:
        return None

    def validation_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        x, y = batch
        y_hat = self.forward(x)
        val_loss = self.eval_loss(y, y_hat)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return val_loss

    def test_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        x, y = batch
        y_hat = self.forward(x)
        test_loss = self.eval_loss(y, y_hat)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)

        return test_loss

    def predict_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        x, y = batch
        x_in = torch.cat([x[:, 1:, :], y], dim=1)

        return self.model.forecast(x_in)
