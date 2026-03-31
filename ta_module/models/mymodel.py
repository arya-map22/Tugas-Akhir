from typing import Callable, Iterator

import lightning as L
import torch
from torch import Tensor, nn


class MyModel(L.LightningModule):

    def __init__(
        self,
        model: nn.Module,
        train_loss: nn.Module | Callable[[Tensor, Tensor], Tensor],
        eval_loss: nn.Module | Callable[[Tensor, Tensor], Tensor],
        # Pakai factory karena optimizer dan lr_scheduler harus dibuat di dalam configure_optimizers
        # Kalau passing objek jadi nanti params yang ketrack jadi ambigu
        # (bisa jadi tidak sesuai params model di model yang dibuat)
        optimizer_factory: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
        # Regularization untuk ditambahkan pada loss saat train
        regularization_loss: nn.Module | Callable[[Tensor, Tensor], Tensor] = None,
        lr_scheduler_factory: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ] = None,
        loss_log_scale: int = 0,
    ):
        super().__init__()
        # Semua argumen dalam __init__ yang bukan tipe primitif harus ignore dalam save_hyperparameters
        # agar load_from_checkpoint berfungsi
        self.save_hyperparameters(
            ignore=[
                "model",
                "train_loss",
                "eval_loss",
                "regularization_loss",
                "optimizer_factory",
                "lr_scheduler_factory",
            ]
        )

        # Hyperparameter (statis)
        # Digunakan untuk memberikan tampilan loss yang lebih readable (loss x 10^precision)
        self.log_loss_scale = loss_log_scale

        # Model utama yang diwrap oleh LightningModule (dinamis)
        self.model = model

        # Loss pada tahapan train, validation, test
        self.train_loss = train_loss
        self.eval_loss = eval_loss

        # Loss untuk regularisasi pada proses train
        self.regularization_loss = regularization_loss

        # Factory untuk membuat optimizer dan learning scheduler
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory

    def configure_optimizers(self):
        # optimizer pasti tracking params pada objek model ini
        optimizer = self.optimizer_factory(self.model.parameters())
        if self.lr_scheduler_factory is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.lr_scheduler_factory(optimizer),
                    "monitor": "val_loss",
                },
            }
        else:
            return optimizer

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)

        total_loss = 0.0

        # loss murni tanpa regularisasi
        train_loss = self.train_loss(y_hat, y)
        total_loss += train_loss

        if self.regularization_loss is not None:
            # loss dengan regularisasi -> untuk optimisasi parameter
            total_loss += self.regularization_loss(y_hat, y)

        # Log hanya pada loss murni agar dapat diinterpretasi karena loss murni hanya dipengaruhi oleh data
        # Juga agar train_loss dan val_loss dapat dibandingkan untuk deteksi overfit
        self.log(
            f"train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
        )

        # Tampilkan loss yang sudah discaled untuk memudahkan pengamatan
        train_loss_scaled = train_loss * 10**self.log_loss_scale
        self.log(
            f"train_loss_scaled (x10^{self.log_loss_scale})",
            train_loss_scaled,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Yang dipakai untuk optimisasi adalah total_loss
        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.eval_loss(y_hat, y)

        self.log(
            f"val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
        )

        # Tampilkan loss yang sudah discaled untuk memudahkan pengamatan
        val_loss_scaled = val_loss * 10**self.log_loss_scale
        self.log(
            f"val_loss_scaled (x10^{self.log_loss_scale})",
            val_loss_scaled,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return val_loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        test_loss = self.eval_loss(y_hat, y)

        self.log(
            f"test_loss",
            test_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Tampilkan loss yang sudah discaled untuk memudahkan pengamatan
        test_loss_scaled = test_loss * 10**self.log_loss_scale
        self.log(
            f"test_loss_scaled (x10^{self.log_loss_scale})",
            test_loss_scaled,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return test_loss
