from typing import Callable, Iterator

import lightning as L
import torch
from torch import Tensor, nn


class MyModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_metric: nn.Module | Callable[[Tensor, Tensor], Tensor],
        eval_metric: nn.Module | Callable[[Tensor, Tensor], Tensor],
        # Pakai factory karena optimizer dan lr_scheduler harus dibuat di dalam configure_optimizers
        # Kalau passing objek jadi nanti params yang ketrack jadi ambigu
        # (bisa jadi tidak sesuai params model di model yang dibuat)
        create_optimizer: Callable[[Iterator[nn.Parameter]], torch.optim.Optimizer],
        # Regularization untuk ditambahkan pada loss saat train
        regularization_term: nn.Module | Callable[[], Tensor] = None,
        create_lr_scheduler: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
        ] = None,
    ):
        super().__init__()
        # Semua argumen dalam __init__ yang bukan tipe primitif harus ignore dalam save_hyperparameters
        # agar load_from_checkpoint berfungsi
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss_metric",
                "eval_metric",
                "regularization_term",
                "create_optimizer",
                "create_lr_scheduler",
            ]
        )

        # Model utama yang diwrap oleh LightningModule (dinamis)
        self.model = model

        # Loss pada tahapan train, validation, test
        self.loss_metric = loss_metric
        self.eval_metric = eval_metric

        # Loss untuk regularisasi pada proses train
        self.regularization_term = regularization_term

        # Factory untuk membuat optimizer dan learning scheduler
        self.create_optimizer = create_optimizer
        self.create_lr_scheduler = create_lr_scheduler

        # Untuk avg train_loss dan val_loss di tensorboard plots
        self.train_losses = []
        self.train_regularized_losses = []
        self.val_losses = []
        self.val_scores = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self):
        # optimizer pasti tracking params pada objek model ini
        optimizer = self.create_optimizer(self.model.parameters())
        if self.create_lr_scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.create_lr_scheduler(optimizer),
                    "monitor": "val_loss",
                },
            }
        else:
            return optimizer

    def training_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        x, y = batch
        y_hat = self.model(x)

        total_loss = 0.0

        # loss murni tanpa regularisasi
        train_loss = self.loss_metric(y_hat, y)
        self.train_losses.append(train_loss.detach())
        total_loss += train_loss

        if self.regularization_term is not None:
            # loss dengan regularisasi -> untuk optimisasi parameter
            total_loss += self.regularization_term()
            self.train_regularized_losses.append(total_loss.detach())
            # Log regularized_loss untuk memberi gambaran pengaruh regularisasi
            self.log(
                "train_loss_regularized",
                total_loss,
                on_epoch=True,
                prog_bar=True,
            )

        # Log loss murni agar dapat diinterpretasi karena loss murni hanya dipengaruhi oleh data
        # Juga agar train_loss dan val_loss dapat dibandingkan untuk deteksi overfit
        self.log(f"train_loss", train_loss, on_epoch=True, prog_bar=True)

        # Yang dipakai untuk optimisasi adalah total_loss
        return total_loss

    def on_train_epoch_end(self) -> None:
        avg_train_loss = torch.stack(self.train_losses).mean()
        scalars = {"train_loss": avg_train_loss}

        if self.train_regularized_losses:  # hanya kalau ada regularisasi
            avg_train_regularized_loss = torch.stack(
                self.train_regularized_losses
            ).mean()
            scalars["train_loss_regularized"] = avg_train_regularized_loss

        writer = self.logger.experiment
        writer.add_scalars("Metrics", scalars, global_step=self.current_epoch)

        self.train_losses.clear()
        self.train_regularized_losses.clear()

    def validation_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        x, y = batch
        y_hat = self.model(x)
        val_loss = self.loss_metric(y_hat, y)
        self.val_losses.append(val_loss.detach())
        self.log(f"val_loss", val_loss, on_epoch=True, prog_bar=True)

        val_score = self.eval_metric(y_hat, y)
        self.val_scores.append(val_score.detach())
        self.log(f"val_score", val_score, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        avg_val_loss = torch.stack(self.val_losses).mean()
        avg_val_score = torch.stack(self.val_scores).mean()
        writer = self.logger.experiment
        writer.add_scalars(
            "Metrics",
            {"val_loss": avg_val_loss, "val_score": avg_val_score},
            global_step=self.current_epoch,
        )
        self.val_losses.clear()
        self.val_scores.clear()

    def test_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> None:
        x, y = batch
        y_hat = self.model(x)
        test_score = self.eval_metric(y_hat, y)

        self.log(
            f"test_score",
            test_score,
            on_epoch=True,
            prog_bar=True,
        )
