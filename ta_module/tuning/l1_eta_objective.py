from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator
from zoneinfo import ZoneInfo

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from optuna import Trial
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ta_module.models import MyModel
from ta_module.utils import RegularizationLoss, get_current_run_datetime


def l1_eta_mymodel_objective(
    trial: Trial,
    model_factory: Callable[[], Module],
    train_loss: Module | Callable[[Tensor, Tensor], Tensor],
    eval_loss: Module | Callable[[Tensor, Tensor], Tensor],
    lasso_loss_factory: Callable[
        [float, Callable[[], Iterator[Parameter]]], RegularizationLoss
    ],
    optimizer_factory: Callable[[Iterator[Parameter]], Optimizer],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    log_dir: Path,
    checkpoint_dir: Path,
    eta_candidates: list[float],
) -> float:
    eta = trial.suggest_categorical("eta", eta_candidates)
    model = model_factory()

    mymodel = MyModel(
        model=model,
        train_loss=train_loss,
        eval_loss=eval_loss,
        optimizer_factory=optimizer_factory,
        regularization_loss=lasso_loss_factory(eta, lambda: model.parameters()),
    )

    run_datetime = get_current_run_datetime()

    checkpoint_dir = checkpoint_dir / "l1_eta" / f"Trial_{trial.number}"
    model_checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=run_datetime,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    log_dir = log_dir / "l1_eta"
    log_name = f"eta_{eta:.0e}".replace("-", "_").replace("+", "")
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
        version=run_datetime,
    )

    csv_logger = CSVLogger(save_dir=log_dir, name=log_name, version=run_datetime)

    trainer = Trainer(
        max_epochs=epochs,
        logger=[tensorboard_logger, csv_logger],
        callbacks=[model_checkpoint_cb],
        log_every_n_steps=1,
    )

    print("\n=====================================================================")
    print(f"Trial {trial.number}; eta = {eta:.0e}:")
    print("=====================================================================\n")
    trainer.fit(
        model=mymodel,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    assert model_checkpoint_cb.best_model_score is not None
    best_val_loss = model_checkpoint_cb.best_model_score.item()

    return best_val_loss
