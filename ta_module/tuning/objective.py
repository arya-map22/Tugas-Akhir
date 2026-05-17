from pathlib import Path
from typing import Callable

from lightning import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from optuna import Trial
from torch.utils.data import DataLoader

from ta_module.models import ModelLightning
from ta_module.utils import (
    get_current_run_datetime_str,
)


def reg_coef_objective(
    trial: Trial,
    create_my_model_with_reg_coef: Callable[[float], ModelLightning],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    max_epochs: int,
    log_dir: Path,
    checkpoint_dir: Path,
    reg_coef_candidates: list[float],
    create_callbacks: Callable[[], list[Callback]] = None,
    min_epochs: int = 0,
    gradient_clip_val: float | None = None,
) -> float:
    reg_coef = trial.suggest_categorical("reg_coef", reg_coef_candidates)
    mymodel = create_my_model_with_reg_coef(reg_coef)

    tuning_name = "tune_reg_coef_elasticnet_regularization"
    trial_name = f"Trial_{trial.number}_reg_coef_{reg_coef:.0e}".replace(
        "-", "_"
    ).replace("+", "")

    run_datetime = get_current_run_datetime_str()

    checkpoint_dir = checkpoint_dir / tuning_name / trial_name
    model_checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=run_datetime,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer_callbacks = []
    if create_callbacks is not None:
        trainer_callbacks.extend(create_callbacks())
    trainer_callbacks.append(model_checkpoint_cb)

    log_dir = log_dir / tuning_name
    tensorboard_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=trial_name,
        version=run_datetime,
    )

    csv_logger = CSVLogger(save_dir=log_dir, name=trial_name, version=run_datetime)

    trainer = Trainer(
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        gradient_clip_val=gradient_clip_val,
        logger=[tensorboard_logger, csv_logger],
        callbacks=trainer_callbacks,
        log_every_n_steps=1,
        deterministic=True,
    )

    print("\n=====================================================================")
    print(f"Trial {trial.number}; reg_coef = {reg_coef:.0e}:")
    print("=====================================================================\n")
    trainer.fit(
        model=mymodel,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    assert model_checkpoint_cb.best_model_score is not None
    best_val_loss = model_checkpoint_cb.best_model_score.item()

    return best_val_loss
