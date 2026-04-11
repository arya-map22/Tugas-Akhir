from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from yaml import safe_load

from .data_config import DataConfig
from .dataset_config import DatasetConfig
from .model_config import ModelConfig
from .training_config import TrainingConfig
from .tuning_config import TuningConfig

_field = Field(frozen=True, repr=True)


# Config
class ModeEnum(StrEnum):
    TUNE = "tune"
    TRAIN = "train"
    INFERENCE = "inference"


class Split(BaseModel):
    train: Annotated[float, _field]
    validation: Annotated[float, _field]
    test: Annotated[float, _field]


class Config(BaseModel):
    mode: Annotated[ModeEnum, _field]
    seed: Annotated[int, _field]
    split: Annotated[Split, _field]
    data: Annotated[DataConfig, _field]
    dataset: Annotated[DatasetConfig, _field]
    training: Annotated[TrainingConfig, _field]
    model: Annotated[ModelConfig, _field]
    tuning: Annotated[TuningConfig, _field]


def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if config_path.suffix not in [".yaml", ".yml"]:
        raise ValueError(f"Config file must end with .yaml or .yml")

    with open(config_path, "r") as f:
        raw_config = safe_load(f)

    config = Config(**raw_config)

    return config
