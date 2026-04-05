from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


# Model Config
class LocalGLMnet(BaseModel):
    bias: Annotated[bool, _field]


class LCN(BaseModel):
    kernel_size: Annotated[int, _field]
    stride: Annotated[int, _field]
    dilation: Annotated[int, _field]
    zero_padding: Annotated[bool, _field]
    bias: Annotated[bool, _field]


class ModelConfig(BaseModel):
    num_ensembles: Annotated[int, _field]
    localglmnet: Annotated[LocalGLMnet, _field]
    lcn: Annotated[LCN, _field]


# Training Config
class Optimizer(BaseModel):
    learning_rate: Annotated[float, _field]


class Regularization(BaseModel):
    eta: Annotated[float, _field]
    alfa: Annotated[float, _field]


class TrainingConfig(BaseModel):
    batch_size: Annotated[int, _field]
    epochs: Annotated[int, _field]
    optimizer: Annotated[Optimizer, _field]
    regularization: Annotated[Regularization, _field]


# Tuning config
class Tuning(BaseModel):
    eta_candidates: Annotated[list[float], _field]


# Data Config
class Mortalitas(BaseModel):
    year_col: Annotated[str, _field]
    date_format: Annotated[str, _field]
    age_col: Annotated[str, _field]
    sex_col: Annotated[str, _field]
    mortality_col: Annotated[str, _field]


class BIRate(BaseModel):
    date_col: Annotated[str, _field]
    date_format: Annotated[str, _field]


class DataConfig(BaseModel):
    mortalitas: Annotated[Mortalitas, _field]
    bi_rate: Annotated[BIRate, _field]


# Dataset Config
class DatasetConfig(BaseModel):
    lookback: Annotated[int, _field]
    horizon: Annotated[int, _field]


# Config
class ModeEnum(StrEnum):
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
    tuning: Annotated[Tuning, _field]


def load_config(config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if config_path.suffix not in [".yaml", ".yml"]:
        raise ValueError(f"Config file must end with .yaml or .yml")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    config = Config(**raw_config)

    return config
