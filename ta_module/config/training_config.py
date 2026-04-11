from typing import Annotated

from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


# Training Config
class Optimizer(BaseModel):
    lr: Annotated[float, _field]


class LRScheduler(BaseModel):
    patience: Annotated[int, _field]
    factor: Annotated[float, _field]
    min_lr: Annotated[float, _field]


class Regularization(BaseModel):
    alfa: Annotated[float, _field]


class EarlyStopping(BaseModel):
    patience: Annotated[int, _field]
    min_delta: Annotated[float, _field]


class TrainingConfig(BaseModel):
    batch_size: Annotated[int, _field]
    max_epochs: Annotated[int, _field]
    min_epochs: Annotated[int, _field]
    optimizer: Annotated[Optimizer, _field]
    lr_scheduler: Annotated[LRScheduler, _field]
    regularization: Annotated[Regularization, _field]
    early_stopping: Annotated[EarlyStopping, _field]
