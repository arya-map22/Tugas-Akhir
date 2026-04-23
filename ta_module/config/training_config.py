from typing import Annotated

from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


# Training Config
class Optimizer(BaseModel):
    lr: Annotated[float, _field]


class LRScheduler(BaseModel):
    # LinearLR
    start_factor: Annotated[float, _field]
    end_factor: Annotated[float, _field]
    total_iters: Annotated[int, _field]

    # CosineAnnealingWarmRestarts
    T_0: Annotated[int, _field]
    T_mult: Annotated[int, _field]
    eta_min: Annotated[float, _field]


class Regularization(BaseModel):
    alpha: Annotated[float, _field]
    gradient_clip_val: Annotated[float, _field]


class TrainingConfig(BaseModel):
    batch_size: Annotated[int, _field]
    max_epochs: Annotated[int, _field]
    min_epochs: Annotated[int, _field]
    optimizer: Annotated[Optimizer, _field]
    lr_scheduler: Annotated[LRScheduler, _field]
    regularization: Annotated[Regularization, _field]
