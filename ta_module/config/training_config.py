from typing import Annotated
from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


# Training Config
class Optimizer(BaseModel):
    learning_rate: Annotated[float, _field]


class Regularization(BaseModel):
    alfa: Annotated[float, _field]


class TrainingConfig(BaseModel):
    batch_size: Annotated[int, _field]
    epochs: Annotated[int, _field]
    optimizer: Annotated[Optimizer, _field]
    regularization: Annotated[Regularization, _field]
