from typing import Annotated
from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


# Dataset Config
class DatasetConfig(BaseModel):
    lookback: Annotated[int, _field]
    horizon: Annotated[int, _field]
