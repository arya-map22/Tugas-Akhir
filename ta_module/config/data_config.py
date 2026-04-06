from typing import Annotated
from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


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
