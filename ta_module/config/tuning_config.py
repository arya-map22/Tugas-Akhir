from typing import Annotated

from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


# Tuning config
class TuningConfig(BaseModel):
    reg_coef_candidates: Annotated[list[float], _field]
