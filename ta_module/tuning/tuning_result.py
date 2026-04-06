from typing import Any

from pandas import DataFrame
from pydantic import BaseModel


class TuningResult(BaseModel):
    study_name: str
    trials: DataFrame
    best_params: dict[str, Any]
