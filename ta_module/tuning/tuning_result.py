from typing import Any
from pydantic import BaseModel


class TuningResult(BaseModel):
    study_name: str
    trials: dict[str, list[Any]]
    best_params: dict[str, Any]
