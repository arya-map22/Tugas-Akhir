from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from ta_module.tuning import TuningResult

_field = Field(frozen=True, repr=True)


class TuneMetadata(BaseModel):
    datetime: Annotated[datetime, _field]
    result: Annotated[TuningResult, _field]


class TrainMetadata(BaseModel):
    datetime: Annotated[datetime, _field]
    checkpoint_file_paths: Annotated[list[Path], _field]
