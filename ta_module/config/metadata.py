from json import load
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


def load_last_tune_metadata(metadata_filepath: Path) -> TuneMetadata:
    if not metadata_filepath.exists():
        raise FileNotFoundError(f"{metadata_filepath} is not exist")

    with open(metadata_filepath) as f:
        raw_metadata = load(f)

    metadata = TuneMetadata.model_validate(raw_metadata, extra="forbid")
    return metadata


def load_last_train_metadata(metadata_filepath: Path) -> TrainMetadata:
    if not metadata_filepath.exists():
        raise FileNotFoundError(f"{metadata_filepath} is not exist")

    with open(metadata_filepath) as f:
        raw_metadata = load(f)

    metadata = TrainMetadata.model_validate(raw_metadata, extra="forbid")
    return metadata
