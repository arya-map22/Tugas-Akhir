import json
from pathlib import Path

from ta_module.config import TrainMetadata, TuneMetadata


def load_last_tune_metadata(metadata_filepath: Path) -> TuneMetadata:
    if not metadata_filepath.exists():
        raise FileNotFoundError(f"{metadata_filepath} is not exist")

    with open(metadata_filepath) as f:
        raw_metadata = json.load(f)

    metadata = TuneMetadata.model_validate(raw_metadata, extra="forbid")
    return metadata


def load_last_train_metadata(metadata_filepath: Path) -> TrainMetadata:
    if not metadata_filepath.exists():
        raise FileNotFoundError(f"{metadata_filepath} is not exist")

    with open(metadata_filepath) as f:
        raw_metadata = json.load(f)

    metadata = TrainMetadata.model_validate(raw_metadata, extra="forbid")
    return metadata
