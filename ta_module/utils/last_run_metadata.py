import json
from pathlib import Path

from ta_module.config import LastRunMetadata


def load_last_run_metadata(metadata_filepath: Path):
    if not metadata_filepath.exists():
        raise FileNotFoundError(f"{metadata_filepath} is not exist")

    with open(metadata_filepath) as f:
        raw_metadata = json.load(f)

    metadata = LastRunMetadata.model_validate(raw_metadata, extra="forbid")
    return metadata
