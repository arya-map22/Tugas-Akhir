import json
import os

from pathlib import Path


def load_last_run_metadata(metadata_filepath: str | Path):
    if not os.path.exists(metadata_filepath):
        raise ValueError(f"{metadata_filepath} is not exist")

    metadata = None
    with open(metadata_filepath) as f:
        metadata = json.load(f)

    return metadata
