from pathlib import Path
from typing import Annotated

from pydantic import DirectoryPath, Field, FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict

_field = Field(frozen=True, repr=True)


class DotEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Folder directories
    plots_dir: Annotated[DirectoryPath, _field]
    tuning_logs_dir: Annotated[DirectoryPath, _field]
    training_logs_dir: Annotated[DirectoryPath, _field]
    tuning_checkpoints_dir: Annotated[DirectoryPath, _field]
    training_checkpoints_dir: Annotated[DirectoryPath, _field]
    results_dir: Annotated[DirectoryPath, _field]

    # File paths
    last_tune_metadata_file: Annotated[FilePath, _field]
    last_train_metadata_file: Annotated[FilePath, _field]

    config_file: Annotated[FilePath, _field]

    mortalitas_file: Annotated[FilePath, _field]
    populasi_file: Annotated[FilePath, _field]
    bi_rate_file: Annotated[FilePath, _field]

    optuna_db_url: Annotated[str, _field]


def load_dot_env(env_file: FilePath = Path("./.env")) -> DotEnv:
    if not env_file.exists():
        raise FileNotFoundError(env_file)

    return DotEnv(_env_file=env_file)
