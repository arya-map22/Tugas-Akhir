from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_field = Field(frozen=True, repr=True)


class DotEnv(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Folder directories
    plots_dir: Annotated[Path, _field]
    tuning_logs_dir: Annotated[Path, _field]
    training_logs_dir: Annotated[Path, _field]
    tuning_checkpoints_dir: Annotated[Path, _field]
    training_checkpoints_dir: Annotated[Path, _field]

    # File paths
    last_run_metadata_file: Annotated[Path, _field]
    config_file: Annotated[Path, _field]

    mortalitas_file: Annotated[Path, _field]
    populasi_file: Annotated[Path, _field]
    bi_rate_file: Annotated[Path, _field]

    optuna_db_url: Annotated[str, _field]


def load_dot_env(env_file: Path = Path("./.env")) -> DotEnv:
    if not env_file.exists():
        raise FileNotFoundError(env_file)

    return DotEnv(_env_file=env_file)
