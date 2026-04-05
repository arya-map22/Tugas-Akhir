from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


class LastRunMetadata(BaseModel):
    datetime: Annotated[datetime, _field]
    checkpoint_file_paths: Annotated[list[Path], _field]
