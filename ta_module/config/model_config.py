from typing import Annotated

from pydantic import BaseModel, Field

_field = Field(frozen=True, repr=True)


# Model Config
class LocalGLMnet(BaseModel):
    bias: Annotated[bool, _field]


class LCN(BaseModel):
    kernel_size: Annotated[int, _field]
    stride: Annotated[int, _field]
    dilation: Annotated[int, _field]
    zero_padding: Annotated[bool, _field]
    bias: Annotated[bool, _field]


class ModelConfig(BaseModel):
    num_ensembles: Annotated[int, _field]
    localglmnet: Annotated[LocalGLMnet, _field]
    lcn: Annotated[LCN, _field]
