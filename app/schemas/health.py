from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Response contract for the application health endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    status: Literal["ok"]
    service: Literal["doculens-api"]