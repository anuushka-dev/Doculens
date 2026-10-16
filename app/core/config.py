from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.paths import PROJECT_ROOT


class Settings(BaseSettings):
    """Validated configuration for the DocuLens application."""

    app_name: str = Field(
        default="DocuLens",
        min_length=1,
        max_length=80,
    )

    environment: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    log_level: Literal[
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
    ] = "INFO"

    max_upload_bytes: int = Field(
        default=20 * 1024 * 1024,
        gt=0,
        le=100 * 1024 * 1024,
    )

    max_pdf_pages: int = Field(
        default=250,
        gt=0,
        le=5000,
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="DOCULENS_",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
        frozen=True,
    )

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load validated settings once per application process."""
    return Settings()