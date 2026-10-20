
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.paths import PROJECT_ROOT

DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ALLOWED_UPLOAD_BYTES = 100 * 1024 * 1024

DEFAULT_MAX_PDF_PAGES = 250
MAX_ALLOWED_PDF_PAGES = 5_000


class Settings(BaseSettings):
    """Validated, immutable application configuration."""

    app_name: str = Field(
        default="DocuLens",
        min_length=1,
        max_length=80,
        description="Application display name.",
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
        default=DEFAULT_MAX_UPLOAD_BYTES,
        gt=0,
        le=MAX_ALLOWED_UPLOAD_BYTES,
        description="Maximum accepted PDF upload size, in bytes.",
    )

    max_pdf_pages: int = Field(
        default=DEFAULT_MAX_PDF_PAGES,
        gt=0,
        le=MAX_ALLOWED_PDF_PAGES,
        description="Maximum accepted PDF page count.",
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="DOCULENS_",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
        frozen=True,
        validate_default=True,
    )

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: object) -> object:
        """Accept valid environment names regardless of surrounding spaces."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Normalize log-level names before validating allowed values."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator(
        "max_upload_bytes",
        "max_pdf_pages",
        mode="before",
    )
    @classmethod
    def reject_non_integer_numeric_limits(
        cls,
        value: object,
    ) -> object:
        """Reject booleans and floats as resource limits.

        Numeric environment-variable strings such as "250" remain supported
        by Pydantic's normal integer parsing.
        """
        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                "Resource limits must be integers, not booleans."
            )

        if isinstance(value, float):
            raise ValueError(  # noqa: TRY004
                "Resource limits must be integers, not floats."
            )

        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated settings for this process."""
    return Settings()
