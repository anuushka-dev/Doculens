
"""Validated, immutable application configuration."""

from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.paths import PROJECT_ROOT

# PDF upload limits
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ALLOWED_UPLOAD_BYTES = 100 * 1024 * 1024

DEFAULT_MAX_PDF_PAGES = 250
MAX_ALLOWED_PDF_PAGES = 5_000


# Ollama configuration
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:2b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
DEFAULT_OLLAMA_NUM_CTX = 8_192
DEFAULT_OLLAMA_NUM_PREDICT = 1_024
DEFAULT_OLLAMA_TEMPERATURE = 0.0
DEFAULT_OLLAMA_SEED = 42


# Retrieval and evidence configuration
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_EVIDENCE_THRESHOLD = 0.25
DEFAULT_MAX_CONTEXT_CHARS = 12_000


# Structured extraction configuration
DEFAULT_EXTRACTION_MAX_REPAIR_ATTEMPTS = 1


class Settings(BaseSettings):
    """Validated, immutable application configuration."""

    # Application
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

    # PDF upload limits
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

    # Ollama
    ollama_base_url: str = Field(
        default=DEFAULT_OLLAMA_BASE_URL,
        min_length=1,
        max_length=500,
        description="Base URL of the local Ollama server.",
    )

    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        min_length=1,
        max_length=200,
        description="Ollama model name or tag.",
    )

    ollama_timeout_seconds: float = Field(
        default=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
        gt=0,
        le=600,
        allow_inf_nan=False,
        description="Maximum time allowed for an Ollama request.",
    )

    ollama_num_ctx: int = Field(
        default=DEFAULT_OLLAMA_NUM_CTX,
        gt=0,
        le=131_072,
        description="Ollama context window size.",
    )

    ollama_num_predict: int = Field(
        default=DEFAULT_OLLAMA_NUM_PREDICT,
        gt=0,
        le=8_192,
        description="Maximum number of generated tokens.",
    )

    ollama_temperature: float = Field(
        default=DEFAULT_OLLAMA_TEMPERATURE,
        ge=0,
        le=2,
        allow_inf_nan=False,
        description="Ollama generation temperature.",
    )

    ollama_seed: int = Field(
        default=DEFAULT_OLLAMA_SEED,
        ge=0,
        description="Fixed random seed for reproducible generation.",
    )

    # Retrieval and evidence
    retrieval_top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        ge=1,
        le=20,
        description="Maximum number of chunks retrieved for Q&A.",
    )

    evidence_threshold: float = Field(
        default=DEFAULT_EVIDENCE_THRESHOLD,
        ge=-1,
        le=1,
        allow_inf_nan=False,
        description=(
            "Provisional minimum cosine-similarity score. "
            "This is not a probability of answer correctness."
        ),
    )

    max_context_chars: int = Field(
        default=DEFAULT_MAX_CONTEXT_CHARS,
        ge=500,
        le=100_000,
        description="Maximum source-context characters sent to Ollama.",
    )

    # Structured extraction
    extraction_max_repair_attempts: int = Field(
        default=DEFAULT_EXTRACTION_MAX_REPAIR_ATTEMPTS,
        ge=0,
        le=1,
        description="Maximum schema/evidence repair attempts.",
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
        """Normalize environment names."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Normalize log-level names."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        """Require a valid HTTP or HTTPS server URL."""

        parsed = urlsplit(value)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "ollama_base_url must be a valid HTTP or HTTPS URL "
                "without credentials, query parameters, or fragments."
            )

        return value.rstrip("/")

    @field_validator("ollama_model")
    @classmethod
    def validate_ollama_model(cls, value: str) -> str:
        """Reject a blank model identifier."""
        if not value.strip():
            raise ValueError("ollama_model must not be blank")
        return value.strip()

    @field_validator(
        "max_upload_bytes",
        "max_pdf_pages",
        "ollama_num_ctx",
        "ollama_num_predict",
        "ollama_seed",
        "retrieval_top_k",
        "max_context_chars",
        "extraction_max_repair_attempts",
        mode="before",
    )
    @classmethod
    def reject_non_integer_numeric_limits(
        cls,
        value: object,
    ) -> object:
        """Reject booleans and floats for integer configuration fields.

        Numeric environment-variable strings remain supported by Pydantic.
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

    @field_validator(
        "ollama_timeout_seconds",
        "ollama_temperature",
        "evidence_threshold",
        mode="before",
    )
    @classmethod
    def reject_boolean_float_values(
        cls,
        value: object,
    ) -> object:
        """Do not accept booleans as numeric generation settings."""

        if isinstance(value, bool):
            raise ValueError(  # noqa: TRY004
                "Numeric settings must not be booleans."
            )

        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated settings for this process."""
    return Settings()
