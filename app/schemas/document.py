"""Validated response schemas for extracted PDF documents."""

from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

NEAR_EMPTY_CHAR_THRESHOLD = 20

PageTextStatus = Literal["ok", "empty_text", "near_empty_text"]


class ExtractedPage(BaseModel):
    """Validated extracted text and metadata for one PDF page."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    page_number: int = Field(ge=1, strict=True)
    text: str = Field(strict=True)

    @computed_field
    @property
    def character_count(self) -> int:
        """Count extracted characters after trimming outer whitespace."""
        return len(self.text.strip())

    @computed_field
    @property
    def word_count(self) -> int:
        """Count whitespace-separated tokens for diagnostics."""
        return len(self.text.split())

    @computed_field
    @property
    def status(self) -> PageTextStatus:
        """Flag pages that may provide insufficient textual evidence."""
        character_count = self.character_count

        if character_count == 0:
            return "empty_text"

        if character_count < NEAR_EMPTY_CHAR_THRESHOLD:
            return "near_empty_text"

        return "ok"


class PDFExtractionResult(BaseModel):
    """Validated extraction result with document identity and page provenance."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    document_id: UUID
    filename: str = Field(
        min_length=1,
        max_length=255,
        strict=True,
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        strict=True,
    )
    pages: tuple[ExtractedPage, ...] = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Require a printable basename rather than a filesystem path."""
        value = value.strip()

        if not value:
            raise ValueError("filename must not be blank")

        if any(char in value for char in ("/", "\\", "\x00")):
            raise ValueError("filename must be a basename, not a path")

        if any(not char.isprintable() for char in value):
            raise ValueError("filename must not contain control characters")

        return value

    @model_validator(mode="after")
    def validate_page_sequence(self) -> "PDFExtractionResult":
        """Require consecutive, one-based pages and text somewhere in the PDF."""
        actual_numbers = tuple(page.page_number for page in self.pages)
        expected_numbers = tuple(range(1, len(self.pages) + 1))

        if actual_numbers != expected_numbers:
            raise ValueError(
                "pages must be ordered and numbered consecutively from 1"
            )

        if not any(page.text.strip() for page in self.pages):
            raise ValueError(
                "extraction must contain text on at least one page"
            )

        return self

    @computed_field
    @property
    def page_count(self) -> int:
        """Return the number of extracted pages."""
        return len(self.pages)

    @computed_field
    @property
    def total_character_count(self) -> int:
        """Return the combined character count across all pages."""
        return sum(page.character_count for page in self.pages)

    @computed_field
    @property
    def empty_page_numbers(self) -> tuple[int, ...]:
        """Return one-based page numbers with no extracted text."""
        return tuple(
            page.page_number
            for page in self.pages
            if page.status == "empty_text"
        )

    @computed_field
    @property
    def near_empty_page_numbers(self) -> tuple[int, ...]:
        """Return one-based page numbers with fewer than 20 characters."""
        return tuple(
            page.page_number
            for page in self.pages
            if page.status == "near_empty_text"
        )
