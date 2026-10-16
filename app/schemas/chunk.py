
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextChunk(BaseModel):
    """A bounded text passage with traceable PDF source metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: str = Field(min_length=1, max_length=200)
    document_id: UUID
    filename: str = Field(min_length=1, max_length=255)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    # Zero-based character offsets into the normalized source page text.
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)

    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_span(self) -> "TextChunk":
        """Ensure offsets describe exactly the stored passage."""
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")

        if self.end_char - self.start_char != len(self.text):
            raise ValueError(
                "source span length must match the chunk text length"
            )

        if not self.text.strip():
            raise ValueError("chunk text must not be blank")

        return self
