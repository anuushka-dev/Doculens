"""Pydantic contracts for page-aware chunks and ranked retrieval results.

This module defines and validates the data passed between Phase 2 components.
It intentionally does not perform chunking, embedding, vector search, file I/O,
API handling, or answer generation.
"""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Allow a tiny floating-point tolerance around the mathematical cosine range.
# Scores are still similarity scores, never calibrated probabilities.
COSINE_SCORE_TOLERANCE = 1e-5


class ChunkRecord(BaseModel):
    """A non-empty source-text chunk with immutable PDF provenance.

    start_char and end_char form a zero-based, half-open character span into
    the original extracted page text and can be used as page_text[start_char:end_char].
    page_number is the original, one-based PDF page number. Chunk offsets
    are Python string character offsets, not UTF-8 byte offsets or model tokens.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1, max_length=80, strict=True)
    document_id: str = Field(min_length=1, max_length=128, strict=True)
    file_name: str = Field(min_length=1, max_length=255, strict=True)
    page_number: int = Field(ge=1, strict=True)
    chunk_index: int = Field(ge=0, strict=True)
    start_char: int = Field(ge=0, strict=True)
    end_char: int = Field(gt=0, strict=True)
    text: str = Field(min_length=1, strict=True)

    @field_validator("chunk_id", "document_id", "file_name")
    @classmethod
    def normalize_identity_strings(cls, value: str) -> str:
        """Trim identity fields and reject values that are blank after trim."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("file_name")
    @classmethod
    def require_filename_not_path(cls, value: str) -> str:
        """Keep display metadata as a filename, never an arbitrary path."""
        if any(character in value for character in ("/", "\\", "\x00")):
            raise ValueError("file_name must be a basename, not a path")
        if any(not character.isprintable() for character in value):
            raise ValueError("file_name must not contain control characters")
        return value

    @model_validator(mode="after")
    def validate_offsets_and_text(self) -> ChunkRecord:
        """Ensure the stored excerpt matches its original character span."""
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("offset span length must equal the chunk text length")
        if not self.text.strip():
            raise ValueError("chunk text must contain non-whitespace characters")
        return self


class RetrievedChunk(ChunkRecord):
    """A source chunk returned by search with a cosine-similarity score.

    Inherits all chunk provenance and offset checks from ``ChunkRecord`` so
    that retrieval cannot silently discard or weaken source metadata checks.
    """

    score: float = Field(
        ge=-1.0 - COSINE_SCORE_TOLERANCE,
        le=1.0 + COSINE_SCORE_TOLERANCE,
        allow_inf_nan=False,
        strict=True,
        description="Cosine similarity; ranking signal, not a probability.",
    )


class RetrievalResponse(BaseModel):
    """Response contract for retrieval, before answer generation is added.

    results may contain fewer than top_k entries when the index has
    fewer chunks than requested. The tuple order must already be descending
    by similarity score, and source identifiers must be unique in one result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, strict=True)
    top_k: int = Field(ge=1, strict=True)
    results: tuple[RetrievedChunk, ...]

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        """Reject whitespace-only queries without changing the user's query."""
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def validate_ranked_results(self) -> RetrievalResponse:
        """Enforce result count, uniqueness, and descending score order."""
        if len(self.results) > self.top_k:
            raise ValueError("results cannot contain more entries than top_k")

        chunk_ids = [result.chunk_id for result in self.results]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("results must not contain duplicate chunk_id values")

        for previous, current in zip(self.results, self.results[1:]):
            if previous.score < current.score:
                raise ValueError("results must be ordered by descending score")

        return self
