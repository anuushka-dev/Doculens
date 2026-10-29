"""Deterministic word-window chunking that preserves PDF page provenance.

Chunking deliberately happens inside each extracted page. Every returned
record keeps the original one-based page number and character offsets into
the exact normalized page text provided by Phase 1.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

from app.schemas.retrieval import ChunkRecord


class PageLike(Protocol):
    """Structural interface implemented by Phase 1 extracted-page models."""

    page_number: int
    text: str


_WORD = re.compile(r"\S+", flags=re.UNICODE)


def _normalized_identity(
    value: str,
    *,
    field_name: str,
    max_length: int,
) -> str:
    """Validate and normalize a non-blank identity string."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()

    if not normalized:
        raise ValueError(f"{field_name} must not be blank")

    if len(normalized) > max_length:
        raise ValueError(
            f"{field_name} must be at most {max_length} characters"
        )

    return normalized


def _validate_file_name(file_name: str) -> str:
    """Require display metadata to be a printable basename, not a path."""
    normalized = _normalized_identity(
        file_name,
        field_name="file_name",
        max_length=255,
    )

    if any(character in normalized for character in ("/", "\\", "\x00")):
        raise ValueError("file_name must be a basename, not a path")

    if any(not character.isprintable() for character in normalized):
        raise ValueError("file_name must not contain control characters")

    return normalized


def chunk_pages(
    pages: Sequence[PageLike],
    *,
    document_id: str,
    file_name: str,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> list[ChunkRecord]:
    """Split each page into deterministic, overlapping word windows.

    Args:
        pages: Phase 1 page records with positive one-based page numbers
            and extracted text.
        document_id: Stable identity assigned by the ingestion layer.
        file_name: Printable source filename, not a filesystem path.
        chunk_size_words: Maximum number of non-whitespace word spans
            per chunk.
        overlap_words: Number of word spans repeated between adjacent chunks.

    Returns:
        Chunk records in input page order and then ascending chunk order.
        Empty or whitespace-only pages produce no chunks and do not change
        the original page numbers.

    Raises:
        TypeError: Identity fields or numeric parameters have the wrong type.
        ValueError: Configuration, page ordering, or source metadata is invalid.

    Notes:
        Character offsets are zero-based, half-open Python string indices
        into the original page text. Therefore:
        page.text[chunk.start_char:chunk.end_char] == chunk.text

        Whitespace between words is retained. Whitespace outside the first
        and last word in a chunk is excluded.
    """
    normalized_document_id = _normalized_identity(
        document_id,
        field_name="document_id",
        max_length=128,
    )
    normalized_file_name = _validate_file_name(file_name)

    if (
        isinstance(chunk_size_words, bool)
        or not isinstance(chunk_size_words, int)
    ):
        raise TypeError("chunk_size_words must be an integer")

    if isinstance(overlap_words, bool) or not isinstance(overlap_words, int):
        raise TypeError("overlap_words must be an integer")

    if chunk_size_words < 1:
        raise ValueError("chunk_size_words must be positive")

    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be >= 0 and smaller than chunk_size_words"
        )

    chunk_records: list[ChunkRecord] = []
    seen_pages: set[int] = set()
    previous_page_number = 0

    for page in pages:
        page_number = page.page_number
        page_text = page.text

        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise ValueError(
                "page_number must be a positive one-based integer"
            )

        if page_number in seen_pages:
            raise ValueError(f"duplicate page_number: {page_number}")

        if page_number <= previous_page_number:
            raise ValueError(
                "pages must be ordered by ascending page_number"
            )

        seen_pages.add(page_number)
        previous_page_number = page_number

        if not isinstance(page_text, str):
            raise TypeError(
                f"page {page_number} text must be a string"
            )

        word_spans = list(_WORD.finditer(page_text))

        if not word_spans:
            continue

        start_word = 0
        chunk_index = 0

        while start_word < len(word_spans):
            end_word = min(
                start_word + chunk_size_words,
                len(word_spans),
            )

            start_char = word_spans[start_word].start()
            end_char = word_spans[end_word - 1].end()
            chunk_text = page_text[start_char:end_char]

            # Deterministic identity for the same document, page,
            # chunk position, offsets, and source text.
            identity = "\0".join(
                (
                    normalized_document_id,
                    str(page_number),
                    str(chunk_index),
                    str(start_char),
                    str(end_char),
                    chunk_text,
                )
            )

            chunk_id = (
                "chunk_"
                + hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()[:24]
            )

            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=normalized_document_id,
                    file_name=normalized_file_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    text=chunk_text,
                )
            )

            if end_word == len(word_spans):
                break

            start_word = end_word - overlap_words
            chunk_index += 1

    return chunk_records
