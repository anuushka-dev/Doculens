"""Deterministic citation and field-evidence provenance checks."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.answer import Citation, EvidenceQuoteDraft, EvidenceReference
from app.schemas.extraction import (
    ExtractionEvidenceDraft,
    ExtractionFieldCitation,
    MetadataField,
)
from app.schemas.retrieval import ChunkRecord, RetrievedChunk

# Normalize quote glyphs for comparison only. Returned excerpts are sliced
# from the unmodified original source text, retaining its typographic form.
_QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "‹": "'",
        "›": "'",
    }
)


@dataclass(frozen=True)
class VerifiedQuote:
    """A verified quote tied to its original source chunk."""

    chunk: ChunkRecord
    excerpt: str


class EvidenceValidationError(ValueError):
    """A proposed evidence reference cannot be tied to retrieved source text."""


def _normalize_with_source_map(
    value: str,
) -> tuple[str, list[int], list[int]]:
    """Normalize text while mapping normalized characters to source spans.

    Applies quote-glyph translation, NFKC normalization and case-folding,
    then collapses whitespace. Combining marks are grouped with their
    preceding character before normalization. Source offsets allow verified
    excerpts to be sliced from original text instead of returning model text.
    """
    translated = value.translate(_QUOTE_TRANSLATION)
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    pending_space: tuple[int, int] | None = None

    index = 0
    while index < len(translated):
        cluster_start = index
        index += 1
        while (
            index < len(translated)
            and unicodedata.combining(translated[index]) != 0
        ):
            index += 1
        cluster_end = index

        normalized = unicodedata.normalize(
            "NFKC",
            translated[cluster_start:cluster_end],
        ).casefold()

        for char in normalized:
            if char.isspace():
                if chars:
                    if pending_space is None:
                        pending_space = (cluster_start, cluster_end)
                    else:
                        pending_space = (pending_space[0], cluster_end)
                continue

            if pending_space is not None:
                chars.append(" ")
                starts.append(pending_space[0])
                ends.append(pending_space[1])
                pending_space = None

            chars.append(char)
            starts.append(cluster_start)
            ends.append(cluster_end)

    return "".join(chars), starts, ends


def _find_source_excerpt(source: str, quote: str) -> str:
    """Find a quote in normalized text and return its original source span."""
    normalized_source, starts, ends = _normalize_with_source_map(source)
    normalized_quote, _, _ = _normalize_with_source_map(quote)

    if not normalized_quote:
        raise EvidenceValidationError(
            "evidence quote is blank after normalization"
        )

    position = normalized_source.find(normalized_quote)
    if position < 0:
        raise EvidenceValidationError(
            "evidence quote does not match the retrieved chunk"
        )

    first_index = position
    last_index = position + len(normalized_quote) - 1
    excerpt = source[starts[first_index]:ends[last_index]]
    if not excerpt.strip():
        raise EvidenceValidationError("matched evidence excerpt is empty")
    return excerpt


def _is_superseded_document_date(source: str, value: str) -> bool:
    """Identify dates explicitly attributed to a superseded document."""
    date_pattern = re.escape(value.strip())
    pattern = rf"\bsupersedes\b[^.]*?\bdated\s+{date_pattern}(?!\w)"
    return re.search(
        pattern,
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ) is not None


def _find_source_value_excerpt(
    source: str,
    value: str,
    *,
    context_chars: int = 120,
    ignore_punctuation: bool = False,
) -> str | None:
    """Find a metadata value and return its original source context."""
    normalized_source, starts, ends = _normalize_with_source_map(source)
    normalized_value, _, _ = _normalize_with_source_map(value)

    if ignore_punctuation:
        source_items = [
            (char, start, end)
            for char, start, end in zip(
                normalized_source,
                starts,
                ends,
                strict=True,
            )
            if not unicodedata.category(char).startswith("P")
        ]
        value_items = [
            char
            for char in normalized_value
            if not unicodedata.category(char).startswith("P")
        ]
        normalized_source = "".join(item[0] for item in source_items)
        starts = [item[1] for item in source_items]
        ends = [item[2] for item in source_items]
        normalized_value = "".join(value_items)

    if not normalized_value:
        raise EvidenceValidationError(
            "extracted metadata value is blank after normalization"
        )

    search_from = 0
    while True:
        position = normalized_source.find(normalized_value, search_from)
        if position < 0:
            return None

        value_end = position + len(normalized_value)
        left_boundary_ok = (
            position == 0
            or not normalized_value[0].isalnum()
            or not normalized_source[position - 1].isalnum()
        )
        right_boundary_ok = (
            value_end == len(normalized_source)
            or not normalized_value[-1].isalnum()
            or not normalized_source[value_end].isalnum()
        )

        if left_boundary_ok and right_boundary_ok:
            source_start = starts[position]
            source_end = ends[value_end - 1]
            excerpt_start = max(0, source_start - context_chars)
            excerpt_end = min(len(source), source_end + context_chars)
            excerpt = source[excerpt_start:excerpt_end].strip()
            return excerpt or None

        search_from = position + 1


class EvidenceValidator:
    """Validate evidence against the exact chunks retrieved for one request."""

    def __init__(self, retrieved_chunks: Sequence[ChunkRecord]) -> None:
        self._chunks: dict[str, ChunkRecord] = {}
        for chunk in retrieved_chunks:
            if chunk.chunk_id in self._chunks:
                raise EvidenceValidationError(
                    "retrieval returned duplicate chunk IDs"
                )
            self._chunks[chunk.chunk_id] = chunk

    def verify_quote(self, *, chunk_id: str, quote: str) -> VerifiedQuote:
        """Resolve a chunk ID and verify the quote against its source text."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            raise EvidenceValidationError(
                "evidence refers to a chunk outside this retrieval result"
            )
        return VerifiedQuote(
            chunk=chunk,
            excerpt=_find_source_excerpt(chunk.text, quote),
        )

    def build_citation(self, evidence: EvidenceQuoteDraft) -> Citation:
        """Build a citation using only application-owned chunk metadata."""
        verified = self.verify_quote(
            chunk_id=evidence.chunk_id,
            quote=evidence.quote,
        )
        chunk = verified.chunk
        return Citation(
            citation_id=chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            page_number=chunk.page_number,
            excerpt=verified.excerpt,
        )

    def build_references(self) -> tuple[EvidenceReference, ...]:
        """Return cosine-scored retrieval metadata in original order."""
        references: list[EvidenceReference] = []
        for chunk in self._chunks.values():
            if not isinstance(chunk, RetrievedChunk):
                continue
            references.append(
                EvidenceReference(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    file_name=chunk.file_name,
                    page_number=chunk.page_number,
                    score=chunk.score,
                )
            )
        return tuple(references)

    def build_field_citation(
        self,
        evidence: ExtractionEvidenceDraft,
        *,
        expected_value: str,
    ) -> ExtractionFieldCitation:
        """Verify legacy field quote evidence and return a field citation."""
        if not isinstance(expected_value, str) or not expected_value.strip():
            raise EvidenceValidationError(
                "field evidence requires a non-blank extracted value"
            )

        verified = self.verify_quote(
            chunk_id=evidence.chunk_id,
            quote=evidence.quote,
        )
        normalized_quote, _, _ = _normalize_with_source_map(verified.excerpt)
        normalized_value, _, _ = _normalize_with_source_map(expected_value)
        if normalized_value not in normalized_quote:
            raise EvidenceValidationError(
                "extracted value is not present in its supporting quote"
            )

        chunk = verified.chunk
        citation = Citation(
            citation_id=chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            page_number=chunk.page_number,
            excerpt=verified.excerpt,
        )
        return ExtractionFieldCitation(
            field_name=evidence.field_name,
            citation=citation,
        )

    def build_field_citation_for_value(
        self,
        *,
        field_name: MetadataField,
        expected_value: str,
    ) -> ExtractionFieldCitation | None:
        """Find a metadata value in source chunks and construct its citation."""
        if not isinstance(expected_value, str) or not expected_value.strip():
            raise EvidenceValidationError(
                "field evidence requires a non-blank extracted value"
            )

        ordered_chunks = sorted(
            self._chunks.values(),
            key=lambda chunk: (chunk.page_number, chunk.chunk_index),
        )
        for chunk in ordered_chunks:
            if (
                field_name is MetadataField.PUBLICATION_DATE
                and _is_superseded_document_date(chunk.text, expected_value)
            ):
                continue

            excerpt = _find_source_value_excerpt(
                source=chunk.text,
                value=expected_value,
                ignore_punctuation=(
                    field_name is MetadataField.DOCUMENT_TITLE
                ),
            )
            if excerpt is None:
                continue

            citation = Citation(
                citation_id=chunk.chunk_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                file_name=chunk.file_name,
                page_number=chunk.page_number,
                excerpt=excerpt,
            )
            return ExtractionFieldCitation(
                field_name=field_name,
                citation=citation,
            )

        return None
