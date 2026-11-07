"""Hybrid dense and lexical retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from collections import Counter

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.adapters.vectorstore.faiss_store import FaissVectorStore
from app.schemas.retrieval import RetrievedChunk
from app.services.retrieval_pipeline import (
    EmbeddingAdapter,
    RetrievalPipeline,
)

_TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")

BM25_K1 = 1.5
BM25_B = 0.75


def _positive_integer(value: object, *, field_name: str) -> int:
    """Validate a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")

    if value < 1:
        raise ValueError(f"{field_name} must be positive")

    return value


def _tokenize(text: str) -> tuple[str, ...]:
    """Create case-folded Unicode word tokens."""
    return tuple(
        token.casefold()
        for token in _TOKEN_PATTERN.findall(text)
    )


class HybridRetrievedChunk(RetrievedChunk):
    """Retrieved source chunk with separate dense, lexical, and fusion data.

    ``score`` is inherited from RetrievedChunk and remains cosine similarity.
    It is not replaced by the BM25 or RRF score.
    """

    dense_rank: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Rank in the dense candidate list, if selected.",
    )
    bm25_rank: int | None = Field(
        default=None,
        ge=1,
        strict=True,
        description="Rank in the lexical candidate list, if selected.",
    )
    bm25_score: float = Field(
        ge=0.0,
        allow_inf_nan=False,
        strict=True,
        description="BM25 lexical relevance score.",
    )
    rrf_score: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        strict=True,
        description="Reciprocal Rank Fusion score.",
    )
    fusion_rank: int = Field(
        ge=1,
        strict=True,
        description="Final one-based rank after RRF.",
    )


class HybridRetrievalResponse(BaseModel):
    """Hybrid retrieval results ordered by fusion rank, not cosine score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, strict=True)
    top_k: int = Field(ge=1, strict=True)
    dense_candidate_k: int = Field(ge=1, strict=True)
    lexical_candidate_k: int = Field(ge=1, strict=True)
    rrf_k: int = Field(ge=1, strict=True)
    results: tuple[HybridRetrievedChunk, ...]

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        """Reject whitespace-only queries."""
        if not value.strip():
            raise ValueError("query must not be blank")

        return value

    @model_validator(mode="after")
    def validate_fusion_order(self) -> HybridRetrievalResponse:
        """Validate result count, uniqueness, and final fusion ordering."""
        if len(self.results) > self.top_k:
            raise ValueError("results cannot contain more than top_k entries")

        chunk_ids = [chunk.chunk_id for chunk in self.results]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("results must have unique chunk IDs")

        expected_ranks = list(range(1, len(self.results) + 1))
        actual_ranks = [chunk.fusion_rank for chunk in self.results]

        if actual_ranks != expected_ranks:
            raise ValueError("fusion_rank values must be consecutive")

        for previous, current in zip(
            self.results,
            self.results[1:],
            strict=False,
        ):
            if previous.rrf_score < current.rrf_score:
                raise ValueError(
                    "results must be ordered by descending RRF score"
                )

        return self


class HybridRetrievalPipeline(RetrievalPipeline):
    """Combine dense FAISS retrieval and BM25 with deterministic RRF.

    The inherited indexing path remains unchanged:
    page chunking -> sentence-transformer embeddings -> FAISS.

    Search retrieves dense and lexical candidate lists independently,
    fuses their ranks, and returns the highest-ranked unique chunks.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingAdapter | None = None,
        vector_store: FaissVectorStore | None = None,
        chunk_size_words: int = 220,
        overlap_words: int = 40,
        top_k: int = 5,
        dense_candidate_k: int = 20,
        lexical_candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.dense_candidate_k = _positive_integer(
            dense_candidate_k,
            field_name="dense_candidate_k",
        )
        self.lexical_candidate_k = _positive_integer(
            lexical_candidate_k,
            field_name="lexical_candidate_k",
        )
        self.rrf_k = _positive_integer(
            rrf_k,
            field_name="rrf_k",
        )

        super().__init__(
            embedder=embedder,
            vector_store=vector_store,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            top_k=top_k,
        )

    def _rank_bm25(
        self,
        query: str,
    ) -> list[tuple[float, int]]:
        """Return (BM25 score, record index) pairs in deterministic order."""
        records = self.vector_store.records

        if not records:
            return []

        query_terms = set(_tokenize(query))

        if not query_terms:
            return []

        tokenized_documents = [
            _tokenize(record.text)
            for record in records
        ]

        document_count = len(tokenized_documents)
        document_lengths = [
            len(tokens)
            for tokens in tokenized_documents
        ]

        average_length = (
            sum(document_lengths) / document_count
            if document_count
            else 0.0
        )

        if average_length == 0.0:
            return []

        document_frequency: Counter[str] = Counter()

        for tokens in tokenized_documents:
            document_frequency.update(set(tokens))

        ranked: list[tuple[float, int]] = []

        for index, tokens in enumerate(tokenized_documents):
            term_frequencies = Counter(tokens)
            document_length = document_lengths[index]
            score = 0.0

            for term in sorted(query_terms):
                term_frequency = term_frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                frequency = document_frequency[term]

                inverse_document_frequency = math.log1p(
                    (document_count - frequency + 0.5)
                    / (frequency + 0.5)
                )

                length_normalization = (
                    BM25_K1
                    * (
                        1.0
                        - BM25_B
                        + BM25_B * document_length / average_length
                    )
                )

                term_score = (
                    inverse_document_frequency
                    * term_frequency
                    * (BM25_K1 + 1.0)
                    / (term_frequency + length_normalization)
                )

                score += term_score

            if score > 0.0:
                ranked.append((score, index))

        ranked.sort(
            key=lambda item: (
                -item[0],
                records[item[1]].chunk_id,
            )
        )

        return ranked

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> HybridRetrievalResponse:
        """Retrieve dense and BM25 candidates, then fuse their ranks."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be blank")

        result_count = (
            self.top_k
            if top_k is None
            else _positive_integer(top_k, field_name="top_k")
        )

        if self.vector_store.size == 0:
            raise ValueError("cannot search an empty retrieval index")

        query_vector = self.embedder.embed_query(query)

        # Search all indexed chunks so BM25-only candidates also retain
        # their actual cosine score. This is an exact FAISS search.
        dense_all = self.vector_store.search(
            query_vector,
            top_k=self.vector_store.size,
        )

        dense_candidates = dense_all[: self.dense_candidate_k]
        lexical_candidates = self._rank_bm25(query)[
            : self.lexical_candidate_k
        ]

        cosine_by_id = {
            chunk.chunk_id: chunk
            for chunk in dense_all
        }

        bm25_ranked = [
            (
                score,
                self.vector_store.records[record_index].chunk_id,
            )
            for score, record_index in lexical_candidates
        ]

        bm25_score_by_id = {
            chunk_id: score
            for score, chunk_id in bm25_ranked
        }

        dense_rank_by_id = {
            chunk.chunk_id: rank
            for rank, chunk in enumerate(dense_candidates, start=1)
        }

        bm25_rank_by_id = {
            chunk_id: rank
            for rank, (_, chunk_id) in enumerate(
                bm25_ranked,
                start=1,
            )
        }

        fusion_scores: dict[str, float] = {}

        for rank, chunk in enumerate(dense_candidates, start=1):
            fusion_scores[chunk.chunk_id] = (
                fusion_scores.get(chunk.chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        for rank, (_, chunk_id) in enumerate(
            bm25_ranked,
            start=1,
        ):
            fusion_scores[chunk_id] = (
                fusion_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        ranked_ids = sorted(
            fusion_scores,
            key=lambda chunk_id: (
                -fusion_scores[chunk_id],
                -cosine_by_id[chunk_id].score,
                chunk_id,
            ),
        )

        hybrid_results: list[HybridRetrievedChunk] = []

        for fusion_rank, chunk_id in enumerate(
            ranked_ids[:result_count],
            start=1,
        ):
            source_chunk = cosine_by_id[chunk_id]

            result_data = source_chunk.model_dump()
            result_data.update(
                {
                    "dense_rank": dense_rank_by_id.get(chunk_id),
                    "bm25_rank": bm25_rank_by_id.get(chunk_id),
                    "bm25_score": bm25_score_by_id.get(chunk_id, 0.0),
                    "rrf_score": fusion_scores[chunk_id],
                    "fusion_rank": fusion_rank,
                }
            )

            hybrid_results.append(
                HybridRetrievedChunk.model_validate(result_data)
            )

        return HybridRetrievalResponse(
            query=query,
            top_k=result_count,
            dense_candidate_k=self.dense_candidate_k,
            lexical_candidate_k=self.lexical_candidate_k,
            rrf_k=self.rrf_k,
            results=tuple(hybrid_results),
        )
