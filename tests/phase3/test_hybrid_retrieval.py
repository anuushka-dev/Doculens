"""Tests for BM25 lexical retrieval and Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from app.schemas.retrieval import ChunkRecord, RetrievedChunk
from app.services.hybrid_retrieval import HybridRetrievalPipeline


class FakeEmbedder:
    """Minimal embedding fake that avoids loading a real model."""

    model_name = "test-embedding-model"
    revision = None

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        return [[1.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        return [1.0]


class FakeVectorStore:
    """Controlled vector-store fake with deterministic dense ranking."""

    embedding_model = "test-embedding-model"
    embedding_revision = None

    def __init__(self, chunks: tuple[RetrievedChunk, ...]) -> None:
        self._chunks = chunks
        self.records = tuple(
            ChunkRecord.model_validate(
                chunk.model_dump(exclude={"score"})
            )
            for chunk in chunks
        )

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query_embedding: object,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        return list(self._chunks[:top_k])


def make_chunk(
    *,
    chunk_id: str,
    text: str,
    page_number: int,
    score: float,
) -> RetrievedChunk:
    """Build a source chunk with valid character offsets."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-001",
        file_name="technical-guide.pdf",
        page_number=page_number,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        score=score,
    )


def make_pipeline(
    *,
    chunks: tuple[RetrievedChunk, ...] | None = None,
    top_k: int = 5,
    dense_candidate_k: int = 20,
    lexical_candidate_k: int = 20,
    rrf_k: int = 60,
) -> HybridRetrievalPipeline:
    """Build a hybrid pipeline with deterministic fake dependencies."""
    selected_chunks = chunks or (
        make_chunk(
            chunk_id="chunk-a",
            text="needle",
            page_number=1,
            score=0.92,
        ),
        make_chunk(
            chunk_id="chunk-b",
            text="dense only material",
            page_number=2,
            score=0.81,
        ),
        make_chunk(
            chunk_id="chunk-c",
            text="needle needle needle",
            page_number=3,
            score=0.73,
        ),
    )

    return HybridRetrievalPipeline(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(selected_chunks),
        top_k=top_k,
        dense_candidate_k=dense_candidate_k,
        lexical_candidate_k=lexical_candidate_k,
        rrf_k=rrf_k,
    )


class TestHybridRetrievalConfiguration:
    def test_default_candidate_configuration(self):
        pipeline = make_pipeline()

        assert pipeline.dense_candidate_k == 20
        assert pipeline.lexical_candidate_k == 20
        assert pipeline.rrf_k == 60

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dense_candidate_k", 0),
            ("lexical_candidate_k", 0),
            ("rrf_k", 0),
            ("dense_candidate_k", True),
            ("lexical_candidate_k", True),
            ("rrf_k", True),
        ],
    )
    def test_candidate_configuration_rejects_invalid_values(
        self,
        field: str,
        value: object,
    ):
        kwargs = {
            "dense_candidate_k": 20,
            "lexical_candidate_k": 20,
            "rrf_k": 60,
        }
        kwargs[field] = value

        with pytest.raises((TypeError, ValueError)):
            make_pipeline(**kwargs)


class TestBM25Ranking:
    def test_lexical_ranking_prefers_repeated_query_term(self):
        pipeline = make_pipeline()

        ranked = pipeline._rank_bm25("needle")

        ranked_chunk_ids = [
            pipeline.vector_store.records[index].chunk_id
            for _, index in ranked
        ]

        assert ranked_chunk_ids == ["chunk-c", "chunk-a"]

    def test_lexical_search_returns_no_results_for_unmatched_terms(self):
        pipeline = make_pipeline()

        assert pipeline._rank_bm25("unmatchedterm") == []

    def test_lexical_search_is_case_insensitive(self):
        pipeline = make_pipeline()

        lowercase = pipeline._rank_bm25("needle")
        uppercase = pipeline._rank_bm25("NEEDLE")

        assert lowercase == uppercase


class TestReciprocalRankFusion:
    def test_fusion_includes_candidates_from_both_retrievers(self):
        pipeline = make_pipeline(
            dense_candidate_k=2,
            lexical_candidate_k=2,
            top_k=3,
        )

        result = pipeline.search("needle")

        assert [
            chunk.chunk_id for chunk in result.results
        ] == ["chunk-a", "chunk-c", "chunk-b"]

        # chunk-c was outside dense top two but entered through BM25.
        lexical_only_candidate = result.results[1]

        assert lexical_only_candidate.chunk_id == "chunk-c"
        assert lexical_only_candidate.dense_rank is None
        assert lexical_only_candidate.bm25_rank == 1

        # chunk-b entered through dense retrieval, not BM25.
        dense_only_candidate = result.results[2]

        assert dense_only_candidate.chunk_id == "chunk-b"
        assert dense_only_candidate.dense_rank == 2
        assert dense_only_candidate.bm25_rank is None

    def test_cosine_scores_are_preserved_separately_from_rrf(self):
        pipeline = make_pipeline(
            dense_candidate_k=2,
            lexical_candidate_k=2,
            top_k=3,
        )

        result = pipeline.search("needle")

        scores_by_id = {
            chunk.chunk_id: chunk.score
            for chunk in result.results
        }

        assert scores_by_id == {
            "chunk-a": 0.92,
            "chunk-c": 0.73,
            "chunk-b": 0.81,
        }

        assert all(
            chunk.rrf_score > 0.0
            for chunk in result.results
        )

        assert all(
            chunk.score != chunk.rrf_score
            for chunk in result.results
        )

    def test_fusion_ranks_are_consecutive_and_descending(self):
        pipeline = make_pipeline(
            dense_candidate_k=2,
            lexical_candidate_k=2,
            top_k=3,
        )

        result = pipeline.search("needle")

        assert [
            chunk.fusion_rank for chunk in result.results
        ] == [1, 2, 3]

        assert (
            result.results[0].rrf_score
            > result.results[1].rrf_score
            > result.results[2].rrf_score
        )

    def test_result_count_respects_requested_top_k(self):
        pipeline = make_pipeline(
            dense_candidate_k=2,
            lexical_candidate_k=2,
            top_k=3,
        )

        result = pipeline.search("needle", top_k=2)

        assert result.top_k == 2
        assert len(result.results) == 2

    def test_blank_query_is_rejected(self):
        pipeline = make_pipeline()

        with pytest.raises(ValueError, match="query must not be blank"):
            pipeline.search("   ")

    def test_source_metadata_survives_fusion(self):
        pipeline = make_pipeline(
            dense_candidate_k=2,
            lexical_candidate_k=2,
            top_k=3,
        )

        result = pipeline.search("needle")
        chunk = result.results[1]

        assert chunk.chunk_id == "chunk-c"
        assert chunk.document_id == "doc-001"
        assert chunk.file_name == "technical-guide.pdf"
        assert chunk.page_number == 3
        assert chunk.text == "needle needle needle"
        assert chunk.start_char == 0
        assert chunk.end_char == len(chunk.text)
