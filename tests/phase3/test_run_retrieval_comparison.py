"""Tests for weighted fusion and page-level retrieval comparison metrics."""

from __future__ import annotations

from app.schemas.retrieval import ChunkRecord, RetrievedChunk
from scripts.phase3.run_retrieval_comparison import (
    RankedCandidate,
    fuse_rankings,
    score_ranking,
)


def _chunk(chunk_id: str, page: int, cosine: float) -> RetrievedChunk:
    text = f"source text for {chunk_id}"
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        file_name="source.pdf",
        page_number=page,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        score=cosine,
    )


def _record(chunk: RetrievedChunk) -> ChunkRecord:
    return ChunkRecord.model_validate(chunk.model_dump(exclude={"score"}))


def test_equal_rrf_combines_votes_and_preserves_cosine_score() -> None:
    dense = [_chunk("A", 1, 0.95), _chunk("B", 2, 0.80)]
    lexical_records = [_record(_chunk("B", 2, 0.80)), _record(_chunk("C", 3, 0.70))]
    all_cosine = {item.chunk_id: item for item in [*dense, _chunk("C", 3, 0.70)]}

    fused = fuse_rankings(
        dense_candidates=dense,
        bm25_candidates=[(9.0, 0), (8.0, 1)],
        cosine_by_id=all_cosine,
        records=lexical_records,
        rrf_k=60,
    )

    assert fused[0].chunk.chunk_id == "B"
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[0].chunk.score == 0.80
    assert fused[0].rrf_score is not None


def test_weighted_rrf_can_increase_lexical_only_candidate_rank() -> None:
    dense = [_chunk("A", 1, 0.95), _chunk("B", 2, 0.80)]
    lexical_records = [_record(_chunk("C", 3, 0.70)), _record(_chunk("B", 2, 0.80))]
    all_cosine = {item.chunk_id: item for item in [*dense, _chunk("C", 3, 0.70)]}

    equal = fuse_rankings(
        dense_candidates=dense,
        bm25_candidates=[(9.0, 0), (8.0, 1)],
        cosine_by_id=all_cosine,
        records=lexical_records,
        rrf_k=60,
    )
    lexical_weighted = fuse_rankings(
        dense_candidates=dense,
        bm25_candidates=[(9.0, 0), (8.0, 1)],
        cosine_by_id=all_cosine,
        records=lexical_records,
        rrf_k=60,
        dense_weight=0.8,
        bm25_weight=1.2,
    )

    equal_rank_c = next(
        i for i, item in enumerate(equal, start=1) if item.chunk.chunk_id == "C"
    )
    weighted_rank_c = next(
        i
        for i, item in enumerate(lexical_weighted, start=1)
        if item.chunk.chunk_id == "C"
    )
    assert weighted_rank_c < equal_rank_c
    assert lexical_weighted[0].chunk.score == 0.80


def test_page_recall_counts_unique_pages_not_duplicate_chunks() -> None:
    first = _chunk("A", 1, 0.95)
    second = _chunk("B", 1, 0.90)
    third = _chunk("C", 3, 0.85)
    ranked = [
        RankedCandidate(first, 1, None, 0.0, None),
        RankedCandidate(second, 2, None, 0.0, None),
        RankedCandidate(third, 3, None, 0.0, None),
    ]
    report = score_ranking(
        [
            {
                "relevant_page_numbers": [1, 2],
                "benchmark_document_id": "doc-1",
                "ranked": ranked,
            }
        ],
        top_k_values=(2, 3),
    )

    assert report["page_recall_at_2"] == 0.5
    assert report["query_hit_rate_at_2"] == 1.0
    assert report["mrr_at_2"] == 1.0
    assert report["page_recall_at_3"] == 0.5


def test_mrr_at_k_is_zero_when_first_relevant_result_is_outside_k() -> None:
    ranked = [
        RankedCandidate(_chunk("X", 9, 0.99), 1, None, 0.0, None),
        RankedCandidate(_chunk("Y", 8, 0.98), 2, None, 0.0, None),
        RankedCandidate(_chunk("Z", 2, 0.97), 3, None, 0.0, None),
    ]
    report = score_ranking(
        [
            {
                "relevant_page_numbers": [2],
                "benchmark_document_id": "doc-1",
                "ranked": ranked,
            }
        ],
        top_k_values=(2, 3),
    )

    assert report["mrr_at_2"] == 0.0
    assert report["mrr_at_3"] == round(1 / 3, 6)
