"""Tests for query-level retrieval metrics and latency summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.evaluation.retrieval_metrics import (
    FailureCategory,
    QueryEvaluationResult,
    RetrievalMetricsReport,
    score_query,
    summarize_evaluation,
)


@dataclass(frozen=True)
class Result:
    document_id: str
    page_number: int
    chunk_id: str
    score: float


def result(
    document_id: str,
    page_number: int,
    chunk_id: str,
    score: float,
) -> Result:
    return Result(document_id, page_number, chunk_id, score)


def score(
    query_id: str,
    *,
    relevant_document_id: str = "doc-1",
    relevant_pages: tuple[int, ...] = (2,),
    results: tuple[Result, ...] = (),
    latency_ms: float = 10.0,
    k: int = 5,
    miss_category: FailureCategory | str | None = None,
    miss_diagnostic: str | None = None,
) -> QueryEvaluationResult:
    return score_query(
        query_id=query_id,
        relevant_document_id=relevant_document_id,
        relevant_page_numbers=relevant_pages,
        retrieved_results=results,
        latency_ms=latency_ms,
        k=k,
        miss_category=miss_category,
        miss_diagnostic=miss_diagnostic,
    )


def test_relevant_document_and_page_in_top_k_is_a_hit() -> None:
    evaluated = score(
        "q-1",
        results=(
            result("doc-1", 1, "chunk-a", 0.9),
            result("doc-1", 2, "chunk-b", 0.8),
        ),
    )

    assert evaluated.hit is True
    assert evaluated.retrieved_page_numbers == (1, 2)
    assert evaluated.retrieved_chunk_ids == ("chunk-a", "chunk-b")


def test_matching_page_from_the_wrong_document_is_not_a_hit() -> None:
    evaluated = score(
        "q-1",
        results=(result("another-doc", 2, "chunk-a", 0.9),),
    )

    assert evaluated.hit is False


def test_expected_page_beyond_top_k_does_not_count_as_hit() -> None:
    evaluated = score(
        "q-1",
        results=(
            result("doc-1", 1, "chunk-a", 0.9),
            result("doc-1", 1, "chunk-b", 0.8),
            result("doc-1", 2, "chunk-c", 0.7),
        ),
        k=2,
    )

    assert evaluated.hit is False
    assert evaluated.retrieved_page_numbers == (1, 1)
    assert evaluated.retrieved_chunk_ids == ("chunk-a", "chunk-b")


def test_hit_is_query_level_even_if_multiple_chunks_match() -> None:
    evaluated = score(
        "q-1",
        results=(
            result("doc-1", 2, "chunk-a", 0.9),
            result("doc-1", 2, "chunk-b", 0.8),
        ),
    )

    summary = summarize_evaluation([evaluated])

    assert summary.hits == 1
    assert summary.query_count == 1
    assert summary.recall_at_k == 1.0


def test_to_dict_contains_per_query_evidence_and_status() -> None:
    evaluated = score(
        "q-1",
        results=(result("doc-1", 2, "chunk-a", 0.9),),
    )

    data = evaluated.to_dict()

    assert data["query_id"] == "q-1"
    assert data["relevant_page_numbers"] == [2]
    assert data["retrieved_page_numbers"] == [2]
    assert data["retrieved_chunk_ids"] == ["chunk-a"]
    assert data["hit"] is True
    assert data["status"] == "hit"
    assert data["latency_ms"] == 10.0


def test_rejects_blank_query_id_and_document_id() -> None:
    with pytest.raises(ValueError, match="query_id"):
        score("  ")

    with pytest.raises(ValueError, match="relevant_document_id"):
        score("q-1", relevant_document_id="  ")


@pytest.mark.parametrize(
    ("pages", "error_type"),
    [
        ((), ValueError),
        ((0,), ValueError),
        ((-1,), ValueError),
        ((True,), ValueError),
        ((1.5,), ValueError),
        ((2, 2), ValueError),
        ("2", TypeError),
    ],
)
def test_rejects_invalid_relevant_page_labels(
    pages: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(
        error_type,
        match="relevant_page_numbers|relevant page",
    ):
        score("q-1", relevant_pages=pages)


@pytest.mark.parametrize(
    ("latency", "error_type"),
    [
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("10", TypeError),
    ],
)
def test_rejects_invalid_latency(
    latency: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="latency_ms"):
        score("q-1", latency_ms=latency)


@pytest.mark.parametrize(
    ("k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_rejects_invalid_k(
    k: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="k"):
        score("q-1", k=k)


def test_rejects_duplicate_retrieved_chunk_ids() -> None:
    duplicate = result("doc-1", 2, "chunk-1", 0.9)

    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        score("q-1", results=(duplicate, duplicate))


def test_rejects_scores_not_in_descending_order() -> None:
    with pytest.raises(ValueError, match="descending score"):
        score(
            "q-1",
            results=(
                result("doc-1", 1, "chunk-a", 0.7),
                result("doc-1", 2, "chunk-b", 0.8),
            ),
        )


@pytest.mark.parametrize(
    ("retrieved_score", "error_type"),
    [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (1.5, ValueError),
        (True, TypeError),
    ],
)
def test_rejects_invalid_retrieved_scores(
    retrieved_score: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="score|scores"):
        score(
            "q-1",
            results=(result("doc-1", 2, "chunk-a", retrieved_score),),
        )


def test_rejects_duplicate_query_ids_in_summary() -> None:
    first = score("q-1")
    second = score("q-1", latency_ms=20.0)

    with pytest.raises(ValueError, match="duplicate query IDs"):
        summarize_evaluation([first, second])


def test_rejects_empty_summary_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        summarize_evaluation([])


def test_recall_latency_and_nearest_rank_p95_are_calculated() -> None:
    evaluated = [
        score(
            "q-1",
            results=(result("doc-1", 2, "chunk-1", 0.9),),
            latency_ms=10.0,
        ),
        score(
            "q-2",
            results=(result("doc-1", 1, "chunk-2", 0.8),),
            latency_ms=20.0,
            miss_category=FailureCategory.PARSE,
            miss_diagnostic=(
                "Expected passage was absent from extracted page text."
            ),
        ),
        score(
            "q-3",
            results=(result("doc-1", 2, "chunk-3", 0.7),),
            latency_ms=30.0,
        ),
        score(
            "q-4",
            results=(result("doc-2", 2, "chunk-4", 0.6),),
            latency_ms=40.0,
            miss_category="metadata",
            miss_diagnostic="Top result belongs to a different document.",
        ),
    ]

    summary = summarize_evaluation(evaluated)

    assert isinstance(summary, RetrievalMetricsReport)
    assert summary.k == 5
    assert summary.query_count == 4
    assert summary.hits == 2
    assert summary.misses == 2
    assert summary.recall_at_k == pytest.approx(0.5)
    assert summary.recall_at_k_percent == pytest.approx(50.0)
    assert summary.median_latency_ms == pytest.approx(25.0)
    assert summary.p95_latency_ms == pytest.approx(40.0)
    assert summary.miss_classification_complete is True
    assert summary.unclassified_miss_ids == ()


def test_reports_unclassified_misses_until_inspected() -> None:
    miss = score(
        "q-miss",
        results=(result("doc-1", 1, "chunk-a", 0.9),),
    )

    summary = summarize_evaluation([miss])

    assert summary.miss_classification_complete is False
    assert summary.unclassified_miss_ids == ("q-miss",)
    assert summary.to_dict()["miss_classification_complete"] is False


def test_final_summary_can_require_miss_classification() -> None:
    miss = score(
        "q-miss",
        results=(result("doc-1", 1, "chunk-a", 0.9),),
    )

    with pytest.raises(ValueError, match="category and diagnostic"):
        summarize_evaluation([miss], require_miss_annotations=True)


def test_rejects_blank_miss_diagnostic() -> None:
    with pytest.raises(ValueError, match="miss_diagnostic"):
        score(
            "q-miss",
            results=(result("doc-1", 1, "chunk-a", 0.9),),
            miss_category=FailureCategory.EMBEDDING,
            miss_diagnostic="   ",
        )


def test_rejects_unknown_failure_category() -> None:
    with pytest.raises(ValueError, match="miss_category"):
        score(
            "q-miss",
            results=(result("doc-1", 1, "chunk-a", 0.9),),
            miss_category="not-a-category",
            miss_diagnostic="An example diagnostic.",
        )


def test_rejects_miss_annotations_for_a_hit() -> None:
    with pytest.raises(ValueError, match="only be supplied for a miss"):
        score(
            "q-hit",
            results=(result("doc-1", 2, "chunk-a", 0.9),),
            miss_category=FailureCategory.EMBEDDING,
            miss_diagnostic="Should not be attached to a hit.",
        )


def test_rejects_mixed_k_values_in_summary() -> None:
    first = score("q-1", k=5)
    second = score("q-2", k=3)

    with pytest.raises(ValueError, match="same k"):
        summarize_evaluation([first, second])


def test_p95_uses_nearest_rank_for_thirty_queries() -> None:
    evaluated = [
        score(f"q-{index:02d}", latency_ms=float(index))
        for index in range(1, 31)
    ]

    summary = summarize_evaluation(evaluated)

    # Nearest-rank p95 is ceil(0.95 * 30) = rank 29.
    assert summary.p95_latency_ms == 29.0