"""Calculate query-level retrieval metrics and latency summaries."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Protocol

COSINE_SCORE_TOLERANCE = 1e-4
PERCENTILE = 0.95


class FailureCategory(StrEnum):
    """Allowed diagnostic categories from the Phase 2 evaluation protocol."""

    PARSE = "parse"
    CHUNK_BOUNDARY = "chunk_boundary"
    EMBEDDING = "embedding"
    METADATA = "metadata"
    QUERY_MISMATCH = "query_mismatch"
    INDEX_RUNTIME = "index_runtime"


class RetrievedItem(Protocol):
    """Metadata and score fields needed from a ranked retrieval result."""

    document_id: str
    page_number: int
    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class QueryEvaluationResult:
    """Evaluation evidence for one labelled query."""

    query_id: str
    relevant_document_id: str
    relevant_page_numbers: tuple[int, ...]
    retrieved_document_ids: tuple[str, ...]
    retrieved_page_numbers: tuple[int, ...]
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_scores: tuple[float, ...]
    hit: bool
    latency_ms: float
    k: int
    miss_category: FailureCategory | None = None
    miss_diagnostic: str | None = None

    @property
    def miss_is_classified(self) -> bool:
        """Return whether a miss has both a category and a useful diagnosis."""
        if self.hit:
            return True

        return (
            self.miss_category is not None
            and self.miss_diagnostic is not None
            and bool(self.miss_diagnostic.strip())
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable per-query record."""
        return {
            "query_id": self.query_id,
            "relevant_document_id": self.relevant_document_id,
            "relevant_page_numbers": list(self.relevant_page_numbers),
            "retrieved_document_ids": list(self.retrieved_document_ids),
            "retrieved_page_numbers": list(self.retrieved_page_numbers),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "retrieved_scores": list(self.retrieved_scores),
            "hit": self.hit,
            "status": "hit" if self.hit else "miss",
            "latency_ms": self.latency_ms,
            "k": self.k,
            "miss_category": (
                self.miss_category.value
                if self.miss_category is not None
                else None
            ),
            "miss_diagnostic": self.miss_diagnostic,
        }


@dataclass(frozen=True, slots=True)
class RetrievalMetricsReport:
    """Aggregate retrieval quality, latency, and per-query evidence."""

    k: int
    query_count: int
    hits: int
    misses: int
    recall_at_k: float
    recall_at_k_percent: float
    median_latency_ms: float
    p95_latency_ms: float
    unclassified_miss_ids: tuple[str, ...]
    per_query: tuple[QueryEvaluationResult, ...]

    @property
    def miss_classification_complete(self) -> bool:
        """Whether each miss has a category and diagnostic."""
        return not self.unclassified_miss_ids

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report without losing per-query detail."""
        return {
            "k": self.k,
            "query_count": self.query_count,
            "hits": self.hits,
            "misses": self.misses,
            "recall_at_k": self.recall_at_k,
            "recall_at_k_percent": self.recall_at_k_percent,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "miss_classification_complete": self.miss_classification_complete,
            "unclassified_miss_ids": list(self.unclassified_miss_ids),
            "per_query": [result.to_dict() for result in self.per_query],
        }


def _positive_integer(value: object, *, field_name: str) -> int:
    """Validate a positive integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")

    if value < 1:
        raise ValueError(f"{field_name} must be positive")

    return value


def _nonblank_string(value: object, *, field_name: str) -> str:
    """Validate a non-blank string without changing its stored spelling."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")

    return value


def _validate_relevant_pages(
    relevant_page_numbers: Sequence[int],
) -> tuple[int, ...]:
    """Return unique, positive, sorted evidence-page labels."""
    if isinstance(relevant_page_numbers, (str, bytes)):
        raise TypeError("relevant_page_numbers must be a sequence of integers")

    if not relevant_page_numbers:
        raise ValueError("relevant_page_numbers must contain at least one page")

    pages: list[int] = []

    for page_number in relevant_page_numbers:
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise ValueError("relevant page numbers must be positive integers")

        pages.append(page_number)

    if len(pages) != len(set(pages)):
        raise ValueError("relevant_page_numbers must not contain duplicates")

    return tuple(sorted(pages))


def _validate_latency(latency_ms: object) -> float:
    """Validate a finite, non-negative query latency in milliseconds."""
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, Real):
        raise TypeError("latency_ms must be a real number")

    latency = float(latency_ms)

    if not math.isfinite(latency) or latency < 0.0:
        raise ValueError("latency_ms must be finite and non-negative")

    return latency


def _validate_retrieved_results(
    retrieved_results: Sequence[RetrievedItem],
) -> list[RetrievedItem]:
    """Validate provenance, unique chunk IDs, finite scores, and rank order."""
    if isinstance(retrieved_results, (str, bytes)):
        raise TypeError("retrieved_results must be a sequence of ranked results")

    results = list(retrieved_results)
    seen_chunk_ids: set[str] = set()
    previous_score: float | None = None

    for index, result in enumerate(results):
        document_id = _nonblank_string(
            getattr(result, "document_id", None),
            field_name=f"retrieved_results[{index}].document_id",
        )
        chunk_id = _nonblank_string(
            getattr(result, "chunk_id", None),
            field_name=f"retrieved_results[{index}].chunk_id",
        )
        page_number = getattr(result, "page_number", None)
        score = getattr(result, "score", None)

        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise ValueError(
                f"retrieved_results[{index}].page_number must be positive"
            )

        if chunk_id in seen_chunk_ids:
            raise ValueError("retrieved results contain duplicate chunk IDs")

        seen_chunk_ids.add(chunk_id)

        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(f"retrieved_results[{index}].score must be numeric")

        numeric_score = float(score)

        if not math.isfinite(numeric_score):
            raise ValueError("retrieved scores must be finite")

        if (
            numeric_score < -1.0 - COSINE_SCORE_TOLERANCE
            or numeric_score > 1.0 + COSINE_SCORE_TOLERANCE
        ):
            raise ValueError("retrieved scores must be cosine similarities")

        if previous_score is not None and numeric_score > previous_score:
            raise ValueError(
                "retrieved results must be ranked by descending score"
            )

        previous_score = numeric_score

        # Read the fields here so a malformed duck-typed object fails early.
        _ = document_id, page_number

    return results


def score_query(
    *,
    query_id: str,
    relevant_document_id: str,
    relevant_page_numbers: Sequence[int],
    retrieved_results: Sequence[RetrievedItem],
    latency_ms: float,
    k: int = 5,
    miss_category: FailureCategory | str | None = None,
    miss_diagnostic: str | None = None,
) -> QueryEvaluationResult:
    """Score one query as a hit if top-k contains a labelled document/page.

    A query is a hit when at least one of its top-k chunks comes from the
    expected document and one of the expected evidence pages.

    Latency should include query embedding and FAISS search, but exclude PDF
    parsing and corpus indexing/embedding.
    """
    selected_query_id = _nonblank_string(query_id, field_name="query_id")
    selected_document_id = _nonblank_string(
        relevant_document_id,
        field_name="relevant_document_id",
    )
    selected_k = _positive_integer(k, field_name="k")
    selected_latency = _validate_latency(latency_ms)
    relevant_pages = _validate_relevant_pages(relevant_page_numbers)
    results = _validate_retrieved_results(retrieved_results)
    top_results = results[:selected_k]

    if miss_category is None:
        selected_category = None
    else:
        try:
            selected_category = FailureCategory(miss_category)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(category.value for category in FailureCategory)
            raise ValueError(
                f"miss_category must be one of: {allowed}"
            ) from exc

    if miss_diagnostic is not None:
        _nonblank_string(miss_diagnostic, field_name="miss_diagnostic")

    hit = any(
        result.document_id == selected_document_id
        and result.page_number in relevant_pages
        for result in top_results
    )

    if hit and (selected_category is not None or miss_diagnostic is not None):
        raise ValueError("miss annotations may only be supplied for a miss")

    retrieved_document_ids = tuple(
        str(result.document_id) for result in top_results
    )
    retrieved_page_numbers = tuple(
        int(result.page_number) for result in top_results
    )
    retrieved_chunk_ids = tuple(
        str(result.chunk_id) for result in top_results
    )
    retrieved_scores = tuple(
        float(result.score) for result in top_results
    )

    return QueryEvaluationResult(
        query_id=selected_query_id,
        relevant_document_id=selected_document_id,
        relevant_page_numbers=relevant_pages,
        retrieved_document_ids=retrieved_document_ids,
        retrieved_page_numbers=retrieved_page_numbers,
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_scores=retrieved_scores,
        hit=hit,
        latency_ms=selected_latency,
        k=selected_k,
        miss_category=selected_category,
        miss_diagnostic=miss_diagnostic.strip() if miss_diagnostic else None,
    )


def summarize_evaluation(
    query_results: Sequence[QueryEvaluationResult],
    *,
    require_miss_annotations: bool = False,
) -> RetrievalMetricsReport:
    """Summarize recall and latency using the nearest-rank p95 percentile.

    By default, incomplete miss classifications are reported in the returned
    summary so diagnostics can be filled in after inspecting the retrieved
    evidence. Set require_miss_annotations=True for final acceptance reporting.
    """
    if isinstance(query_results, (str, bytes)):
        raise TypeError("query_results must be a sequence of QueryEvaluationResult")

    if not isinstance(require_miss_annotations, bool):
        raise TypeError("require_miss_annotations must be a boolean")

    results = tuple(query_results)

    if not results:
        raise ValueError("query_results must contain at least one labelled query")

    if not all(isinstance(result, QueryEvaluationResult) for result in results):
        raise TypeError("all entries must be QueryEvaluationResult instances")

    query_ids = [result.query_id for result in results]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query_results must not contain duplicate query IDs")

    k_values = {result.k for result in results}

    if len(k_values) != 1:
        raise ValueError("all query results must use the same k")

    selected_k = next(iter(k_values))
    hits = sum(result.hit for result in results)
    misses = len(results) - hits
    latencies = sorted(result.latency_ms for result in results)
    p95_rank = max(1, math.ceil(PERCENTILE * len(latencies)))
    p95_latency = latencies[p95_rank - 1]

    unclassified_miss_ids = tuple(
        result.query_id
        for result in results
        if not result.hit and not result.miss_is_classified
    )

    if require_miss_annotations and unclassified_miss_ids:
        ids = ", ".join(unclassified_miss_ids)
        raise ValueError(f"misses need a category and diagnostic: {ids}")

    recall = hits / len(results)

    return RetrievalMetricsReport(
        k=selected_k,
        query_count=len(results),
        hits=hits,
        misses=misses,
        recall_at_k=recall,
        recall_at_k_percent=recall * 100.0,
        median_latency_ms=float(_median(latencies)),
        p95_latency_ms=float(p95_latency),
        unclassified_miss_ids=unclassified_miss_ids,
        per_query=results,
    )


def _median(sorted_values: Sequence[float]) -> float:
    """Calculate median for non-empty sorted numeric values."""
    count = len(sorted_values)
    midpoint = count // 2

    if count % 2:
        return float(sorted_values[midpoint])

    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0