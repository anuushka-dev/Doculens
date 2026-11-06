"""Deterministic aggregate metrics for grounded Q&A evaluation."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QAEvaluationObservation:
    """Metric inputs for one evaluated question.

    ``expected_answerable`` and ``actual_answered`` drive the answerability
    confusion matrix. Page citations are evaluated only for answerable cases,
    matching the Phase 3 report definition. Latency is measured in milliseconds.
    """

    expected_answerable: bool
    actual_answered: bool
    expected_pages: tuple[int, ...]
    cited_pages: tuple[int, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.expected_answerable, bool):
            raise TypeError("expected_answerable must be a boolean")
        if not isinstance(self.actual_answered, bool):
            raise TypeError("actual_answered must be a boolean")

        if self.expected_answerable and not self.expected_pages:
            raise ValueError("answerable observations require expected pages")
        if not self.expected_answerable and self.expected_pages:
            raise ValueError("unanswerable observations must not have expected pages")

        for page in (*self.expected_pages, *self.cited_pages):
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                raise ValueError("page numbers must be positive integers")

        if isinstance(self.latency_ms, bool) or not isinstance(
            self.latency_ms, (int, float)
        ):
            raise TypeError("latency_ms must be a number")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")


def _rate(numerator: int, denominator: int) -> float | None:
    """Calculate a rate without inventing a value for an empty group."""

    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _percentile_95(values: Sequence[float]) -> float | None:
    """Calculate the nearest-rank 95th percentile."""

    if not values:
        return None

    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 3)


def calculate_qa_metrics(
    observations: Sequence[QAEvaluationObservation],
) -> dict[str, object]:
    """Aggregate the Phase 3 grounded-Q&A metrics.

    Citation-page accuracy counts citations from answerable cases only. A page
    is correct when it appears in that case's expected source-page set. Citation
    coverage counts answerable cases that returned at least one citation.
    Empty metric denominators yield ``None`` rather than a misleading zero.
    """

    true_answered = 0
    true_abstained = 0
    false_answered = 0
    false_abstained = 0

    citation_pages_correct = 0
    citation_pages_total = 0
    answerable_cases_with_citations = 0
    latencies: list[float] = []

    for observation in observations:
        latencies.append(float(observation.latency_ms))

        if observation.expected_answerable and observation.actual_answered:
            true_answered += 1
        elif observation.expected_answerable and not observation.actual_answered:
            false_abstained += 1
        elif not observation.expected_answerable and observation.actual_answered:
            false_answered += 1
        else:
            true_abstained += 1

        if observation.expected_answerable:
            if observation.cited_pages:
                answerable_cases_with_citations += 1

            citation_pages_total += len(observation.cited_pages)
            citation_pages_correct += sum(
                page in observation.expected_pages
                for page in observation.cited_pages
            )

    total = len(observations)
    answerable_total = true_answered + false_abstained
    unanswerable_total = true_abstained + false_answered
    classification_correct = true_answered + true_abstained

    return {
        "total_cases": total,
        "classification_accuracy": _rate(classification_correct, total),
        "answerable_answer_rate": _rate(true_answered, answerable_total),
        "unanswerable_abstention_rate": _rate(
            true_abstained,
            unanswerable_total,
        ),
        "citation_page_accuracy": _rate(
            citation_pages_correct,
            citation_pages_total,
        ),
        "answerable_citation_coverage": _rate(
            answerable_cases_with_citations,
            answerable_total,
        ),
        "citation_pages_correct": citation_pages_correct,
        "citation_pages_total": citation_pages_total,
        "confusion_matrix": {
            "true_answered": true_answered,
            "true_abstained": true_abstained,
            "false_answered": false_answered,
            "false_abstained": false_abstained,
        },
        "median_latency_ms": (
            round(statistics.median(latencies), 3) if latencies else None
        ),
        "p95_latency_ms": _percentile_95(latencies),
        "semantic_answer_correctness": None,
        "semantic_answer_correctness_note": (
            "Not automatically assessed. Requires reviewed "
            "answer-content labels or human evaluation."
        ),
    }
