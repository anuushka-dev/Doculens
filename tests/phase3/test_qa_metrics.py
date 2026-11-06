"""Tests for deterministic grounded-Q&A metric calculations."""

from __future__ import annotations

import pytest

from app.evaluation.qa_metrics import (
    QAEvaluationObservation,
    calculate_qa_metrics,
)


def observation(
    *,
    expected_answerable: bool,
    actual_answered: bool,
    expected_pages: tuple[int, ...] = (),
    cited_pages: tuple[int, ...] = (),
    latency_ms: float = 10.0,
) -> QAEvaluationObservation:
    """Build one validated metric observation."""

    return QAEvaluationObservation(
        expected_answerable=expected_answerable,
        actual_answered=actual_answered,
        expected_pages=expected_pages,
        cited_pages=cited_pages,
        latency_ms=latency_ms,
    )


class TestQAEvaluationObservation:
    def test_valid_answerable_observation(self):
        item = observation(
            expected_answerable=True,
            actual_answered=True,
            expected_pages=(2, 3),
            cited_pages=(2,),
        )

        assert item.expected_pages == (2, 3)
        assert item.cited_pages == (2,)

    def test_valid_unanswerable_observation(self):
        item = observation(
            expected_answerable=False,
            actual_answered=False,
        )

        assert item.expected_pages == ()

    def test_answerable_case_requires_expected_pages(self):
        with pytest.raises(ValueError, match="require expected pages"):
            observation(
                expected_answerable=True,
                actual_answered=True,
            )

    def test_unanswerable_case_cannot_have_expected_pages(self):
        with pytest.raises(ValueError, match="must not have expected pages"):
            observation(
                expected_answerable=False,
                actual_answered=False,
                expected_pages=(1,),
            )

    @pytest.mark.parametrize("page", [0, -1, True, 1.5])
    def test_page_numbers_must_be_positive_integers(self, page):
        with pytest.raises(ValueError, match="positive integers"):
            QAEvaluationObservation(
                expected_answerable=True,
                actual_answered=True,
                expected_pages=(1,),
                cited_pages=(page,),
                latency_ms=1.0,
            )

    @pytest.mark.parametrize("latency", [-0.1, float("inf"), float("nan")])
    def test_latency_must_be_finite_and_non_negative(self, latency):
        with pytest.raises(ValueError, match="finite and non-negative"):
            observation(
                expected_answerable=True,
                actual_answered=True,
                expected_pages=(1,),
                latency_ms=latency,
            )

    def test_latency_must_be_numeric(self):
        with pytest.raises(TypeError, match="must be a number"):
            observation(
                expected_answerable=True,
                actual_answered=True,
                expected_pages=(1,),
                latency_ms="10",  # type: ignore[arg-type]
            )


class TestCalculateQAMetrics:
    def test_perfect_classification_and_citation_metrics(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(2, 3),
                    cited_pages=(2, 3),
                    latency_ms=10,
                ),
                observation(
                    expected_answerable=False,
                    actual_answered=False,
                    latency_ms=20,
                ),
            ]
        )

        assert metrics["total_cases"] == 2
        assert metrics["classification_accuracy"] == 1.0
        assert metrics["answerable_answer_rate"] == 1.0
        assert metrics["unanswerable_abstention_rate"] == 1.0
        assert metrics["citation_page_accuracy"] == 1.0
        assert metrics["answerable_citation_coverage"] == 1.0
        assert metrics["citation_pages_correct"] == 2
        assert metrics["citation_pages_total"] == 2

    def test_mixed_classes_produce_expected_confusion_matrix(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(2,),
                    cited_pages=(2,),
                    latency_ms=10,
                ),
                observation(
                    expected_answerable=True,
                    actual_answered=False,
                    expected_pages=(4,),
                    latency_ms=30,
                ),
                observation(
                    expected_answerable=False,
                    actual_answered=False,
                    latency_ms=20,
                ),
                observation(
                    expected_answerable=False,
                    actual_answered=True,
                    cited_pages=(8,),
                    latency_ms=40,
                ),
            ]
        )

        assert metrics["classification_accuracy"] == 0.5
        assert metrics["answerable_answer_rate"] == 0.5
        assert metrics["unanswerable_abstention_rate"] == 0.5
        assert metrics["confusion_matrix"] == {
            "true_answered": 1,
            "true_abstained": 1,
            "false_answered": 1,
            "false_abstained": 1,
        }

    def test_citation_page_accuracy_uses_citation_count_as_denominator(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(2, 4),
                    cited_pages=(2, 3, 4),
                )
            ]
        )

        assert metrics["citation_pages_correct"] == 2
        assert metrics["citation_pages_total"] == 3
        assert metrics["citation_page_accuracy"] == 0.666667

    def test_citation_coverage_counts_answerable_cases_with_any_citation(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(1,),
                    cited_pages=(1, 1),
                ),
                observation(
                    expected_answerable=True,
                    actual_answered=False,
                    expected_pages=(2,),
                ),
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(3,),
                    cited_pages=(4,),
                ),
            ]
        )

        assert metrics["answerable_citation_coverage"] == 0.666667
        assert metrics["citation_pages_total"] == 3
        assert metrics["citation_pages_correct"] == 2

    def test_citations_on_unanswerable_cases_are_excluded_from_page_accuracy(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=False,
                    actual_answered=True,
                    cited_pages=(7,),
                ),
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(2,),
                    cited_pages=(2,),
                ),
            ]
        )

        assert metrics["citation_pages_total"] == 1
        assert metrics["citation_pages_correct"] == 1
        assert metrics["citation_page_accuracy"] == 1.0

    def test_no_citations_returns_none_page_accuracy_and_zero_coverage(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(1,),
                )
            ]
        )

        assert metrics["citation_page_accuracy"] is None
        assert metrics["citation_pages_total"] == 0
        assert metrics["answerable_citation_coverage"] == 0.0

    def test_empty_dataset_returns_none_rates_and_latency(self):
        metrics = calculate_qa_metrics([])

        assert metrics["total_cases"] == 0
        assert metrics["classification_accuracy"] is None
        assert metrics["answerable_answer_rate"] is None
        assert metrics["unanswerable_abstention_rate"] is None
        assert metrics["citation_page_accuracy"] is None
        assert metrics["answerable_citation_coverage"] is None
        assert metrics["median_latency_ms"] is None
        assert metrics["p95_latency_ms"] is None
        assert metrics["confusion_matrix"] == {
            "true_answered": 0,
            "true_abstained": 0,
            "false_answered": 0,
            "false_abstained": 0,
        }

    def test_absent_answerable_class_returns_none_answer_rate_and_coverage(self):
        metrics = calculate_qa_metrics(
            [observation(expected_answerable=False, actual_answered=False)]
        )

        assert metrics["answerable_answer_rate"] is None
        assert metrics["answerable_citation_coverage"] is None
        assert metrics["unanswerable_abstention_rate"] == 1.0

    def test_absent_unanswerable_class_returns_none_abstention_rate(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(1,),
                )
            ]
        )

        assert metrics["unanswerable_abstention_rate"] is None
        assert metrics["answerable_answer_rate"] == 1.0

    def test_median_latency_uses_median_and_rounds_to_three_decimals(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=False,
                    actual_answered=False,
                    latency_ms=1.1114,
                ),
                observation(
                    expected_answerable=False,
                    actual_answered=False,
                    latency_ms=2.2224,
                ),
                observation(
                    expected_answerable=False,
                    actual_answered=False,
                    latency_ms=9.9994,
                ),
                observation(
                    expected_answerable=False,
                    actual_answered=False,
                    latency_ms=10.0,
                ),
            ]
        )

        assert metrics["median_latency_ms"] == 6.111

    def test_p95_uses_nearest_rank(self):
        values = [
            observation(
                expected_answerable=False,
                actual_answered=False,
                latency_ms=value,
            )
            for value in range(1, 21)
        ]

        metrics = calculate_qa_metrics(values)

        assert metrics["p95_latency_ms"] == 19.0

    def test_single_latency_is_both_median_and_p95(self):
        metrics = calculate_qa_metrics(
            [
                observation(
                    expected_answerable=True,
                    actual_answered=True,
                    expected_pages=(1,),
                    latency_ms=12.34567,
                )
            ]
        )

        assert metrics["median_latency_ms"] == 12.346
        assert metrics["p95_latency_ms"] == 12.346

    def test_semantic_correctness_remains_unmeasured_with_explanation(self):
        metrics = calculate_qa_metrics([])

        assert metrics["semantic_answer_correctness"] is None
        assert "human evaluation" in metrics["semantic_answer_correctness_note"]
