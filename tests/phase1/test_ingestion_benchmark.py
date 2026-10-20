"""Regression tests for the Phase 1 ingestion benchmark gate."""

import json
from types import SimpleNamespace

import pytest

from scripts.phase1 import run_ingestion_benchmark as benchmark


def _page_result(page_number: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(page_number=page_number, text=text)


def _page_case(
    *,
    page_number: int,
    expected_text: str,
    extracted_text: str,
    actual_page_number: int | None = None,
    visual_reviewed: bool = True,
) -> dict[str, object]:
    expected_page = {
        "page_number": page_number,
        "expected_text": expected_text,
        "visual_reviewed": visual_reviewed,
    }
    result = SimpleNamespace(
        pages=(
            _page_result(
                actual_page_number
                if actual_page_number is not None
                else page_number,
                extracted_text,
            ),
        )
    )
    return benchmark.evaluate_page(expected_page, result)


def test_tokenize_normalizes_compatibility_forms_and_case() -> None:
    assert benchmark.tokenize("ＦＯＯ foo CAFÉ café") == [
        "foo",
        "foo",
        "café",
        "café",
    ]


def test_token_coverage_counts_repeated_expected_occurrences() -> None:
    result = _page_case(
        page_number=1,
        expected_text="evidence evidence source",
        extracted_text="evidence source source",
    )

    assert result["expected_token_count"] == 3
    assert result["matched_expected_token_count"] == 2
    assert result["expected_body_text_coverage_percent"] == 66.6667
    assert result["meets_page_target"] is False


def test_page_check_uses_expected_page_position() -> None:
    result = _page_case(
        page_number=1,
        expected_text="target evidence stays on the first page",
        extracted_text="target evidence stays on the first page",
        actual_page_number=2,
    )

    assert result["actual_page_number"] == 2
    assert result["page_id_correct"] is False
    assert result["meets_page_target"] is False
    assert result["status"] == "page_id_mismatch"


def test_nearest_rank_percentile() -> None:
    assert benchmark.nearest_rank_percentile(
        [10.0, 20.0, 30.0, 40.0],
        0.95,
    ) == 40.0
    assert benchmark.nearest_rank_percentile(
        [40.0, 10.0, 30.0, 20.0],
        0.50,
    ) == 20.0


@pytest.mark.parametrize(
    ("values", "percentile", "message"),
    [
        ([], 0.95, "not be empty"),
        ([1.0], 0.0, r"\(0, 1\]"),
        ([1.0], 1.1, r"\(0, 1\]"),
        ([float("nan")], 0.95, "finite"),
    ],
)
def test_nearest_rank_percentile_rejects_invalid_input(
    values: list[float],
    percentile: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.nearest_rank_percentile(values, percentile)


def test_aggregate_coverage_cannot_hide_a_failing_page() -> None:
    pages = [
        {
            "page_number": 1,
            "page_id_correct": True,
            "expected_token_count": 100,
            "matched_expected_token_count": 100,
            "expected_body_text_coverage_percent": 100.0,
            "meets_page_target": True,
            "visual_reviewed": True,
            "status": "pass",
        },
        {
            "page_number": 2,
            "page_id_correct": True,
            "expected_token_count": 10,
            "matched_expected_token_count": 9,
            "expected_body_text_coverage_percent": 90.0,
            "meets_page_target": False,
            "visual_reviewed": True,
            "status": "coverage_below_target",
        },
    ]
    report = benchmark.summarize_results(
        [
            {
                "status": "ok",
                "elapsed_ms": 10.0,
                "sampled_pages": pages,
            }
        ]
    )

    assert report["expected_body_text_coverage_percent"] > 95.0
    assert report["all_sampled_pages_meet_target"] is False
    assert report["acceptance_passed"] is False


def test_acceptance_requires_visual_review() -> None:
    page = {
        "page_number": 1,
        "page_id_correct": True,
        "expected_token_count": 5,
        "matched_expected_token_count": 5,
        "expected_body_text_coverage_percent": 100.0,
        "meets_page_target": True,
        "visual_reviewed": False,
        "status": "pass",
    }
    report = benchmark.summarize_results(
        [
            {
                "status": "ok",
                "elapsed_ms": 12.0,
                "sampled_pages": [page],
            }
        ]
    )

    assert report["all_sampled_pages_meet_target"] is True
    assert report["all_sampled_pages_visually_reviewed"] is False
    assert report["acceptance_passed"] is False


def test_empty_summary_fails_closed_without_division_by_zero() -> None:
    report = benchmark.summarize_results([])

    assert report["expected_body_text_coverage_percent"] == 0.0
    assert report["latency_ms"]["sample_count"] == 0
    assert report["latency_ms"]["median_ms"] is None
    assert report["latency_ms"]["p95_nearest_rank_ms"] is None
    assert report["acceptance_passed"] is False


def test_manifest_defaults_visual_review_to_false(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path.resolve())
    (tmp_path / "sample.pdf").write_bytes(b"%PDF-synthetic-test")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "documents": [
                    {
                        "path": "sample.pdf",
                        "pages": [
                            {
                                "page_number": 1,
                                "expected_text": "verified expected words",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    documents = benchmark.load_manifest(manifest_path)

    assert documents[0]["pages"][0]["visual_reviewed"] is False


def test_manifest_rejects_duplicate_json_keys(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path.resolve())
    (tmp_path / "sample.pdf").write_bytes(b"%PDF-synthetic-test")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "schema_version": "1.0",
          "schema_version": "1.0",
          "documents": []
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(
        benchmark.BenchmarkInputError,
        match="duplicate JSON key",
    ):
        benchmark.load_manifest(manifest_path)


def test_manifest_rejects_symlinked_pdf(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path.resolve())
    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(b"%PDF-synthetic-test")
    linked_pdf = tmp_path / "linked.pdf"

    try:
        linked_pdf.symlink_to(real_pdf)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not supported in this environment.")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "documents": [
                    {
                        "path": "linked.pdf",
                        "pages": [
                            {
                                "page_number": 1,
                                "expected_text": "expected words",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        benchmark.BenchmarkInputError,
        match="symlinks",
    ):
        benchmark.load_manifest(manifest_path)
