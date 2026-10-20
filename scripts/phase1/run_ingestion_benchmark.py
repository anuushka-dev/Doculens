"""Benchmark PDF ingestion against manually verified page-level references.

The manifest contains expected body text for sampled pages. The benchmark
reports token coverage, page-number correctness, extraction status, and
latency. Document text is never written to the generated report.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import pymupdf

from app.adapters.pdf.extractor import PDFExtractionError
from app.core.config import get_settings
from app.core.paths import DATA_DIR, PROJECT_ROOT
from app.schemas.document import PDFExtractionResult
from app.services.pdf_ingestion import ingest_pdf_upload

SCHEMA_VERSION = "1.0"
TARGET_COVERAGE_PERCENT = 95.0
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

DEFAULT_MANIFEST_PATH = (
    DATA_DIR / "phase1" / "ingestion_benchmark_manifest.json"
)
DEFAULT_REPORT_PATH = (
    DATA_DIR / "phase1" / "ingestion_benchmark_report.json"
)


class BenchmarkInputError(ValueError):
    """Raised when the benchmark manifest is invalid."""


def resolve_cli_path(path: Path) -> Path:
    """Resolve relative CLI paths from the project root."""
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def tokenize(text: str) -> list[str]:
    """Normalize text and return Unicode-aware, case-insensitive tokens."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_PATTERN.findall(normalized)


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Reject ambiguous JSON objects containing repeated keys."""
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise BenchmarkInputError(
                f"The manifest contains a duplicate JSON key: {key!r}."
            )
        result[key] = value

    return result


def nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """Return a percentile using the nearest-rank method."""
    if not values:
        raise ValueError("values must not be empty")

    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in the interval (0, 1]")

    if any(not isfinite(value) for value in values):
        raise ValueError("values must all be finite numbers")

    ordered = sorted(values)
    rank = max(1, ceil(percentile * len(ordered)))

    return ordered[rank - 1]


def resolve_pdf_path(
    raw_path: str,
    document_index: int,
) -> tuple[Path, str]:
    """Resolve a repository-contained PDF without traversing symlinks."""
    project_root = PROJECT_ROOT.resolve()
    candidate = Path(raw_path)

    if not candidate.is_absolute():
        candidate = project_root / candidate

    # Normalize lexical ".." segments before checking the repository boundary.
    candidate = Path(os.path.abspath(candidate))

    try:
        candidate.relative_to(project_root)
    except ValueError as exc:
        raise BenchmarkInputError(
            f"documents[{document_index}].path must be inside the project root."
        ) from exc

    # Reject a symlink in any path component, not only the PDF itself.
    current = candidate

    while current != project_root:
        if current.is_symlink():
            raise BenchmarkInputError(
                f"documents[{document_index}].path must not traverse symlinks."
            )

        parent = current.parent

        if parent == current:
            raise BenchmarkInputError(
                f"documents[{document_index}].path is outside the project root."
            )

        current = parent

    try:
        resolved_path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BenchmarkInputError(
            f"documents[{document_index}].path does not resolve to a file."
        ) from exc

    try:
        relative_path = resolved_path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise BenchmarkInputError(
            f"documents[{document_index}].path must resolve inside "
            "the project root."
        ) from exc

    if not resolved_path.is_file():
        raise BenchmarkInputError(
            f"documents[{document_index}].path is not a regular file."
        )

    if resolved_path.suffix.casefold() != ".pdf":
        raise BenchmarkInputError(
            f"documents[{document_index}].path must reference a PDF."
        )

    return resolved_path, relative_path


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    """Load and validate the manually curated benchmark manifest."""
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_keys,
        )
    except json.JSONDecodeError as exc:
        raise BenchmarkInputError(
            f"The manifest contains invalid JSON at line {exc.lineno}."
        ) from exc

    if not isinstance(payload, dict):
        raise BenchmarkInputError(
            "The manifest root must be a JSON object."
        )

    if set(payload) != {"schema_version", "documents"}:
        raise BenchmarkInputError(
            "The manifest root must contain exactly "
            "'schema_version' and 'documents'."
        )

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkInputError(
            f"Manifest schema_version must be {SCHEMA_VERSION!r}."
        )

    raw_documents = payload.get("documents")

    if not isinstance(raw_documents, list) or not raw_documents:
        raise BenchmarkInputError(
            "The manifest must contain a non-empty documents array."
        )

    documents: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    for document_index, raw_document in enumerate(raw_documents):
        if (
            not isinstance(raw_document, dict)
            or set(raw_document) != {"path", "pages"}
        ):
            raise BenchmarkInputError(
                f"documents[{document_index}] must contain exactly "
                "'path' and 'pages'."
            )

        raw_path = raw_document["path"]

        if not isinstance(raw_path, str) or not raw_path.strip():
            raise BenchmarkInputError(
                f"documents[{document_index}].path must be a non-empty string."
            )

        pdf_path, relative_path = resolve_pdf_path(
            raw_path.strip(),
            document_index,
        )

        if pdf_path in seen_paths:
            raise BenchmarkInputError(
                f"Duplicate PDF in manifest: {relative_path}"
            )

        seen_paths.add(pdf_path)

        raw_pages = raw_document["pages"]

        if not isinstance(raw_pages, list) or not raw_pages:
            raise BenchmarkInputError(
                f"documents[{document_index}].pages must be a non-empty list."
            )

        pages: list[dict[str, Any]] = []
        seen_page_numbers: set[int] = set()
        previous_page_number = 0

        for page_index, raw_page in enumerate(raw_pages):
            required_page_keys = {"page_number", "expected_text"}
            allowed_page_keys = required_page_keys | {"visual_reviewed"}

            if (
                not isinstance(raw_page, dict)
                or not required_page_keys.issubset(raw_page)
                or not set(raw_page).issubset(allowed_page_keys)
            ):
                raise BenchmarkInputError(
                    f"documents[{document_index}].pages[{page_index}] "
                    "must contain 'page_number' and 'expected_text', with "
                    "optional 'visual_reviewed'."
                )

            page_number = raw_page["page_number"]

            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
            ):
                raise BenchmarkInputError(
                    f"documents[{document_index}].pages[{page_index}]"
                    ".page_number must be a positive integer."
                )

            if page_number in seen_page_numbers:
                raise BenchmarkInputError(
                    f"Duplicate page number {page_number} in {relative_path}."
                )

            if page_number <= previous_page_number:
                raise BenchmarkInputError(
                    f"Sampled page numbers in {relative_path} "
                    "must be in ascending order."
                )

            expected_text = raw_page["expected_text"]
            visual_reviewed = raw_page.get("visual_reviewed", False)

            if not isinstance(expected_text, str):
                raise BenchmarkInputError(
                    f"Expected text for page {page_number} must be a string."
                )

            if not isinstance(visual_reviewed, bool):
                raise BenchmarkInputError(
                    f"visual_reviewed for page {page_number} must be a boolean."
                )

            if not tokenize(expected_text):
                raise BenchmarkInputError(
                    f"Expected text for page {page_number} in "
                    f"{relative_path} must contain at least one token."
                )

            pages.append(
                {
                    "page_number": page_number,
                    "expected_text": expected_text,
                    "visual_reviewed": visual_reviewed,
                }
            )

            seen_page_numbers.add(page_number)
            previous_page_number = page_number

        documents.append(
            {
                "path": pdf_path,
                "relative_path": relative_path,
                "pages": pages,
            }
        )

    return documents


def read_upload_bytes(path: Path, max_upload_bytes: int) -> bytes:
    """Read no more than one byte beyond the configured upload limit."""
    if (
        isinstance(max_upload_bytes, bool)
        or not isinstance(max_upload_bytes, int)
        or max_upload_bytes < 1
    ):
        raise ValueError("max_upload_bytes must be a positive integer")

    with path.open("rb") as source:
        return source.read(max_upload_bytes + 1)


def evaluate_page(
    expected_page: dict[str, Any],
    extracted_result: PDFExtractionResult | None,
) -> dict[str, Any]:
    """Measure expected-token coverage and verify the sampled page ID."""
    page_number = expected_page["page_number"]
    expected_counts = Counter(tokenize(expected_page["expected_text"]))
    expected_token_count = sum(expected_counts.values())

    actual_page = None

    if extracted_result is not None:
        # Compare the expected source page to the record at that position.
        # Searching by page_number first would make this check tautological.
        page_index = page_number - 1

        if page_index < len(extracted_result.pages):
            actual_page = extracted_result.pages[page_index]

    if actual_page is None:
        matched_token_count = 0
        page_id_correct = False
    else:
        actual_counts = Counter(tokenize(actual_page.text))
        matched_token_count = sum(
            (expected_counts & actual_counts).values()
        )
        page_id_correct = actual_page.page_number == page_number

    coverage_percent = round(
        100.0 * matched_token_count / expected_token_count,
        4,
    )

    meets_target = (
        page_id_correct
        and coverage_percent >= TARGET_COVERAGE_PERCENT
    )

    if extracted_result is None:
        page_status = "ingestion_failed"
    elif actual_page is None:
        page_status = "expected_page_missing"
    elif not page_id_correct:
        page_status = "page_id_mismatch"
    elif coverage_percent < TARGET_COVERAGE_PERCENT:
        page_status = "coverage_below_target"
    else:
        page_status = "pass"

    return {
        "page_number": page_number,
        "actual_page_number": (
            actual_page.page_number if actual_page is not None else None
        ),
        "page_id_correct": page_id_correct,
        "visual_reviewed": expected_page.get("visual_reviewed", False),
        "expected_token_count": expected_token_count,
        "matched_expected_token_count": matched_token_count,
        "expected_body_text_coverage_percent": coverage_percent,
        "meets_page_target": meets_target,
        "status": page_status,
    }


def benchmark_document(
    document: dict[str, Any],
    *,
    max_upload_bytes: int,
) -> dict[str, Any]:
    """Run the application ingestion service for one benchmark PDF."""
    pdf_path: Path = document["path"]
    extracted_result: PDFExtractionResult | None = None
    status = "ok"
    error_code: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    file_size_bytes: int | None = None
    source_sha256: str | None = None

    started_at = perf_counter()

    try:
        file_size_bytes = pdf_path.stat().st_size
        pdf_bytes = read_upload_bytes(pdf_path, max_upload_bytes)

        # Never label a truncated over-limit prefix as the PDF's full digest.
        if (
            file_size_bytes <= max_upload_bytes
            and len(pdf_bytes) == file_size_bytes
        ):
            source_sha256 = hashlib.sha256(pdf_bytes).hexdigest()

        extracted_result = ingest_pdf_upload(
            pdf_bytes=pdf_bytes,
            filename=pdf_path.name,
        )

    except PDFExtractionError as exc:
        status = "ingestion_error"
        error_code = exc.code
        error_message = str(exc)

    except OSError as exc:
        status = "file_access_error"
        error_type = type(exc).__name__

    except (RuntimeError, TypeError, ValueError) as exc:
        # Preserve a report when one sample hits an unexpected application bug.
        status = "processing_error"
        error_type = type(exc).__name__

    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)

    page_results = [
        evaluate_page(page, extracted_result)
        for page in document["pages"]
    ]

    return {
        "path": document["relative_path"],
        "filename": pdf_path.name,
        "file_size_bytes": file_size_bytes,
        "sha256": source_sha256,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "page_count": (
            extracted_result.page_count
            if extracted_result is not None
            else None
        ),
        "sampled_page_count": len(page_results),
        "all_sampled_pages_meet_target": bool(page_results)
        and all(page["meets_page_target"] for page in page_results),
        "all_sampled_pages_visually_reviewed": bool(page_results)
        and all(page["visual_reviewed"] is True for page in page_results),
        "error_code": error_code,
        "error_message": error_message,
        "error_type": error_type,
        "sampled_pages": page_results,
    }


def summarize_results(
    document_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate weighted coverage, per-page gates, review status, and latency."""
    sampled_pages = [
        page
        for document in document_results
        for page in document["sampled_pages"]
    ]

    expected_token_count = sum(
        page["expected_token_count"] for page in sampled_pages
    )
    matched_token_count = sum(
        page["matched_expected_token_count"] for page in sampled_pages
    )

    coverage_percent = (
        round(100.0 * matched_token_count / expected_token_count, 4)
        if expected_token_count > 0
        else 0.0
    )

    all_documents_ingested = bool(document_results) and all(
        document["status"] == "ok"
        for document in document_results
    )

    page_ids_correct = bool(sampled_pages) and all(
        page["page_id_correct"] for page in sampled_pages
    )

    all_pages_meet_target = bool(sampled_pages) and all(
        page["meets_page_target"] for page in sampled_pages
    )

    all_pages_visually_reviewed = bool(sampled_pages) and all(
        page.get("visual_reviewed", False) is True
        for page in sampled_pages
    )

    latencies = [
        float(document["elapsed_ms"])
        for document in document_results
        if isinstance(document.get("elapsed_ms"), (int, float))
        and not isinstance(document.get("elapsed_ms"), bool)
        and isfinite(float(document["elapsed_ms"]))
        and float(document["elapsed_ms"]) >= 0
    ]

    latency_summary: dict[str, float | int | None] = {
        "sample_count": len(latencies),
        "median_ms": (
            round(float(median(latencies)), 2) if latencies else None
        ),
        "p95_nearest_rank_ms": (
            round(nearest_rank_percentile(latencies, 0.95), 2)
            if latencies
            else None
        ),
        "max_ms": round(max(latencies), 2) if latencies else None,
    }

    acceptance_passed = (
        bool(sampled_pages)
        and all_documents_ingested
        and page_ids_correct
        and all_pages_meet_target
        and all_pages_visually_reviewed
        and coverage_percent >= TARGET_COVERAGE_PERCENT
    )

    return {
        "document_count": len(document_results),
        "sampled_page_count": len(sampled_pages),
        "expected_token_count": expected_token_count,
        "matched_expected_token_count": matched_token_count,
        "expected_body_text_coverage_percent": coverage_percent,
        "target_coverage_percent": TARGET_COVERAGE_PERCENT,
        "all_documents_ingested": all_documents_ingested,
        "page_ids_correct": page_ids_correct,
        "all_sampled_pages_meet_target": all_pages_meet_target,
        "all_sampled_pages_visually_reviewed": all_pages_visually_reviewed,
        "latency_ms": latency_summary,
        "acceptance_passed": acceptance_passed,
    }


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write the JSON report atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(
                payload,
                temporary_file,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            temporary_file.write("\n")

        os.replace(temporary_path, path)

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 1 ingestion benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PDF ingestion against manually verified "
            "page-level expected text."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="JSON manifest containing PDFs and expected page text.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Destination for the JSON benchmark report.",
    )

    args = parser.parse_args(argv)
    manifest_path = resolve_cli_path(args.manifest)
    report_path = resolve_cli_path(args.report)

    try:
        documents = load_manifest(manifest_path)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Benchmark input error: {exc}", file=sys.stderr)
        return 2

    protected_inputs = {manifest_path.resolve()}
    protected_inputs.update(
        document["path"].resolve() for document in documents
    )

    if report_path.resolve() in protected_inputs:
        print(
            "Benchmark output path must not overwrite the manifest or a source PDF.",
            file=sys.stderr,
        )
        return 2

    try:
        settings = get_settings()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            "Could not load DocuLens settings due to a configuration "
            f"or environment error: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2

    document_results: list[dict[str, Any]] = []

    print(f"Manifest: {manifest_path}")
    print(f"Documents to benchmark: {len(documents)}")
    print(f"Target expected-text coverage: >= {TARGET_COVERAGE_PERCENT:.2f}%")

    for document in documents:
        result = benchmark_document(
            document,
            max_upload_bytes=settings.max_upload_bytes,
        )
        document_results.append(result)

        print(
            f"{result['path']}: "
            f"status={result['status']}, "
            f"pages={result['page_count']}, "
            f"sampled_pages={result['sampled_page_count']}, "
            f"elapsed_ms={result['elapsed_ms']}"
        )

    summary = summarize_results(document_results)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "pymupdf_version": getattr(pymupdf, "__version__", "unknown"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "purpose": "Phase 1 PDF ingestion acceptance benchmark",
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_pdf_pages": settings.max_pdf_pages,
        },
        "measurement": {
            "token_normalization": "NFKC, case-insensitive, Unicode words",
            "coverage_method": (
                "Matched expected-token occurrences divided by expected "
                "token occurrences, weighted across sampled pages. "
                "This is expected-text token coverage, not extraction precision "
                "or a general-purpose F1 score."
            ),
            "page_lookup_method": (
                "Compare each expected source page to the extracted record "
                "at its one-based position; the schema separately enforces "
                "consecutive one-based page numbering."
            ),
            "latency_method": (
                "Per-document elapsed time includes bounded file reading "
                "and the ingestion service call; summary includes all attempts."
            ),
            "p95_method": "Nearest rank",
            "acceptance_rule": (
                "At least one sampled page; every document ingests successfully; "
                "every labelled page has the expected page number and at least "
                "95% expected-token coverage; every sampled page is marked "
                "visually reviewed in the manifest."
            ),
            "scope_note": (
                "This script calls the ingestion service directly. HTTP status "
                "and error-code acceptance is covered separately by API tests."
            ),
            "document_text_saved_in_report": False,
        },
        "summary": summary,
        "documents": document_results,
    }

    try:
        write_json_atomically(report_path, report)
    except OSError as exc:
        print(
            f"Could not write benchmark report: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 2

    print()
    print(
        "Expected body-text coverage: "
        f"{summary['expected_body_text_coverage_percent']:.2f}%"
    )
    print(f"Page IDs correct: {summary['page_ids_correct']}")
    print(
        "Every sampled page meets target: "
        f"{summary['all_sampled_pages_meet_target']}"
    )
    print(
        "Every sampled page visually reviewed: "
        f"{summary['all_sampled_pages_visually_reviewed']}"
    )
    print(f"All documents ingested: {summary['all_documents_ingested']}")
    print(f"Latency summary: {summary['latency_ms']}")
    print(f"Acceptance passed: {summary['acceptance_passed']}")
    print(f"Report written to: {report_path}")

    return 0 if summary["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())