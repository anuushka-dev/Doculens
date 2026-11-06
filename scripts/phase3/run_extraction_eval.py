
"""Evaluate structured metadata extraction against a reviewed JSONL dataset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.adapters.llm.ollama_client import (
    OllamaClient,
    OllamaClientError,
)
from app.core.config import get_settings
from app.schemas.extraction import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSchemaName,
)
from app.services.pdf_ingestion import ingest_pdf_upload
from app.services.structured_extraction import (
    StructuredExtractionError,
    StructuredExtractionService,
)

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "tests"
    / "phase3"
    / "fixtures"
    / "extraction_eval.jsonl"
)

DEFAULT_EXPECTED_TOTAL = 20

METADATA_FIELDS = (
    "document_title",
    "issuing_organization",
    "document_identifier",
    "publication_date",
    "version",
)


@dataclass(frozen=True, slots=True)
class ExtractionExample:
    """One reviewed metadata extraction example."""

    case_id: str
    pdf_path: str
    expected_metadata: dict[str, str | None]
    expected_field_pages: dict[str, tuple[int, ...]]
    review_status: str
    review_note: str
    line_number: int


def _normalize_value(value: str | None) -> str | None:
    """Normalize text for case-insensitive, whitespace-tolerant comparison."""

    if value is None:
        return None

    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.casefold().split())


def _parse_expected_metadata(
    value: Any,
    line_number: int,
) -> dict[str, str | None]:
    """Validate all five expected metadata fields."""

    if not isinstance(value, dict):
        raise TypeError(
            f"Line {line_number}: expected_metadata must be an object."
        )

    missing = set(METADATA_FIELDS) - set(value)
    extra = set(value) - set(METADATA_FIELDS)

    if missing:
        raise ValueError(
            f"Line {line_number}: missing metadata fields: "
            f"{', '.join(sorted(missing))}"
        )

    if extra:
        raise ValueError(
            f"Line {line_number}: unexpected metadata fields: "
            f"{', '.join(sorted(extra))}"
        )

    metadata: dict[str, str | None] = {}

    for field_name in METADATA_FIELDS:
        field_value = value[field_name]

        if field_value is not None and not isinstance(field_value, str):
            raise TypeError(
                f"Line {line_number}: {field_name} must be a string or null."
            )

        if isinstance(field_value, str) and not field_value.strip():
            raise ValueError(
                f"Line {line_number}: {field_name} must be null "
                "instead of an empty string."
            )

        metadata[field_name] = field_value

    return metadata


def _parse_field_pages(
    value: Any,
    line_number: int,
) -> dict[str, tuple[int, ...]]:
    """Parse optional field-specific source-page annotations."""

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(
            f"Line {line_number}: expected_field_pages must be an object."
        )

    unknown = set(value) - set(METADATA_FIELDS)

    if unknown:
        raise ValueError(
            f"Line {line_number}: unknown field-page keys: "
            f"{', '.join(sorted(unknown))}"
        )

    result: dict[str, tuple[int, ...]] = {}

    for field_name, page_values in value.items():
        if not isinstance(page_values, list):
            raise TypeError(
                f"Line {line_number}: pages for {field_name} must be a list."
            )

        pages: list[int] = []

        for page in page_values:
            if (
                isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
            ):
                raise ValueError(
                    f"Line {line_number}: page numbers must be "
                    "positive one-based integers."
                )

            pages.append(page)

        result[field_name] = tuple(sorted(set(pages)))

    return result


def _parse_example(
    row: dict[str, Any],
    line_number: int,
) -> ExtractionExample:
    """Validate and construct one extraction example."""

    case_id = row.get("case_id", row.get("id"))

    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Line {line_number}: missing case_id.")

    pdf_path = row.get("pdf_path")

    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise ValueError(f"Line {line_number}: missing pdf_path.")

    expected_metadata = _parse_expected_metadata(
        row.get("expected_metadata"),
        line_number,
    )

    expected_field_pages = _parse_field_pages(
        row.get("expected_field_pages"),
        line_number,
    )

    review_status = row.get("review_status", "pending_review")

    if not isinstance(review_status, str):
        raise TypeError(
            f"Line {line_number}: review_status must be a string."
        )

    review_note = row.get("review_note", "")

    if not isinstance(review_note, str):
        raise TypeError(
            f"Line {line_number}: review_note must be a string."
        )

    return ExtractionExample(
        case_id=case_id.strip(),
        pdf_path=pdf_path.strip(),
        expected_metadata=expected_metadata,
        expected_field_pages=expected_field_pages,
        review_status=review_status.strip().lower(),
        review_note=review_note.strip(),
        line_number=line_number,
    )


def load_dataset(
    dataset_path: Path,
) -> tuple[list[ExtractionExample], str]:
    """Load extraction examples and calculate the source-file hash."""

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Extraction dataset not found: {dataset_path}"
        )

    raw_bytes = dataset_path.read_bytes()

    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()

    examples: list[ExtractionExample] = []

    for line_number, raw_line in enumerate(
        raw_bytes.decode("utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue

        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Line {line_number}: invalid JSON: {exc.msg}"
            ) from exc

        if not isinstance(row, dict):
            raise TypeError(
                f"Line {line_number}: each JSONL row must be an object."
            )

        examples.append(
            _parse_example(row, line_number)
        )

    if not examples:
        raise ValueError(
            "The extraction dataset contains no examples."
        )

    identifiers = [example.case_id for example in examples]

    duplicates = [
        case_id
        for case_id, count in Counter(identifiers).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate extraction IDs: {', '.join(duplicates)}"
        )

    return examples, dataset_hash


def validate_dataset(
    examples: list[ExtractionExample],
    *,
    expected_total: int,
) -> None:
    """Enforce dataset size and gold-label review requirements."""

    if len(examples) != expected_total:
        raise ValueError(
            f"Expected {expected_total} extraction examples, "
            f"found {len(examples)}."
        )

    pending = [
        example.case_id
        for example in examples
        if (
            example.review_status != "reviewed"
            or not example.review_note
        )
    ]

    if pending:
        preview = ", ".join(pending[:10])

        raise ValueError(
            "Extraction evaluation blocked because gold annotations "
            "are not fully reviewed. Every row needs "
            "review_status='reviewed' and a non-empty review_note. "
            f"Pending IDs: {preview}"
        )


def resolve_pdf_path(
    example: ExtractionExample,
    *,
    dataset_path: Path,
) -> Path:
    """Resolve a PDF path relative to the project or fixture directory."""

    path = Path(example.pdf_path).expanduser()

    if not path.is_absolute():
        project_relative = PROJECT_ROOT / path
        dataset_relative = dataset_path.parent / path

        if project_relative.is_file():
            path = project_relative
        elif dataset_relative.is_file():
            path = dataset_relative
        else:
            path = project_relative

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF for case {example.case_id} not found: {path}"
        )

    return path


def _rate(
    numerator: int,
    denominator: int,
) -> float | None:
    """Calculate a rate, returning null for an empty denominator."""

    if denominator == 0:
        return None

    return round(numerator / denominator, 6)


def _percentile_95(
    values: list[float],
) -> float | None:
    """Calculate the nearest-rank 95th percentile."""

    if not values:
        return None

    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)

    return round(ordered[index], 3)

def _exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    """Return safe diagnostics for chained exceptions."""

    chain: list[dict[str, Any]] = []
    cause = exc.__cause__

    while cause is not None:
        detail: dict[str, Any] = {
            "type": type(cause).__name__,
        }

        if isinstance(cause, ValidationError):
            detail["validation_errors"] = [
                {
                    "location": list(error["loc"]),
                    "type": error["type"],
                    "message": error["msg"],
                }
                for error in cause.errors(include_input=False)
            ]

        chain.append(detail)
        cause = cause.__cause__

    return chain


def _extract_metadata(
    response: ExtractionResponse,
) -> dict[str, str | None]:
    """Convert the typed extraction response to a plain metadata mapping."""

    metadata = response.data.model_dump(mode="python")

    return {
        field_name: metadata[field_name]
        for field_name in METADATA_FIELDS
    }


def _extract_citations(
    response: ExtractionResponse,
    example: ExtractionExample,
) -> list[dict[str, Any]]:
    """Serialize field citations and optional page-match annotations."""

    results: list[dict[str, Any]] = []

    for item in response.field_citations:
        citation = item.citation
        expected_pages = example.expected_field_pages.get(
            item.field_name
        )

        results.append(
            {
                "field_name": item.field_name,
                "citation_id": citation.citation_id,
                "chunk_id": citation.chunk_id,
                "page_number": citation.page_number,
                "expected_pages": (
                    list(expected_pages)
                    if expected_pages is not None
                    else None
                ),
                "page_matches": (
                    citation.page_number in expected_pages
                    if expected_pages is not None
                    else None
                ),
            }
        )

    return results


async def run_evaluation(
    examples: list[ExtractionExample],
    *,
    dataset_hash: str,
    dataset_path: Path,
) -> dict[str, Any]:
    """Run structured extraction and calculate field-level metrics."""

    settings = get_settings()

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
        temperature=settings.ollama_temperature,
        seed=settings.ollama_seed,
    )

    service = StructuredExtractionService(
        llm_client=client,
        evidence_threshold=settings.evidence_threshold,
        max_context_chars=settings.max_context_chars,
        max_repair_attempts=settings.extraction_max_repair_attempts,
    )

    extracted_documents: dict[Path, Any] = {}

    results: list[dict[str, Any]] = []
    latencies: list[float] = []

    field_correct: dict[str, int] = {
        field_name: 0
        for field_name in METADATA_FIELDS
    }
    field_total: dict[str, int] = {
        field_name: 0
        for field_name in METADATA_FIELDS
    }

    populated_correct = 0
    populated_total = 0

    null_correct = 0
    null_total = 0

    overall_correct = 0
    completed_records = 0
    schema_valid_records = 0

    page_matches = 0
    page_annotations = 0

    status_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()

    try:
        for index, example in enumerate(examples, start=1):
            pdf_path = resolve_pdf_path(
                example,
                dataset_path=dataset_path,
            )

            if pdf_path not in extracted_documents:
                pdf_bytes = pdf_path.read_bytes()

                extraction = await asyncio.to_thread(
                    ingest_pdf_upload,
                    pdf_bytes=pdf_bytes,
                    filename=pdf_path.name,
                )

                extracted_documents[pdf_path] = extraction

            extraction = extracted_documents[pdf_path]

            request = ExtractionRequest(
                schema_name=(
                    ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
                ),
                top_k=settings.retrieval_top_k,
            )

            start = time.perf_counter()

            try:
                response = await service.extract_pages(
                    pages=extraction.pages,
                    document_id=str(extraction.document_id),
                    file_name=extraction.filename,
                    request=request,
                )

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(latency_ms)

                if not isinstance(response, ExtractionResponse):
                    raise TypeError(
                        "Service returned an unexpected response type."
                    )

                schema_valid_records += 1
                completed_records += 1

                actual_metadata = _extract_metadata(response)

                case_field_correct = 0
                case_field_total = 0

                for field_name in METADATA_FIELDS:
                    expected = example.expected_metadata[field_name]
                    actual = actual_metadata[field_name]

                    field_total[field_name] += 1
                    case_field_total += 1

                    expected_normalized = _normalize_value(expected)
                    actual_normalized = _normalize_value(actual)

                    matches = (
                        expected_normalized == actual_normalized
                    )

                    if matches:
                        field_correct[field_name] += 1
                        case_field_correct += 1

                    if expected is None:
                        null_total += 1

                        if matches:
                            null_correct += 1
                    else:
                        populated_total += 1

                        if matches:
                            populated_correct += 1

                exact_record_match = (
                    case_field_correct == len(METADATA_FIELDS)
                )

                if exact_record_match:
                    overall_correct += 1

                citations = _extract_citations(
                    response,
                    example,
                )

                for citation in citations:
                    if citation["page_matches"] is not None:
                        page_annotations += 1

                        if citation["page_matches"]:
                            page_matches += 1

                extraction_status = response.status.value
                status_counts[extraction_status] += 1

                results.append(
                    {
                        "case_id": example.case_id,
                        "status": extraction_status,
                        "schema_valid": True,
                        "expected_metadata": example.expected_metadata,
                        "actual_metadata": actual_metadata,
                        "field_matches": {
                            field_name: (
                                _normalize_value(
                                    example.expected_metadata[field_name]
                                )
                                == _normalize_value(
                                    actual_metadata[field_name]
                                )
                            )
                            for field_name in METADATA_FIELDS
                        },
                        "correct_fields": case_field_correct,
                        "total_fields": case_field_total,
                        "exact_record_match": exact_record_match,
                        "citations": citations,
                        "latency_ms": round(latency_ms, 3),
                    }
                )

                print(
                    f"[{index:02d}/{len(examples):02d}] "
                    f"{example.case_id}: "
                    f"status={extraction_status}, "
                    f"fields={case_field_correct}/{case_field_total}, "
                    f"latency={latency_ms:.1f} ms"
                )

            except (
                StructuredExtractionError,
                OllamaClientError,
                TypeError,
                ValueError,
            ) as exc:
                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                latencies.append(latency_ms)

                failure_type = type(exc).__name__
                failure_counts[failure_type] += 1

                results.append(
                    {
                        "case_id": example.case_id,
                        "status": "failed",
                        "schema_valid": False,
                        "error_type": failure_type,
                        "error": str(exc),
                        "latency_ms": round(latency_ms, 3),
                    }
                )

                print(
                    f"[{index:02d}/{len(examples):02d}] "
                    f"{example.case_id}: FAILED ({failure_type})"
                )

    finally:
        await service.aclose()

    total = len(examples)

    per_field_metrics = {
        field_name: {
            "correct": field_correct[field_name],
            "total": field_total[field_name],
            "accuracy": _rate(
                field_correct[field_name],
                field_total[field_name],
            ),
        }
        for field_name in METADATA_FIELDS
    }

    timestamp = datetime.now(UTC)

    return {
        "schema_version": 1,
        "created_at": timestamp.isoformat(),
        "evaluation": "phase3_structured_extraction",
        "schema_name": "TechnicalDocumentMetadata",
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
            "total_examples": total,
        },
        "model": {
            "name": settings.ollama_model,
            "base_url": settings.ollama_base_url,
            "ollama_version": None,
        },
        "configuration": {
            "retrieval_top_k": settings.retrieval_top_k,
            "evidence_threshold": settings.evidence_threshold,
            "max_context_chars": settings.max_context_chars,
            "timeout_seconds": settings.ollama_timeout_seconds,
            "num_ctx": settings.ollama_num_ctx,
            "num_predict": settings.ollama_num_predict,
            "temperature": settings.ollama_temperature,
            "seed": settings.ollama_seed,
            "max_repair_attempts": (
                settings.extraction_max_repair_attempts
            ),
        },
        "metrics": {
            "total_cases": total,
            "completed_cases": completed_records,
            "failed_cases": total - completed_records,
            "completion_rate": _rate(
                completed_records,
                total,
            ),
            "schema_valid_count": schema_valid_records,
            "schema_valid_rate": _rate(
                schema_valid_records,
                total,
            ),
            "field_accuracy": _rate(
                populated_correct,
                populated_total,
            ),
            "field_correct": populated_correct,
            "field_total": populated_total,
            "overall_field_accuracy_including_nulls": _rate(
                sum(field_correct.values()),
                sum(field_total.values()),
            ),
            "null_field_accuracy": _rate(
                null_correct,
                null_total,
            ),
            "null_field_correct": null_correct,
            "null_field_total": null_total,
            "exact_record_accuracy": _rate(
                overall_correct,
                completed_records,
            ),
            "exact_records": overall_correct,
            "per_field": per_field_metrics,
            "citation_page_accuracy": _rate(
                page_matches,
                page_annotations,
            ),
            "citation_pages_correct": page_matches,
            "citation_pages_annotated": page_annotations,
            "status_distribution": dict(status_counts),
            "failure_distribution": dict(failure_counts),
            "median_latency_ms": (
                round(statistics.median(latencies), 3)
                if latencies
                else None
            ),
            "p95_latency_ms": _percentile_95(latencies),
        },
        "results": results,
    }


def build_output_path(model_name: str) -> Path:
    """Build a dated report filename."""

    date_tag = datetime.now(UTC).strftime("%Y%m%d")
    model_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", model_name)

    return (
        PROJECT_ROOT
        / "data"
        / "phase3"
        / f"extraction_eval_{date_tag}_{model_tag}.json"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run DocuLens Phase 3 structured metadata extraction evaluation."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Reviewed extraction evaluation JSONL fixture.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )

    parser.add_argument(
        "--expected-total",
        type=int,
        default=DEFAULT_EXPECTED_TOTAL,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset without sending Ollama requests.",
    )

    return parser.parse_args()


def main() -> int:
    """Run dataset validation and optional model evaluation."""

    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()

    try:
        examples, dataset_hash = load_dataset(dataset_path)

        validate_dataset(
            examples,
            expected_total=args.expected_total,
        )

        print("Extraction dataset validation passed.")
        print(f"Examples: {len(examples)}")
        print(f"SHA-256: {dataset_hash}")

        if args.validate_only:
            print(
                "Validation-only mode: no Ollama requests were sent."
            )
            return 0

        for example in examples:
            resolve_pdf_path(
                example,
                dataset_path=dataset_path,
            )

        report = asyncio.run(
            run_evaluation(
                examples,
                dataset_hash=dataset_hash,
                dataset_path=dataset_path,
            )
        )

        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else build_output_path(report["model"]["name"])
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        metrics = report["metrics"]

        print("\n===== PHASE 3 EXTRACTION RESULTS =====")
        print(
            "Schema-valid rate:",
            metrics["schema_valid_rate"],
        )
        print(
            "Field accuracy:",
            metrics["field_accuracy"],
        )
        print(
            "Null-field accuracy:",
            metrics["null_field_accuracy"],
        )
        print(
            "Exact-record accuracy:",
            metrics["exact_record_accuracy"],
        )
        print(
            "Citation-page accuracy:",
            metrics["citation_page_accuracy"],
        )
        print(
            "Median latency:",
            metrics["median_latency_ms"],
            "ms",
        )
        print(
            "P95 latency:",
            metrics["p95_latency_ms"],
            "ms",
        )
        print(f"\nReport saved to: {output_path}")

        return 0

    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        print(
            f"Extraction evaluation failed: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
