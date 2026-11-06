"""Run the Phase 3 grounded-Q&A evaluation against a reviewed JSONL dataset."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.adapters.llm.ollama_client import OllamaClient
from app.core.config import get_settings
from app.evaluation.qa_metrics import QAEvaluationObservation, calculate_qa_metrics
from app.schemas.answer import AnswerStatus
from app.services.answer_generation import AnswerGenerationService
from app.services.pdf_ingestion import ingest_pdf_upload

DEFAULT_DATASET = PROJECT_ROOT / "tests" / "phase3" / "fixtures" / "qa_eval.jsonl"
DEFAULT_EXPECTED_TOTAL = 30
DEFAULT_EXPECTED_ANSWERABLE = 20
DEFAULT_EXPECTED_UNANSWERABLE = 10


@dataclass(frozen=True, slots=True)
class QAExample:
    """One reviewed question in the evaluation dataset."""

    case_id: str
    question: str
    answerable: bool
    expected_pages: tuple[int, ...]
    review_status: str
    review_note: str
    pdf_path: str | None
    line_number: int


def _first_value(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-null field."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _parse_answerability(row: dict[str, Any], line_number: int) -> bool:
    """Parse an explicit answerable/unanswerable label."""
    value = _first_value(
        row,
        "expected_answerable",
        "answerable",
        "is_answerable",
        "expected_answerability",
        "answerability",
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"answerable", "true", "yes", "1"}:
            return True
        if normalized in {"unanswerable", "false", "no", "0"}:
            return False
    raise ValueError(f"Line {line_number}: missing or invalid answerability label.")


def _parse_expected_pages(
    row: dict[str, Any],
    line_number: int,
) -> tuple[int, ...]:
    """Normalize expected source pages into positive one-based integers."""
    value = _first_value(
        row,
        "expected_pages",
        "relevant_pages",
        "gold_pages",
        "source_pages",
        "expected_page",
        default=[],
    )
    if value is None:
        value = []
    if isinstance(value, int) and not isinstance(value, bool):
        values = [value]
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            values = []
        else:
            try:
                values = [int(part.strip()) for part in stripped.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    f"Line {line_number}: invalid expected page value."
                ) from exc
    elif isinstance(value, list):
        values = value
    else:
        raise TypeError(
            f"Line {line_number}: expected pages must be a list or integer."
        )

    pages: list[int] = []
    for page in values:
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(
                f"Line {line_number}: page numbers must be positive integers."
            )
        pages.append(page)
    return tuple(sorted(set(pages)))


def _parse_example(row: dict[str, Any], line_number: int) -> QAExample:
    """Convert one JSON object into a validated QA example."""
    case_id = _first_value(row, "id", "case_id", "question_id")
    question = row.get("question")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"Line {line_number}: missing case ID.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Line {line_number}: missing question.")

    answerable = _parse_answerability(row, line_number)
    expected_pages = _parse_expected_pages(row, line_number)
    if answerable and not expected_pages:
        raise ValueError(
            f"Line {line_number}: answerable cases require expected pages."
        )
    if not answerable and expected_pages:
        raise ValueError(
            f"Line {line_number}: unanswerable cases must not have expected pages."
        )

    review_status = _first_value(row, "review_status", default="pending_review")
    review_note = _first_value(row, "review_note", "reviewer_note", default="")
    pdf_path = _first_value(row, "pdf_path", "document_path")

    if not isinstance(review_status, str):
        raise TypeError(f"Line {line_number}: review status must be a string.")
    if not isinstance(review_note, str):
        raise TypeError(f"Line {line_number}: review note must be a string.")
    if pdf_path is not None and not isinstance(pdf_path, str):
        raise ValueError(f"Line {line_number}: PDF path must be a string.")

    return QAExample(
        case_id=case_id.strip(),
        question=question.strip(),
        answerable=answerable,
        expected_pages=expected_pages,
        review_status=review_status.strip().lower(),
        review_note=review_note.strip(),
        pdf_path=pdf_path.strip() if isinstance(pdf_path, str) else None,
        line_number=line_number,
    )


def load_dataset(dataset_path: Path) -> tuple[list[QAExample], str]:
    """Load JSONL and return validated examples plus the raw-file SHA-256."""
    path = dataset_path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    examples: list[QAExample] = []
    identifiers: list[str] = []

    for line_number, raw_line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON.") from exc
        if not isinstance(row, dict):
            raise TypeError(
                f"Line {line_number}: each JSONL row must be an object."
            )
        example = _parse_example(row, line_number)
        examples.append(example)
        identifiers.append(example.case_id)

    if not examples:
        raise ValueError("Dataset contains no examples.")

    duplicates = [
        case_id
        for case_id, count in Counter(identifiers).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"Duplicate evaluation IDs: {', '.join(duplicates)}"
        )
    return examples, digest


def validate_dataset(
    examples: list[QAExample],
    *,
    expected_total: int,
    expected_answerable: int,
    expected_unanswerable: int,
) -> None:
    """Enforce dataset size, class balance, and completed source review."""
    if expected_total != expected_answerable + expected_unanswerable:
        raise ValueError("Expected dataset counts are inconsistent.")

    answerable_count = sum(example.answerable for example in examples)
    unanswerable_count = len(examples) - answerable_count
    if len(examples) != expected_total:
        raise ValueError(
            f"Expected {expected_total} examples, found {len(examples)}."
        )
    if answerable_count != expected_answerable:
        raise ValueError(
            f"Expected {expected_answerable} answerable examples, "
            f"found {answerable_count}."
        )
    if unanswerable_count != expected_unanswerable:
        raise ValueError(
            f"Expected {expected_unanswerable} unanswerable examples, "
            f"found {unanswerable_count}."
        )

    pending = [
        example.case_id
        for example in examples
        if example.review_status != "reviewed" or not example.review_note
    ]
    if pending:
        preview = ", ".join(pending[:10])
        raise ValueError(
            "Evaluation blocked because some gold labels are not fully "
            "reviewed. Each row needs review_status='reviewed' and a "
            f"non-empty review_note. Pending IDs: {preview}"
        )


def resolve_pdf_path(
    example: QAExample,
    *,
    default_pdf: Path | None,
    dataset_path: Path,
) -> Path:
    """Resolve a source PDF, preferring project-relative paths."""
    if example.pdf_path:
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
    elif default_pdf is not None:
        path = default_pdf.expanduser()
    else:
        raise ValueError(
            f"Case {example.case_id} has no pdf_path. Pass --pdf or add pdf_path to the dataset."
        )

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"PDF for case {example.case_id} was not found: {path}"
        )
    return path


def build_output_path(model_name: str) -> Path:
    """Build a date-stamped output path with a filesystem-safe model label."""
    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    date_stamp = datetime.now(UTC).strftime("%Y%m%d")
    return PROJECT_ROOT / "data" / "phase3" / f"qa_eval_{date_stamp}_{safe_model}.json"


async def run_evaluation(
    examples: list[QAExample],
    *,
    dataset_hash: str,
    dataset_path: Path,
    default_pdf: Path | None,
) -> dict[str, Any]:
    """Run questions sequentially, reusing ingestion for repeated PDFs."""
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
    service = AnswerGenerationService(
        llm_client=client,
        top_k=settings.retrieval_top_k,
        evidence_threshold=settings.evidence_threshold,
        max_context_chars=settings.max_context_chars,
    )

    ingested_documents: dict[Path, Any] = {}
    results: list[dict[str, Any]] = []
    observations: list[QAEvaluationObservation] = []

    try:
        for example in examples:
            pdf_path = resolve_pdf_path(
                example,
                default_pdf=default_pdf,
                dataset_path=dataset_path,
            )
            if pdf_path not in ingested_documents:
                ingested_documents[pdf_path] = ingest_pdf_upload(
                    pdf_bytes=pdf_path.read_bytes(),
                    filename=pdf_path.name,
                )
            document = ingested_documents[pdf_path]

            started = time.perf_counter()
            response = await service.answer_pages(
                pages=document.pages,
                document_id=str(document.document_id),
                file_name=document.filename,
                question=example.question,
            )
            latency_ms = (time.perf_counter() - started) * 1_000.0
            actual_answered = response.status == AnswerStatus.ANSWERED
            citations = tuple(response.citations)
            cited_pages = tuple(citation.page_number for citation in citations)
            page_matches = sum(
                page in example.expected_pages
                for page in cited_pages
            ) if example.answerable else 0
            observations.append(
                QAEvaluationObservation(
                    expected_answerable=example.answerable,
                    actual_answered=actual_answered,
                    expected_pages=example.expected_pages,
                    cited_pages=cited_pages,
                    latency_ms=latency_ms,
                )
            )
            results.append(
                {
                    "case_id": example.case_id,
                    "question": example.question,
                    "expected_answerable": example.answerable,
                    "actual_answered": actual_answered,
                    "status": response.status.value,
                    "answer": response.answer,
                    "expected_pages": list(example.expected_pages),
                    "cited_pages": list(cited_pages),
                    "citation_count": len(citations),
                    "citation_page_matches": page_matches,
                    "latency_ms": round(latency_ms, 3),
                    "citations": [item.model_dump(mode="json") for item in citations],
                }
            )
    finally:
        await service.aclose()

    metrics = calculate_qa_metrics(observations)
    answerable_count = sum(example.answerable for example in examples)
    return {
        "schema_version": 1,
        "evaluation": "phase3_grounded_qa",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": dataset_hash,
            "total_examples": len(examples),
            "answerable_examples": answerable_count,
            "unanswerable_examples": len(examples) - answerable_count,
        },
        "model": {
            "name": settings.ollama_model,
            "base_url": settings.ollama_base_url,
        },
        "configuration": {
            "retrieval_top_k": settings.retrieval_top_k,
            "evidence_threshold": settings.evidence_threshold,
            "max_context_chars": settings.max_context_chars,
        },
        "metrics": metrics,
        "results": results,
        "limitations": [
            "Answer correctness and claim entailment require separate human review.",
            "Citation-page accuracy measures page matching, not factual support.",
            "Cosine similarity is not a calibrated probability of answer correctness.",
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse dataset, PDF, output and validation-count options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-total", type=int, default=DEFAULT_EXPECTED_TOTAL)
    parser.add_argument(
        "--expected-answerable",
        type=int,
        default=DEFAULT_EXPECTED_ANSWERABLE,
    )
    parser.add_argument(
        "--expected-unanswerable",
        type=int,
        default=DEFAULT_EXPECTED_UNANSWERABLE,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the dataset without sending requests to Ollama.",
    )
    return parser.parse_args()


def main() -> int:
    """Validate the gold set, then optionally run and save the benchmark."""
    args = parse_args()
    try:
        examples, dataset_hash = load_dataset(args.dataset)
        validate_dataset(
            examples,
            expected_total=args.expected_total,
            expected_answerable=args.expected_answerable,
            expected_unanswerable=args.expected_unanswerable,
        )
        if args.validate_only:
            print("QA dataset validation passed.")
            print(f"Examples: {len(examples)}")
            print(f"SHA256: {dataset_hash}")
            print("Validation-only mode: no Ollama requests were sent.")
            return 0

        report = asyncio.run(
            run_evaluation(
                examples,
                dataset_hash=dataset_hash,
                dataset_path=args.dataset,
                default_pdf=args.pdf,
            )
        )
        output_path = args.output or build_output_path(report["model"]["name"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Report saved to: {output_path}")
        print(json.dumps(report["metrics"], indent=2))
        return 0
    except (FileNotFoundError, ValueError, TypeError, OSError) as exc:
        print(f"QA evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
