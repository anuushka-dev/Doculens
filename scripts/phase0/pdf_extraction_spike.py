from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pymupdf

from app.core.config import get_settings
from app.core.paths import DATA_DIR, PROJECT_ROOT, SAMPLE_DOCS_DIR

REPORT_SCHEMA_VERSION = 1
MINIMUM_SAMPLE_COUNT = 5
NEAR_EMPTY_CHAR_THRESHOLD = 20
HASH_BLOCK_SIZE = 1024 * 1024


def resolve_project_path(path: Path) -> Path:
    """Resolve relative paths from the repository root."""
    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def calculate_sha256(path: Path) -> str:
    """Calculate a file digest without loading the entire file into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(HASH_BLOCK_SIZE),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def inspect_pdf(
    path: Path,
    *,
    max_upload_bytes: int,
    max_pdf_pages: int,
) -> dict[str, Any]:
    """Collect PDF extraction diagnostics without storing extracted text."""
    started_at = perf_counter()

    result: dict[str, Any] = {
        "filename": path.name,
        "status": "pending",
        "file_size_bytes": None,
        "sha256": None,
        "page_count": None,
        "total_text_characters": 0,
        "empty_text_pages": 0,
        "near_empty_text_pages": 0,
        "page_extraction_errors": 0,
        "pages": [],
        "elapsed_ms": None,
        "error_type": None,
    }

    try:
        if path.is_symlink():
            result["status"] = "skipped_symlink"
            return result

        if not path.is_file():
            result["status"] = "not_a_regular_file"
            return result

        file_size = path.stat().st_size
        result["file_size_bytes"] = file_size

        if file_size > max_upload_bytes:
            result["status"] = "skipped_file_size_limit"
            return result

        result["sha256"] = calculate_sha256(path)

        with pymupdf.open(path) as document:
            page_count = document.page_count
            result["page_count"] = page_count

            if document.needs_pass:
                result["status"] = "password_protected"
                return result

            if page_count > max_pdf_pages:
                result["status"] = "skipped_page_limit"
                return result

            if page_count == 0:
                result["status"] = "zero_page_pdf"
                return result

            for page_index in range(page_count):
                page_number = page_index + 1

                try:
                    page = document.load_page(page_index)
                    extracted_text = page.get_text("text") or ""
                except (RuntimeError, ValueError) as exc:
                    result["page_extraction_errors"] += 1
                    result["pages"].append(
                        {
                            "page_number": page_number,
                            "status": "extraction_error",
                            "error_type": type(exc).__name__,
                            "character_count": None,
                            "word_count": None,
                        }
                    )
                    continue

                cleaned_text = extracted_text.strip()
                character_count = len(cleaned_text)
                word_count = len(cleaned_text.split())

                if character_count == 0:
                    page_status = "empty_text"
                    result["empty_text_pages"] += 1
                elif character_count < NEAR_EMPTY_CHAR_THRESHOLD:
                    page_status = "near_empty_text"
                    result["near_empty_text_pages"] += 1
                else:
                    page_status = "ok"

                result["total_text_characters"] += character_count

                result["pages"].append(
                    {
                        "page_number": page_number,
                        "status": page_status,
                        "character_count": character_count,
                        "word_count": word_count,
                    }
                )

            if result["page_extraction_errors"]:
                result["status"] = "partial_page_error"
            elif result["total_text_characters"] == 0:
                result["status"] = "no_extractable_text"
            else:
                result["status"] = "ok"

    except (OSError, RuntimeError, ValueError) as exc:
        # Record an error category, not exception text that might expose paths.
        result["status"] = "open_or_read_error"
        result["error_type"] = type(exc).__name__

    finally:
        result["elapsed_ms"] = round(
            (perf_counter() - started_at) * 1000,
            2,
        )

    return result


def build_report(
    sample_dir: Path,
    documents: list[dict[str, Any]],
    *,
    max_upload_bytes: int,
    max_pdf_pages: int,
) -> dict[str, Any]:
    """Build a reproducible report containing diagnostic metadata only."""
    regular_pdf_count = sum(
        1
        for document in documents
        if document["status"] != "skipped_symlink"
        and document["status"] != "not_a_regular_file"
    )

    status_counts: dict[str, int] = {}

    for document in documents:
        status = document["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    try:
        sample_dir_display = sample_dir.relative_to(
            PROJECT_ROOT
        ).as_posix()
    except ValueError:
        sample_dir_display = "<external-sample-directory>"

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "pymupdf_version": getattr(
            pymupdf,
            "__version__",
            "unknown",
        ),
        "sample_directory": sample_dir_display,
        "limits": {
            "max_upload_bytes": max_upload_bytes,
            "max_pdf_pages": max_pdf_pages,
            "near_empty_char_threshold": NEAR_EMPTY_CHAR_THRESHOLD,
        },
        "summary": {
            "pdf_candidates": len(documents),
            "regular_pdf_count": regular_pdf_count,
            "minimum_sample_target": MINIMUM_SAMPLE_COUNT,
            "minimum_sample_target_met": (
                regular_pdf_count >= MINIMUM_SAMPLE_COUNT
            ),
            "status_counts": status_counts,
            "total_pages_examined": sum(
                len(document["pages"])
                for document in documents
            ),
            "total_text_characters": sum(
                document["total_text_characters"]
                for document in documents
            ),
            "pages_with_empty_text": sum(
                document["empty_text_pages"]
                for document in documents
            ),
            "pages_with_near_empty_text": sum(
                document["near_empty_text_pages"]
                for document in documents
            ),
        },
        "manual_visual_audit_required": True,
        "documents": documents,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Write JSON atomically to avoid leaving a partially written report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv: list[str] | None = None) -> int:
    """Run the bounded PDF extraction feasibility experiment."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure PDF extraction diagnostics for public sample PDFs."
        )
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=SAMPLE_DOCS_DIR,
        help="Directory containing PDF samples.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DATA_DIR / "phase0" / "pdf_extraction_report.json",
        help="Path for the JSON report.",
    )

    args = parser.parse_args(argv)

    sample_dir = resolve_project_path(args.sample_dir)
    report_path = resolve_project_path(args.report)

    if not sample_dir.is_dir():
        print(
            "Error: sample directory does not exist.",
            file=sys.stderr,
        )
        return 2

    try:
        candidates = sorted(
            (
                path
                for path in sample_dir.iterdir()
                if path.suffix.lower() == ".pdf"
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        print(
            "Error: could not list the sample directory.",
            file=sys.stderr,
        )
        return 2

    if not candidates:
        print(
            "No PDFs found. Add at least five public technical PDFs "
            "to sample_docs/ before running the experiment.",
            file=sys.stderr,
        )
        return 2

    settings = get_settings()

    documents = [
        inspect_pdf(
            path,
            max_upload_bytes=settings.max_upload_bytes,
            max_pdf_pages=settings.max_pdf_pages,
        )
        for path in candidates
    ]

    report = build_report(
        sample_dir,
        documents,
        max_upload_bytes=settings.max_upload_bytes,
        max_pdf_pages=settings.max_pdf_pages,
    )

    try:
        write_report(report_path, report)
    except OSError:
        print(
            "Error: could not write the extraction report.",
            file=sys.stderr,
        )
        return 1

    for document in documents:
        print(
            f"{document['filename']}: "
            f"status={document['status']}, "
            f"pages={document['page_count']}, "
            f"characters={document['total_text_characters']}, "
            f"elapsed_ms={document['elapsed_ms']}"
        )

    print(
        "Minimum sample target met:",
        report["summary"]["minimum_sample_target_met"],
    )
    print(f"Report written to: {report_path}")

    print(
        "Reminder: inspect representative PDF pages visually. "
        "Character counts alone do not establish extraction accuracy."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())