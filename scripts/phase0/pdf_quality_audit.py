
"""Render and inspect representative PDF pages for Phase 0 F1.

The JSON report contains metadata only. Text previews are printed to the
terminal for local review and are not persisted in the report.
"""

import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pymupdf

from app.core.config import get_settings
from app.core.paths import DATA_DIR, PROJECT_ROOT, SAMPLE_DOCS_DIR

SCHEMA_VERSION = "1.0"
NEAR_EMPTY_CHAR_THRESHOLD = 20
PREVIEW_CHAR_LIMIT = 400
RENDER_SCALE = 1.4


def calculate_sha256(path: Path) -> str:
    """Calculate a file fingerprint without loading the whole PDF in memory."""
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def safe_directory_name(filename: str) -> str:
    """Create a filesystem-safe directory name from a sample filename."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem)
    return name.strip("._") or "document"


def choose_sample_pages(page_count: int) -> list[int]:
    """Select one-based first, middle, fourth, and last pages where available."""
    if page_count < 1:
        return []

    pages = {
        1,
        (page_count + 1) // 2,
        page_count,
    }

    if page_count >= 4:
        pages.add(4)

    return sorted(pages)


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write the report without leaving a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".pdf-quality-audit-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(payload, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")

        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def inspect_pdf(
    pdf_path: Path,
    *,
    max_upload_bytes: int,
    max_pdf_pages: int,
    image_root: Path,
) -> dict[str, Any]:
    """Inspect page statistics and render a small set of pages for review."""
    filename = pdf_path.name

    result: dict[str, Any] = {
        "filename": filename,
        "status": "not_inspected",
        "file_size_bytes": None,
        "sha256": None,
        "page_count": None,
        "total_text_characters": 0,
        "empty_text_pages": [],
        "near_empty_text_pages": [],
        "page_extraction_errors": [],
        "sampled_pages": [],
        "manual_visual_review_required": True,
    }

    try:
        file_size = pdf_path.stat().st_size
    except OSError as exc:
        result.update(
            status="file_access_error",
            error_type=type(exc).__name__,
        )
        return result

    result["file_size_bytes"] = file_size

    if file_size == 0:
        result["status"] = "empty_file"
        return result

    if file_size > max_upload_bytes:
        result["status"] = "over_upload_limit"
        result["limit_bytes"] = max_upload_bytes
        return result

    try:
        result["sha256"] = calculate_sha256(pdf_path)
    except OSError as exc:
        result.update(
            status="file_access_error",
            error_type=type(exc).__name__,
        )
        return result

    document: pymupdf.Document | None = None

    try:
        try:
            document = pymupdf.open(pdf_path)
        except (OSError, RuntimeError, ValueError) as exc:
            result.update(
                status="open_error",
                error_type=type(exc).__name__,
            )
            return result

        if document.needs_pass:
            result["status"] = "encrypted"
            return result

        page_count = document.page_count
        result["page_count"] = page_count

        if page_count < 1:
            result["status"] = "zero_page_document"
            return result

        if page_count > max_pdf_pages:
            result.update(
                status="over_page_limit",
                limit_pages=max_pdf_pages,
            )
            return result

        sample_pages = set(choose_sample_pages(page_count))
        document_image_root = image_root / safe_directory_name(filename)
        page_metadata: dict[int, dict[str, Any]] = {}
        total_characters = 0
        empty_pages: list[int] = []
        near_empty_pages: list[int] = []
        extraction_errors: list[dict[str, Any]] = []

        for page_index in range(page_count):
            page_number = page_index + 1

            try:
                page = document.load_page(page_index)
                text = page.get_text("text") or ""
                character_count = len(text.strip())
                word_count = len(text.split())

                total_characters += character_count

                if character_count == 0:
                    empty_pages.append(page_number)
                elif character_count < NEAR_EMPTY_CHAR_THRESHOLD:
                    near_empty_pages.append(page_number)

                page_metadata[page_number] = {
                    "character_count": character_count,
                    "word_count": word_count,
                    "text_preview": text[:PREVIEW_CHAR_LIMIT],
                    "text_error": None,
                    "render_error": None,
                }

            except (OSError, RuntimeError, ValueError) as exc:
                extraction_errors.append(
                    {
                        "page_number": page_number,
                        "error_type": type(exc).__name__,
                    }
                )
                page_metadata[page_number] = {
                    "character_count": None,
                    "word_count": None,
                    "text_preview": "",
                    "text_error": type(exc).__name__,
                    "render_error": None,
                }

        sampled_pages: list[dict[str, Any]] = []

        for page_number in sorted(sample_pages):
            metadata = page_metadata[page_number]
            image_path = (
                document_image_root / f"page_{page_number:03d}.png"
            )

            try:
                page = document.load_page(page_number - 1)
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE),
                    alpha=False,
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                pixmap.save(str(image_path))
                relative_image_path = str(
                    image_path.relative_to(PROJECT_ROOT)
                )
            except (OSError, RuntimeError, ValueError) as exc:
                relative_image_path = None
                metadata["render_error"] = type(exc).__name__

            sampled_pages.append(
                {
                    "page_number": page_number,
                    "character_count": metadata["character_count"],
                    "word_count": metadata["word_count"],
                    "image_path": relative_image_path,
                    "text_error": metadata["text_error"],
                    "render_error": metadata["render_error"],
                }
            )

            print(f"\n  Page {page_number}")
            print(f"    Text characters: {metadata['character_count']}")
            print(f"    Word count: {metadata['word_count']}")

            if metadata["text_error"]:
                print(f"    Extraction error: {metadata['text_error']}")
            elif metadata["text_preview"].strip():
                print(
                    "    Preview (terminal only): "
                    f"{metadata['text_preview']!r}"
                )
            else:
                print("    Preview: NO EXTRACTABLE TEXT")

            if relative_image_path:
                print(f"    Rendered page: {relative_image_path}")
            elif metadata["render_error"]:
                print(f"    Render error: {metadata['render_error']}")

        result.update(
            total_text_characters=total_characters,
            empty_text_pages=empty_pages,
            near_empty_text_pages=near_empty_pages,
            page_extraction_errors=extraction_errors,
            sampled_pages=sampled_pages,
            status=(
                "extraction_error"
                if extraction_errors and total_characters == 0
                else "partial_extraction_error"
                if extraction_errors
                else "no_extractable_text"
                if total_characters == 0
                else "ok"
            ),
        )

    except (OSError, RuntimeError, ValueError) as exc:
        result.update(
            status="inspection_error",
            error_type=type(exc).__name__,
        )
    finally:
        if document is not None:
            document.close()

    return result


def main() -> int:
    """Run the local Phase 0 PDF quality audit."""
    settings = get_settings()

    if not SAMPLE_DOCS_DIR.is_dir():
        print(
            f"Sample directory does not exist: {SAMPLE_DOCS_DIR}",
            file=sys.stderr,
        )
        return 1

    pdf_paths = sorted(
        (
            path
            for path in SAMPLE_DOCS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    )

    if not pdf_paths:
        print(
            f"No PDF samples found in {SAMPLE_DOCS_DIR}",
            file=sys.stderr,
        )
        return 1

    image_root = DATA_DIR / "phase0" / "quality_audit"
    report_path = DATA_DIR / "phase0" / "pdf_quality_audit.json"

    print(f"PDF samples found: {len(pdf_paths)}")
    print(f"Sample directory: {SAMPLE_DOCS_DIR}")
    print("Selected pages: first, middle, fourth when available, and last")
    print("Text previews will not be written to the JSON report.")

    results: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        print(f"\n{'=' * 72}")
        print(f"Document: {pdf_path.name}")

        result = inspect_pdf(
            pdf_path,
            max_upload_bytes=settings.max_upload_bytes,
            max_pdf_pages=settings.max_pdf_pages,
            image_root=image_root,
        )

        print(
            "  Summary: "
            f"status={result['status']}, "
            f"pages={result['page_count']}, "
            f"characters={result['total_text_characters']}"
        )
        results.append(result)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "Phase 0 F1 manual-assisted PDF quality audit",
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_pdf_pages": settings.max_pdf_pages,
        },
        "sample_count": len(results),
        "documents": results,
        "manual_review_complete": False,
    }

    write_json_atomically(report_path, report)

    print(f"\n{'=' * 72}")
    print(f"Audit report: {report_path}")
    print(f"Rendered sample pages: {image_root}")
    print(
        "Manual review still required. The report does not claim "
        "that extraction accuracy passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
