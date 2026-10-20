"""Create a local review pack for the Phase 1 PDF extraction benchmark.

For each in-limit PDF under sample_docs/, select pages near 10%, 50%, and
90% of the physical page sequence. Render source pages and save the current
PyMuPDF text separately for manual comparison. The generated candidate JSON
is not the benchmark manifest and deliberately contains no expected_text.

Run from the repository root:
    uv run python scripts/phase1/create_ingestion_review_pack.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pymupdf

from app.core.config import get_settings
from app.core.paths import DATA_DIR, PROJECT_ROOT, SAMPLE_DOCS_DIR

SCHEMA_VERSION = "1.0"
SAMPLE_FRACTIONS = (0.10, 0.50, 0.90)
RENDER_SCALE = 1.5

OUTPUT_DIR = DATA_DIR / "phase1" / "ingestion_review_pack"
OUTPUT_ZIP = DATA_DIR / "phase1" / "ingestion_review_pack.zip"


def safe_stem(path: Path) -> str:
    """Make a stable filename prefix for generated review artifacts."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return (stem or "document")[:80]


def sample_page_numbers(page_count: int) -> list[int]:
    """Select approximately 10%, 50%, and 90% of a 1-based page sequence."""
    if page_count < 1:
        return []

    selected = {
        max(1, min(page_count, round(page_count * fraction)))
        for fraction in SAMPLE_FRACTIONS
    }
    return sorted(selected)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write review metadata as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inspect_pdf(path: Path, *, max_upload_bytes: int, max_pdf_pages: int) -> dict[str, Any]:
    """Render selected source pages and save extracted text for manual review."""
    relative_pdf_path = path.relative_to(PROJECT_ROOT).as_posix()
    file_size = path.stat().st_size
    result: dict[str, Any] = {
        "path": relative_pdf_path,
        "filename": path.name,
        "file_size_bytes": file_size,
        "status": "pending",
        "page_count": None,
        "encrypted": None,
        "pages_with_text": 0,
        "empty_text_pages": 0,
        "total_extracted_characters": 0,
        "page_extraction_errors": 0,
        "render_errors": 0,
        "sampled_pages": [],
    }

    if file_size == 0:
        result["status"] = "empty_file"
        return result
    if file_size > max_upload_bytes:
        result["status"] = "over_upload_limit"
        return result

    try:
        with pymupdf.open(path) as document:
            result["encrypted"] = bool(document.needs_pass)
            result["page_count"] = document.page_count

            if document.needs_pass:
                result["status"] = "password_protected"
                return result
            if document.page_count < 1:
                result["status"] = "zero_page_pdf"
                return result
            if document.page_count > max_pdf_pages:
                result["status"] = "over_page_limit"
                return result

            stem = safe_stem(path)
            selected_numbers = set(sample_page_numbers(document.page_count))

            for page_index in range(document.page_count):
                page_number = page_index + 1
                is_sampled_page = page_number in selected_numbers

                try:
                    page = document.load_page(page_index)
                    extracted_text = page.get_text("text") or ""
                except (OSError, RuntimeError, ValueError) as exc:
                    result["page_extraction_errors"] += 1
                    if is_sampled_page:
                        result["sampled_pages"].append({
                            "page_number": page_number,
                            "extracted_character_count": None,
                            "extracted_word_count": None,
                            "source_image": None,
                            "extracted_text_file": None,
                            "render_error_type": None,
                            "visual_reviewed": False,
                            "review_notes": "",
                            "expected_text": None,
                            "error_type": type(exc).__name__,
                        })
                    continue

                stripped_text = extracted_text.strip()
                character_count = len(stripped_text)
                result["total_extracted_characters"] += character_count

                if character_count:
                    result["pages_with_text"] += 1
                else:
                    result["empty_text_pages"] += 1

                if not is_sampled_page:
                    continue

                text_relative_path = Path("text") / f"{stem}_page_{page_number:03d}.txt"
                text_path = OUTPUT_DIR / text_relative_path
                text_path.parent.mkdir(parents=True, exist_ok=True)
                text_path.write_text(extracted_text, encoding="utf-8")

                page_record: dict[str, Any] = {
                    "page_number": page_number,
                    "extracted_character_count": character_count,
                    "extracted_word_count": len(stripped_text.split()),
                    "source_image": None,
                    "extracted_text_file": text_relative_path.as_posix(),
                    "render_error_type": None,
                    "visual_reviewed": False,
                    "review_notes": "",
                    "expected_text": None,
                }

                image_relative_path = Path("pages") / f"{stem}_page_{page_number:03d}.png"
                image_path = OUTPUT_DIR / image_relative_path
                image_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(RENDER_SCALE, RENDER_SCALE),
                        alpha=False,
                    )
                    pixmap.save(str(image_path))
                    page_record["source_image"] = image_relative_path.as_posix()
                except (OSError, RuntimeError, ValueError) as exc:
                    result["render_errors"] += 1
                    page_record["render_error_type"] = type(exc).__name__

                result["sampled_pages"].append(page_record)

            if result["page_extraction_errors"]:
                result["status"] = "page_error"
            elif result["total_extracted_characters"] == 0:
                result["status"] = "no_selectable_text"
            else:
                result["status"] = "text_bearing"

    except (OSError, RuntimeError, ValueError) as exc:
        result["status"] = "inspection_error"
        result["error_type"] = type(exc).__name__

    return result


def create_archive(
    output_dir: Path,
    archive_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Archive this run's artifacts, excluding files from earlier runs."""
    relative_files = {"README.txt", "review_candidates.json"}

    for document in manifest["documents"]:
        for page in document.get("sampled_pages", []):
            for key in ("source_image", "extracted_text_file"):
                relative_path = page.get(key)
                if relative_path:
                    relative_files.add(relative_path)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative_path in sorted(relative_files):
            path = output_dir / relative_path
            if path.is_file():
                archive.write(path, relative_path)


def main() -> int:
    """Create and archive the local PDF review pack."""
    if not SAMPLE_DOCS_DIR.is_dir():
        print(f"Sample directory does not exist: {SAMPLE_DOCS_DIR}", file=sys.stderr)
        return 2

    pdf_paths = sorted(
        (
            path
            for path in SAMPLE_DOCS_DIR.iterdir()
            if path.is_file() and not path.is_symlink() and path.suffix.lower() == ".pdf"
        ),
        key=lambda item: item.name.casefold(),
    )
    if not pdf_paths:
        print(f"No PDF samples found in {SAMPLE_DOCS_DIR}", file=sys.stderr)
        return 2

    try:
        settings = get_settings()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Could not load settings: {type(exc).__name__}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    print(f"PDFs: {len(pdf_paths)}")
    print(f"Upload limit: {settings.max_upload_bytes:,} bytes")
    print(f"Page limit: {settings.max_pdf_pages:,}")
    print("Selected physical pages near 10%, 50%, and 90% of each PDF.")

    for path in pdf_paths:
        try:
            result = inspect_pdf(
                path,
                max_upload_bytes=settings.max_upload_bytes,
                max_pdf_pages=settings.max_pdf_pages,
            )
        except OSError as exc:
            result = {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "filename": path.name,
                "status": "file_access_error",
                "error_type": type(exc).__name__,
                "sampled_pages": [],
            }
        results.append(result)
        print(
            f"{result['filename']}: status={result['status']}, "
            f"pages={result.get('page_count')}, "
            f"pages_with_text={result.get('pages_with_text')}, "
            f"page_errors={result.get('page_extraction_errors', 0)}, "
            f"render_errors={result.get('render_errors', 0)}, "
            f"sampled={[item['page_number'] for item in result['sampled_pages']]}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "Manual review candidate list only. This is not the benchmark "
            "manifest and extracted text must not be treated as ground truth."
        ),
        "sampling_fractions": list(SAMPLE_FRACTIONS),
        "render_scale": RENDER_SCALE,
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_pdf_pages": settings.max_pdf_pages,
        },
        "documents": results,
    }
    write_json(OUTPUT_DIR / "review_candidates.json", manifest)

    readme = """# DocuLens Phase 1 manual extraction review pack

This pack contains rendered source pages and the current PyMuPDF extraction for comparison.

IMPORTANT: the extracted text is not ground truth. Do not copy it into `expected_text` without
checking it against the corresponding rendered source page. For each page, transcribe a
sufficiently complete expected body-text reference from the image, record omissions/layout
problems in `review_notes`, and set `visual_reviewed` to true only after review. The
`review_candidates.json` file is not accepted directly by the benchmark runner.

Sampled page positions are approximately 10%, 50%, and 90% of each document's physical page
sequence, deduplicated. Check whether a sampled page is a cover/table of contents and change
sampling if necessary. Documents with no selectable text are included for visual diagnosis but
must remain negative no-text cases for this release because OCR is out of scope.
"""
    (OUTPUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    try:
        create_archive(OUTPUT_DIR, OUTPUT_ZIP, manifest)
    except OSError as exc:
        print(f"Could not create review archive: {type(exc).__name__}", file=sys.stderr)
        return 2

    sampled_page_count = sum(len(item.get("sampled_pages", [])) for item in results)
    print(f"Review files: {OUTPUT_DIR}")
    print(f"Sampled pages: {sampled_page_count}")
    print(f"Review archive: {OUTPUT_ZIP}")
    print("No expected text was generated; manual source-page review is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
