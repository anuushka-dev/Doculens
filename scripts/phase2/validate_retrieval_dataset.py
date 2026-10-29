"""Validate Phase 2 JSONL questions before running a retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    ROOT / "tests" / "phase2" / "fixtures" / "retrieval_eval.jsonl"
)


class DatasetError(ValueError):
    """The benchmark dataset is not ready or has an invalid shape."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object per non-empty line, reporting its source line."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DatasetError(f"could not read dataset: {path}") from exc

    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"line {line_number}: invalid JSON"
            ) from exc

        if not isinstance(row, dict):
            raise DatasetError(
                f"line {line_number}: each line must be a JSON object"
            )

        rows.append(row)

    return rows


def validate_rows(
    rows: list[dict[str, Any]],
    *,
    require_labelled: bool,
) -> list[str]:
    """Return detected dataset issues without modifying annotations."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        prefix = f"row {index}"

        question_id = row.get("question_id")

        if not isinstance(question_id, str) or not question_id.strip():
            errors.append(
                f"{prefix}: question_id must be a non-blank string"
            )
        else:
            normalized_id = question_id.strip()

            if normalized_id in seen_ids:
                errors.append(
                    f"{prefix}: duplicate question_id {normalized_id!r}"
                )
            else:
                seen_ids.add(normalized_id)

        query = row.get("query")

        if (
            not isinstance(query, str)
            or not query.strip()
            or query.strip().startswith("<")
        ):
            errors.append(f"{prefix}: query is not annotated")

        source = row.get("document_path")

        if not isinstance(source, str) or not source.strip():
            errors.append(f"{prefix}: document_path is required")

        pages = row.get("relevant_page_numbers")

        if not isinstance(pages, list) or any(
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
            for page in pages
        ):
            errors.append(
                f"{prefix}: relevant_page_numbers must be a list "
                "of positive 1-based integers"
            )
        elif len(pages) != len(set(pages)):
            errors.append(
                f"{prefix}: relevant_page_numbers must not contain duplicates"
            )
        elif require_labelled and not pages:
            errors.append(
                f"{prefix}: label expected evidence page(s) before evaluation"
            )

        annotation_status = row.get("annotation_status", "pending")

        if annotation_status not in ("pending", "labelled"):
            errors.append(
                f"{prefix}: annotation_status must be 'pending' or 'labelled'"
            )
        elif require_labelled and annotation_status != "labelled":
            errors.append(
                f"{prefix}: annotation_status must be 'labelled'"
            )

    if require_labelled and len(rows) != 30:
        errors.append(
            f"expected exactly 30 labelled questions; found {len(rows)}"
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate the dataset and optionally require the full labelled set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--require-labelled", action="store_true")
    args = parser.parse_args(argv)

    dataset_path = (
        args.dataset
        if args.dataset.is_absolute()
        else ROOT / args.dataset
    )

    try:
        rows = load_jsonl(dataset_path)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    errors = validate_rows(
        rows,
        require_labelled=args.require_labelled,
    )
    pending = sum(
        row.get("annotation_status", "pending") != "labelled"
        for row in rows
    )

    print(f"Questions: {len(rows)} | pending labels: {pending}")

    if errors:
        for error in errors:
            print(f"- {error}")
        return 1

    print("Dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())