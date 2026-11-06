"""Validate the Phase 3 Q&A JSONL fixture without calling Ollama."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase3.run_qa_eval import (
    DEFAULT_DATASET,
    DEFAULT_EXPECTED_ANSWERABLE,
    DEFAULT_EXPECTED_TOTAL,
    DEFAULT_EXPECTED_UNANSWERABLE,
    load_dataset,
    validate_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
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
        "--require-reviewed",
        action="store_true",
        help=(
            "Require review_status='reviewed' and a non-empty "
            "review note on every row."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()

    try:
        examples, dataset_hash = load_dataset(dataset_path)

        if args.expected_total != (
            args.expected_answerable + args.expected_unanswerable
        ):
            raise ValueError("Expected dataset counts are inconsistent.")

        answerable_count = sum(example.answerable for example in examples)
        unanswerable_count = len(examples) - answerable_count

        if len(examples) != args.expected_total:
            raise ValueError(
                f"Expected {args.expected_total} examples, found {len(examples)}."
            )
        if answerable_count != args.expected_answerable:
            raise ValueError(
                f"Expected {args.expected_answerable} answerable examples, "
                f"found {answerable_count}."
            )
        if unanswerable_count != args.expected_unanswerable:
            raise ValueError(
                f"Expected {args.expected_unanswerable} unanswerable examples, "
                f"found {unanswerable_count}."
            )

        if args.require_reviewed:
            validate_dataset(
                examples,
                expected_total=args.expected_total,
                expected_answerable=args.expected_answerable,
                expected_unanswerable=args.expected_unanswerable,
            )

        pending = [
            item.case_id
            for item in examples
            if item.review_status != "reviewed" or not item.review_note
        ]

        print("Q&A dataset structure is valid.")
        print(f"Examples: {len(examples)}")
        print(f"Answerable: {answerable_count}")
        print(f"Unanswerable: {unanswerable_count}")
        print(f"SHA-256: {dataset_hash}")

        if pending:
            print(f"Pending review: {len(pending)} ({', '.join(pending[:10])})")
            if not args.require_reviewed:
                print(
                    "Structural validation only. "
                    "Benchmark execution remains blocked."
                )
        else:
            print("All gold labels have review notes.")

        return 0

    except (TypeError, ValueError, OSError) as exc:
        print(f"Q&A dataset validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
