"""Run the Phase 2 top-five retrieval benchmark against labelled JSONL queries."""

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.adapters.embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from app.evaluation.retrieval_metrics import (
    QueryEvaluationResult,
    score_query,
    summarize_evaluation,
)
from app.services.pdf_ingestion import ingest_pdf_upload
from app.services.retrieval_pipeline import RetrievalPipeline
from scripts.phase2.validate_retrieval_dataset import (
    DatasetError,
    load_jsonl,
    validate_rows,
)

TOP_K = 5
TARGET_RECALL_AT_5 = 0.80
DEFAULT_DATASET = ROOT / "tests" / "phase2" / "fixtures" / "retrieval_eval.jsonl"
DEFAULT_REPORT = ROOT / "data" / "phase2" / "retrieval_eval_report.json"


MISS_ANNOTATIONS: dict[tuple[str, str], tuple[str, str]] = {
    (
        "Q002",
        "Which three main areas do these guidelines cover?",
    ): (
        "embedding",
        (
            "The answer-bearing abstract chunk on physical page 5 is indexed "
            "but ranks 82. A different chunk on page 5 ranks 34, while the "
            "abstract answer is absent from the top five. Dense retrieval "
            "under-ranks the direct answer passage."
        ),
    ),
    (
        "Q012",
        "What must an RP include in its minimum online-service description?",
    ): (
        "embedding",
        (
            "Physical page 36 has indexed chunks, but its highest-ranked "
            "chunk is rank 6, just outside the top five. Related passages "
            "from pages 33, 55, and 32 rank higher."
        ),
    ),
    (
        "Q017",
        "What must an organization document when tailoring assurance levels?",
    ): (
        "embedding",
        (
            "The requirements-bearing chunk on physical page 52 is indexed "
            "but ranks 26; the other chunk on that page ranks 24. Related "
            "assurance-level sections outrank the specific documentation "
            "requirements."
        ),
    ),
    (
        "Q021",
        "What is the title of NIST SP 800-37r2 in the references?",
    ): (
        "embedding",
        (
            "The reference chunk on physical page 65 is indexed and contains "
            "the requested title, but ranks 11. Front matter ranks above "
            "the bibliographic evidence for this exact-identifier query."
        ),
    ),
    (
        "Q024",
        "What is back-channel communication?",
    ): (
        "embedding",
        (
            "The exact definition chunk on physical page 77 is indexed but "
            "ranks 20. Another, non-answer-bearing chunk on the same page "
            "ranks 6, while nearby glossary entries occupy the top five."
        ),
    ),
}


def resolve_pdf(raw_path: str) -> Path:
    """Resolve a repository-local PDF while rejecting traversal and symlinks."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("dataset document_path must be a non-blank string")

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = Path(os.path.abspath(candidate))

    try:
        candidate.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(
            "dataset PDF paths must remain inside the repository"
        ) from exc

    current = candidate
    while current != ROOT:
        if current.is_symlink():
            raise ValueError("dataset PDF paths must not traverse symlinks")
        current = current.parent

    if not candidate.is_file() or candidate.suffix.casefold() != ".pdf":
        raise ValueError(f"dataset PDF path is not a PDF file: {raw_path}")

    return candidate


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write the report atomically, avoiding partially written JSON."""
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
                allow_nan=False,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def run_benchmark(
    *,
    dataset_path: Path,
    report_path: Path,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_revision: str | None = None,
    device: str = "cpu",
    index_path: Path | None = None,
    extraction_fn: Callable[..., Any] | None = None,
    embedder_factory: Callable[..., Any] | None = None,
    pipeline_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run a validated benchmark and write a report containing ranked evidence.

    The existing Phase 1 ingestion service extracts each unique source PDF.
    The stable benchmark document ID is derived from its repository-relative
    path and content hash, not the per-ingestion UUID.
    """
    if (
        isinstance(chunk_size_words, bool)
        or not isinstance(chunk_size_words, int)
        or chunk_size_words < 1
    ):
        raise ValueError("chunk_size_words must be a positive integer")

    if isinstance(overlap_words, bool) or not isinstance(overlap_words, int):
        raise TypeError("overlap_words must be an integer")

    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be non-negative and smaller than chunk_size_words"
        )

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must not be blank")

    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision.strip()
    ):
        raise ValueError("model_revision must be None or a non-blank string")

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must not be blank")

    dataset_path = dataset_path.resolve()
    report_path = report_path.resolve()

    try:
        rows = load_jsonl(dataset_path)
    except DatasetError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise DatasetError(f"could not read dataset: {dataset_path}") from exc

    errors = validate_rows(rows, require_labelled=True)
    if errors:
        raise DatasetError(
            "dataset is not ready. Fix these items before evaluating:\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\nRun scripts/phase2/validate_retrieval_dataset.py "
            "to inspect the annotation checklist."
        )

    selected_extractor = extraction_fn or ingest_pdf_upload
    selected_embedder_factory = embedder_factory or SentenceTransformerEmbedder
    selected_pipeline_factory = pipeline_factory or RetrievalPipeline

    # Extract each unique PDF once. A canonical repository-relative path is the
    # lookup key so alternate spellings cannot create duplicate index entries.
    documents: dict[str, Any] = {}
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    benchmark_document_ids: dict[str, str] = {}
    question_sources: dict[str, str] = {}

    for row in rows:
        raw_source = row["document_path"]
        pdf_path = resolve_pdf(raw_source)
        source = pdf_path.relative_to(ROOT).as_posix()
        question_sources[row["question_id"]] = source

        if source in documents:
            continue

        try:
            pdf_bytes = pdf_path.read_bytes()
            source_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
            benchmark_identity = hashlib.sha256(
                (source + "\0" + source_sha256).encode("utf-8")
            ).hexdigest()[:32]
            benchmark_document_id = f"benchmark-{benchmark_identity}"

            extracted_document = selected_extractor(
                pdf_bytes=pdf_bytes,
                filename=pdf_path.name,
            )
        except Exception as exc:
            raise RuntimeError(
                f"could not extract {source!r}: {type(exc).__name__}: {exc}"
            ) from exc

        source_hashes[source] = source_sha256
        benchmark_document_ids[source] = benchmark_document_id
        documents[source] = extracted_document
        source_paths[source] = pdf_path

    # Reject labels beyond the actual PDF page count before model loading.
    for row in rows:
        source = question_sources[row["question_id"]]
        document = documents[source]
        invalid_pages = [
            page
            for page in row["relevant_page_numbers"]
            if page > document.page_count
        ]

        if invalid_pages:
            raise ValueError(
                f"{row['question_id']} labels page(s) {invalid_pages} "
                f"outside the {document.page_count}-page source PDF "
                f"{source!r}"
            )

    protected_inputs = {dataset_path, *source_paths.values()}
    if report_path in protected_inputs:
        raise ValueError(
            "report path must not overwrite the dataset or a source PDF"
        )

    if index_path is not None:
        index_path = index_path.resolve()
        if index_path in protected_inputs:
            raise ValueError(
                "index path must not point to the dataset or a source PDF"
            )
        if index_path == report_path:
            raise ValueError("report path and index path must be different")

    try:
        embedder = selected_embedder_factory(
            model_name=model_name.strip(),
            revision=model_revision.strip() if model_revision else None,
            device=device.strip(),
        )
        pipeline = selected_pipeline_factory(
            embedder=embedder,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            top_k=TOP_K,
        )

        chunk_count = 0
        for source, document in documents.items():
            chunks = pipeline.index_pages(
                document.pages,
                document_id=benchmark_document_ids[source],
                file_name=document.filename,
            )
            chunk_count += len(chunks)

        if chunk_count < 1:
            raise ValueError("no non-empty chunks were produced")

        if index_path is not None:
            pipeline.save(index_path)

    except Exception as exc:
        raise RuntimeError(
            f"retrieval index build failed: {type(exc).__name__}: {exc}"
        ) from exc

    query_results: list[dict[str, Any]] = []
    scored_queries: list[QueryEvaluationResult] = []

    for row in rows:
        started = time.perf_counter()

        try:
            response = pipeline.search(row["query"], top_k=TOP_K)
        except Exception as exc:
            raise RuntimeError(
                f"query {row['question_id']} failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        source = question_sources[row["question_id"]]
        target_document_id = benchmark_document_ids[source]

        top_results = response.results[:TOP_K]
        hit_under_page_metric = any(
            item.document_id == target_document_id
            and item.page_number in row["relevant_page_numbers"]
            for item in top_results
        )
        miss_annotation = (
            MISS_ANNOTATIONS.get((row["question_id"], row["query"]))
            if not hit_under_page_metric
            else None
        )

        scored = score_query(
            query_id=row["question_id"],
            relevant_document_id=target_document_id,
            relevant_page_numbers=row["relevant_page_numbers"],
            retrieved_results=response.results,
            latency_ms=elapsed_ms,
            k=TOP_K,
            miss_category=(
                miss_annotation[0] if miss_annotation is not None else None
            ),
            miss_diagnostic=(
                miss_annotation[1] if miss_annotation is not None else None
            ),
        )
        scored_queries.append(scored)

        retrieved_evidence = [
            {
                "rank": rank,
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "file_name": item.file_name,
                "page_number": item.page_number,
                "chunk_index": item.chunk_index,
                "start_char": item.start_char,
                "end_char": item.end_char,
                "score": round(float(item.score), 6),
                "text": item.text,
            }
            for rank, item in enumerate(response.results, start=1)
        ]

        query_results.append(
            {
                "question_id": row["question_id"],
                "query": row["query"],
                "document_path": source,
                "relevant_page_numbers": row["relevant_page_numbers"],
                "annotation_note": row.get("annotation_note"),
                "evaluation": scored.to_dict(),
                "retrieved_evidence": retrieved_evidence,
                "hit_at_5": scored.hit,
                "latency_ms": round(elapsed_ms, 3),
                "failure_category": (
                    scored.miss_category.value
                    if scored.miss_category is not None
                    else ("unclassified" if not scored.hit else None)
                ),
                "failure_diagnostic": scored.miss_diagnostic,
            }
        )

    metrics_report = summarize_evaluation(scored_queries)
    metrics = metrics_report.to_dict()
    metrics.pop("per_query", None)

    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    report = {
        "schema_version": "1.1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dataset_path": (
            dataset_path.relative_to(ROOT).as_posix()
            if dataset_path.is_relative_to(ROOT)
            else str(dataset_path)
        ),
        "dataset_sha256": dataset_sha256,
        "source_sha256": source_hashes,
        "source_documents": [
            {
                "path": source,
                "sha256": source_hashes[source],
                "benchmark_document_id": benchmark_document_ids[source],
                "page_count": document.page_count,
                "empty_page_numbers": list(document.empty_page_numbers),
                "near_empty_page_numbers": list(document.near_empty_page_numbers),
            }
            for source, document in documents.items()
        ],
        "chunking": {
            "unit": "words",
            "chunk_size_words": chunk_size_words,
            "overlap_words": overlap_words,
            "page_local": True,
        },
        "embedding": {
            "model_name": embedder.model_name,
            "model_revision": model_revision or "not-pinned",
            "embedding_dimension": embedder.embedding_dimension,
            "device": device,
        },
        "index": {
            "type": "FAISS IndexFlatIP",
            "metric": "cosine similarity via L2-normalized vectors",
            "chunk_count": chunk_count,
            "saved_index_path": str(index_path) if index_path is not None else None,
        },
        "metric_definition": (
            "Query-level Recall@5: a hit requires a retrieved chunk whose "
            "document ID matches the labelled source PDF and whose one-based "
            "page number is in relevant_page_numbers."
        ),
        "metrics": metrics,
        "target_recall_at_5": TARGET_RECALL_AT_5,
        "target_met": metrics_report.recall_at_k >= TARGET_RECALL_AT_5,
        "queries": query_results,
        "limitations": [
            "Recall@5 evaluates page-level retrieval labels, not generated-answer correctness.",
            (
                "Latency includes query embedding and FAISS search but excludes "
                "PDF parsing and initial corpus embedding."
            ),
            (
                "Every miss must be inspected; the runner deliberately does not "
                "guess a failure category."
            ),
            (
                "If model_revision is not pinned, results may not be reproducible "
                "across later model downloads."
            ),
        ],
    }

    write_json_atomically(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the labelled Phase 2 retrieval benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunk-size-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index-path", type=Path, default=None)
    args = parser.parse_args(argv)

    dataset_path = (
        args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    )
    report_path = (
        args.report if args.report.is_absolute() else ROOT / args.report
    )
    index_path = (
        args.index_path
        if args.index_path is None or args.index_path.is_absolute()
        else ROOT / args.index_path
    )

    try:
        report = run_benchmark(
            dataset_path=dataset_path,
            report_path=report_path,
            chunk_size_words=args.chunk_size_words,
            overlap_words=args.overlap_words,
            model_name=args.model_name,
            model_revision=args.model_revision,
            device=args.device,
            index_path=index_path,
        )
    except (DatasetError, ValueError, OSError, RuntimeError) as exc:
        print(
            f"ERROR: benchmark failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    metrics = report["metrics"]
    print(
        f"Questions: {metrics['query_count']}; hits: {metrics['hits']}; "
        f"misses: {metrics['misses']}"
    )
    print(
        f"Recall@5: {metrics['recall_at_k_percent']:.2f}% "
        f"(target: 80.00%; met: {report['target_met']})"
    )
    print(
        f"Median query latency: {metrics['median_latency_ms']:.3f} ms; "
        f"p95: {metrics['p95_latency_ms']:.3f} ms"
    )
    print(f"Report: {report_path}")
    if metrics.get("miss_classification_complete", False):
        print(
            "All retrieval misses have a failure category and diagnostic. "
            "Review remaining dataset annotations before declaring Phase 2 complete."
        )
    else:
        unclassified_ids = metrics.get("unclassified_miss_ids", [])
        if unclassified_ids:
            print(
                "Unclassified retrieval misses: "
                + ", ".join(unclassified_ids)
            )
        print(
            "Inspect every miss and assign a failure category before declaring "
            "Phase 2 complete."
        )

    if args.model_revision is None:
        print(
            "WARNING: no model revision was pinned; record an exact revision "
            "for a reproducible baseline.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())