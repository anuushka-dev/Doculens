"""Compare dense, BM25, equal-RRF, and weighted-RRF retrieval on one fixture.

The fixture supplies reviewed relevant PDF page numbers, not gold chunk IDs or
answer spans. Consequently this runner reports page-level retrieval metrics and
explicitly does not claim answer-bearing-chunk recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.adapters.embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
)
from app.schemas.retrieval import ChunkRecord, RetrievedChunk
from app.services.hybrid_retrieval import HybridRetrievalPipeline
from app.services.pdf_ingestion import ingest_pdf_upload
from scripts.phase2.validate_retrieval_dataset import (
    DatasetError,
    load_jsonl,
    validate_rows,
)

DEFAULT_DATASET = (
    PROJECT_ROOT / "tests" / "phase2" / "fixtures" / "retrieval_eval.jsonl"
)
DEFAULT_REPORT = PROJECT_ROOT / "data" / "phase3" / "retrieval_comparison_report.json"
DEFAULT_TOP_K_VALUES = (5, 10)
DEFAULT_DENSE_CANDIDATE_K = 20
DEFAULT_BM25_CANDIDATE_K = 20
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """One source chunk with independent retrieval/fusion metadata."""

    chunk: RetrievedChunk
    dense_rank: int | None
    bm25_rank: int | None
    bm25_score: float
    rrf_score: float | None


def resolve_pdf(raw_path: str) -> Path:
    """Resolve a repository-local PDF while rejecting traversal and symlinks."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("dataset document_path must be a non-blank string")

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = Path(os.path.abspath(candidate))

    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("dataset PDF paths must remain inside the repository") from exc

    current = candidate
    while current != PROJECT_ROOT:
        if current.is_symlink():
            raise ValueError("dataset PDF paths must not traverse symlinks")
        current = current.parent

    if not candidate.is_file() or candidate.suffix.casefold() != ".pdf":
        raise ValueError(f"dataset PDF path is not a PDF file: {raw_path}")
    return candidate


def _positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not (result > 0.0 and result < float("inf")):
        raise ValueError(f"{name} must be finite and positive")
    return result


def fuse_rankings(
    *,
    dense_candidates: Sequence[RetrievedChunk],
    bm25_candidates: Sequence[tuple[float, int]],
    cosine_by_id: dict[str, RetrievedChunk],
    records: Sequence[ChunkRecord],
    rrf_k: int,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[RankedCandidate]:
    """Fuse ranked lists while retaining the original cosine score unchanged."""
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 1:
        raise ValueError("rrf_k must be a positive integer")
    dense_weight = _positive_finite(dense_weight, name="dense_weight")
    bm25_weight = _positive_finite(bm25_weight, name="bm25_weight")

    dense_rank_by_id = {
        item.chunk_id: rank for rank, item in enumerate(dense_candidates, start=1)
    }
    bm25_rank_by_id: dict[str, int] = {}
    bm25_score_by_id: dict[str, float] = {}
    fusion_scores: dict[str, float] = {}

    for rank, item in enumerate(dense_candidates, start=1):
        fusion_scores[item.chunk_id] = fusion_scores.get(
            item.chunk_id, 0.0
        ) + dense_weight / (rrf_k + rank)

    for rank, (score, record_index) in enumerate(bm25_candidates, start=1):
        if isinstance(record_index, bool) or not isinstance(record_index, int):
            raise TypeError("BM25 record indices must be integers")
        if record_index < 0 or record_index >= len(records):
            raise ValueError("BM25 record index is outside the indexed records")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError("BM25 scores must be numeric")
        score_value = float(score)
        if score_value < 0.0 or not (score_value < float("inf")):
            raise ValueError("BM25 scores must be finite and non-negative")

        chunk_id = records[record_index].chunk_id
        bm25_rank_by_id[chunk_id] = rank
        bm25_score_by_id[chunk_id] = score_value
        fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + bm25_weight / (
            rrf_k + rank
        )

    ranked_ids = sorted(
        fusion_scores,
        key=lambda chunk_id: (
            -fusion_scores[chunk_id],
            -cosine_by_id[chunk_id].score,
            chunk_id,
        ),
    )

    return [
        RankedCandidate(
            chunk=cosine_by_id[chunk_id],
            dense_rank=dense_rank_by_id.get(chunk_id),
            bm25_rank=bm25_rank_by_id.get(chunk_id),
            bm25_score=bm25_score_by_id.get(chunk_id, 0.0),
            rrf_score=fusion_scores[chunk_id],
        )
        for chunk_id in ranked_ids
    ]


def _plain_ranked(
    chunks: Sequence[RetrievedChunk],
    *,
    source: str,
    bm25_candidates: Sequence[tuple[float, int]] | None = None,
    records: Sequence[ChunkRecord] | None = None,
) -> list[RankedCandidate]:
    bm25_by_id: dict[str, tuple[int, float]] = {}
    if bm25_candidates is not None and records is not None:
        bm25_by_id = {
            records[index].chunk_id: (rank, float(score))
            for rank, (score, index) in enumerate(bm25_candidates, start=1)
        }

    output: list[RankedCandidate] = []
    for rank, chunk in enumerate(chunks, start=1):
        bm25_data = bm25_by_id.get(chunk.chunk_id)
        output.append(
            RankedCandidate(
                chunk=chunk,
                dense_rank=rank if source == "dense" else None,
                bm25_rank=(bm25_data[0] if bm25_data else None),
                bm25_score=(bm25_data[1] if bm25_data else 0.0),
                rrf_score=None,
            )
        )
    return output


def _first_relevant_rank(
    ranked: Sequence[RankedCandidate],
    *,
    document_id: str,
    relevant_pages: set[int],
) -> int | None:
    for rank, candidate in enumerate(ranked, start=1):
        chunk = candidate.chunk
        if chunk.document_id == document_id and chunk.page_number in relevant_pages:
            return rank
    return None


def score_ranking(
    ranked_queries: Sequence[dict[str, Any]],
    *,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
) -> dict[str, Any]:
    """Calculate page recall, query hit rate, MRR, and unique-page coverage."""
    if not ranked_queries:
        raise ValueError("ranked_queries must not be empty")

    metrics: dict[str, Any] = {}
    for k in top_k_values:
        query_hits = 0
        page_recalls: list[float] = []
        reciprocal_ranks: list[float] = []

        for item in ranked_queries:
            relevant_pages = set(item["relevant_page_numbers"])
            document_id = item["benchmark_document_id"]
            ranked = item["ranked"]
            top_pages = {
                candidate.chunk.page_number
                for candidate in ranked[:k]
                if candidate.chunk.document_id == document_id
            }
            matched_pages = relevant_pages.intersection(top_pages)
            page_recalls.append(
                len(matched_pages) / len(relevant_pages) if relevant_pages else 0.0
            )
            first_rank = _first_relevant_rank(
                ranked,
                document_id=document_id,
                relevant_pages=relevant_pages,
            )
            if first_rank is not None and first_rank <= k:
                query_hits += 1
            reciprocal_ranks.append(
                1.0 / first_rank if first_rank is not None and first_rank <= k else 0.0
            )

        count = len(ranked_queries)
        metrics[f"page_recall_at_{k}"] = round(sum(page_recalls) / count, 6)
        metrics[f"query_hit_rate_at_{k}"] = round(query_hits / count, 6)
        metrics[f"mrr_at_{k}"] = round(sum(reciprocal_ranks) / count, 6)

    return metrics


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"median_ms": 0.0, "p95_ms": 0.0}
    median_index = len(ordered) // 2
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    if len(ordered) % 2:
        median = ordered[median_index]
    else:
        median = (ordered[median_index - 1] + ordered[median_index]) / 2.0
    return {"median_ms": round(median, 3), "p95_ms": round(ordered[p95_index], 3)}


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
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
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            temporary_path = handle.name
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
        raise


def run_comparison(
    *,
    dataset_path: Path,
    report_path: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_revision: str,
    device: str = "cpu",
    chunk_size_words: int = 220,
    overlap_words: int = 40,
    dense_candidate_k: int = DEFAULT_DENSE_CANDIDATE_K,
    bm25_candidate_k: int = DEFAULT_BM25_CANDIDATE_K,
    rrf_k: int = DEFAULT_RRF_K,
    weighted_dense_weight: float = 0.8,
    weighted_bm25_weight: float = 1.2,
) -> dict[str, Any]:
    """Run all retrieval variants against identical indexed chunks and queries."""
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("model_revision must be a full 40-character commit SHA")
    if chunk_size_words < 1 or overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("invalid chunk size/overlap configuration")
    if dense_candidate_k < 1 or bm25_candidate_k < 1 or rrf_k < 1:
        raise ValueError("candidate limits and rrf_k must be positive")
    weighted_dense_weight = _positive_finite(
        weighted_dense_weight, name="weighted_dense_weight"
    )
    weighted_bm25_weight = _positive_finite(
        weighted_bm25_weight, name="weighted_bm25_weight"
    )

    dataset_path = dataset_path.resolve()
    report_path = report_path.resolve()
    rows = load_jsonl(dataset_path)
    errors = validate_rows(rows, require_labelled=True)
    if errors:
        raise DatasetError(
            "retrieval dataset is not ready:\n"
            + "\n".join(f"- {error}" for error in errors)
        )

    documents: dict[str, Any] = {}
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    benchmark_document_ids: dict[str, str] = {}
    question_sources: dict[str, str] = {}

    for row in rows:
        pdf_path = resolve_pdf(row["document_path"])
        source = pdf_path.relative_to(PROJECT_ROOT).as_posix()
        question_sources[row["question_id"]] = source
        if source in documents:
            continue
        pdf_bytes = pdf_path.read_bytes()
        source_hash = hashlib.sha256(pdf_bytes).hexdigest()
        identity = hashlib.sha256(
            (source + "\0" + source_hash).encode("utf-8")
        ).hexdigest()[:32]
        document_id = f"benchmark-{identity}"
        document = ingest_pdf_upload(pdf_bytes=pdf_bytes, filename=pdf_path.name)
        documents[source] = document
        source_paths[source] = pdf_path
        source_hashes[source] = source_hash
        benchmark_document_ids[source] = document_id

    for row in rows:
        source = question_sources[row["question_id"]]
        document = documents[source]
        invalid_pages = [
            page for page in row["relevant_page_numbers"] if page > document.page_count
        ]
        if invalid_pages:
            raise DatasetError(
                f"{row['question_id']} labels pages {invalid_pages} outside "
                f"the {document.page_count}-page PDF {source!r}"
            )

    if report_path in {dataset_path, *source_paths.values()}:
        raise ValueError("report path must not overwrite the dataset or a source PDF")

    embedder = SentenceTransformerEmbedder(
        model_name=model_name,
        revision=model_revision,
        device=device,
    )
    pipeline = HybridRetrievalPipeline(
        embedder=embedder,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
        top_k=10,
        dense_candidate_k=dense_candidate_k,
        lexical_candidate_k=bm25_candidate_k,
        rrf_k=rrf_k,
    )

    indexing_started = time.perf_counter()
    chunk_count = 0
    for source, document in documents.items():
        indexed = pipeline.index_pages(
            document.pages,
            document_id=benchmark_document_ids[source],
            file_name=document.filename,
        )
        chunk_count += len(indexed)
    indexing_ms = (time.perf_counter() - indexing_started) * 1000.0
    if chunk_count == 0:
        raise RuntimeError("no non-empty chunks were indexed")

    strategy_names = ("dense", "bm25", "rrf_equal", "rrf_weighted")
    query_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in strategy_names}
    query_latencies: dict[str, list[float]] = {name: [] for name in strategy_names}
    query_embedding_latencies: list[float] = []

    for row in rows:
        query = row["query"]
        source = question_sources[row["question_id"]]
        document_id = benchmark_document_ids[source]

        embed_started = time.perf_counter()
        query_vector = embedder.embed_query(query)
        embedding_ms = (time.perf_counter() - embed_started) * 1000.0
        query_embedding_latencies.append(embedding_ms)

        dense_started = time.perf_counter()
        dense_all = pipeline.vector_store.search(
            query_vector,
            top_k=pipeline.vector_store.size,
        )
        dense_ms = (time.perf_counter() - dense_started) * 1000.0

        bm25_started = time.perf_counter()
        bm25_all = pipeline._rank_bm25(query)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000.0

        dense_candidates = dense_all[:dense_candidate_k]
        bm25_candidates = bm25_all[:bm25_candidate_k]
        cosine_by_id = {item.chunk_id: item for item in dense_all}
        records = pipeline.vector_store.records

        fusion_started = time.perf_counter()
        equal_fusion = fuse_rankings(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            cosine_by_id=cosine_by_id,
            records=records,
            rrf_k=rrf_k,
        )
        equal_fusion_ms = (time.perf_counter() - fusion_started) * 1000.0

        weighted_started = time.perf_counter()
        weighted_fusion = fuse_rankings(
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            cosine_by_id=cosine_by_id,
            records=records,
            rrf_k=rrf_k,
            dense_weight=weighted_dense_weight,
            bm25_weight=weighted_bm25_weight,
        )
        weighted_fusion_ms = (time.perf_counter() - weighted_started) * 1000.0

        dense_ranked = _plain_ranked(dense_all, source="dense")
        bm25_ranked: list[RankedCandidate] = []
        for rank, (score, record_index) in enumerate(bm25_all, start=1):
            record = records[record_index]
            source_chunk = cosine_by_id[record.chunk_id]
            bm25_ranked.append(
                RankedCandidate(
                    chunk=source_chunk,
                    dense_rank=None,
                    bm25_rank=rank,
                    bm25_score=float(score),
                    rrf_score=None,
                )
            )

        rankings = {
            "dense": dense_ranked,
            "bm25": bm25_ranked,
            "rrf_equal": equal_fusion,
            "rrf_weighted": weighted_fusion,
        }
        latencies = {
            "dense": embedding_ms + dense_ms,
            "bm25": bm25_ms,
            "rrf_equal": embedding_ms + dense_ms + bm25_ms + equal_fusion_ms,
            "rrf_weighted": embedding_ms + dense_ms + bm25_ms + weighted_fusion_ms,
        }

        for strategy, ranked in rankings.items():
            query_latencies[strategy].append(latencies[strategy])
            top_evidence = [
                {
                    "rank": rank,
                    "chunk_id": item.chunk.chunk_id,
                    "document_id": item.chunk.document_id,
                    "file_name": item.chunk.file_name,
                    "page_number": item.chunk.page_number,
                    "chunk_index": item.chunk.chunk_index,
                    "start_char": item.chunk.start_char,
                    "end_char": item.chunk.end_char,
                    "cosine_score": round(float(item.chunk.score), 8),
                    "dense_rank": item.dense_rank,
                    "bm25_rank": item.bm25_rank,
                    "bm25_score": round(item.bm25_score, 8),
                    "rrf_score": (
                        round(item.rrf_score, 10)
                        if item.rrf_score is not None
                        else None
                    ),
                    "text_preview": item.chunk.text[:1200],
                }
                for rank, item in enumerate(ranked[:10], start=1)
            ]
            query_rows[strategy].append(
                {
                    "question_id": row["question_id"],
                    "query": query,
                    "document_path": source,
                    "benchmark_document_id": document_id,
                    "relevant_page_numbers": row["relevant_page_numbers"],
                    "first_relevant_rank": _first_relevant_rank(
                        ranked,
                        document_id=document_id,
                        relevant_pages=set(row["relevant_page_numbers"]),
                    ),
                    "latency_ms": round(latencies[strategy], 3),
                    "retrieved_top_10": top_evidence,
                }
            )

    strategy_reports: dict[str, Any] = {}
    for strategy in strategy_names:
        metric_input = [
            {
                "relevant_page_numbers": row["relevant_page_numbers"],
                "benchmark_document_id": row["benchmark_document_id"],
                "ranked": rankings_for_metrics,
            }
            for row, rankings_for_metrics in zip(
                query_rows[strategy],
                [
                    _restore_ranked_for_metrics(query_result, pipeline)
                    for query_result in query_rows[strategy]
                ],
                strict=True,
            )
        ]
        strategy_reports[strategy] = {
            "metrics": score_ranking(metric_input),
            "latency": _latency_summary(query_latencies[strategy]),
            "queries": query_rows[strategy],
        }

    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dataset_path": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "source_documents": [
            {
                "path": source,
                "sha256": source_hashes[source],
                "benchmark_document_id": benchmark_document_ids[source],
                "page_count": documents[source].page_count,
            }
            for source in documents
        ],
        "embedding": {
            "model_name": embedder.model_name,
            "model_revision": embedder.revision,
            "dimension": embedder.embedding_dimension,
            "device": device,
        },
        "chunking": {
            "unit": "words",
            "chunk_size_words": chunk_size_words,
            "overlap_words": overlap_words,
            "page_local": True,
            "chunk_count": chunk_count,
        },
        "retrieval_config": {
            "dense_candidate_k": dense_candidate_k,
            "bm25_candidate_k": bm25_candidate_k,
            "rrf_k": rrf_k,
            "weighted_dense_weight": weighted_dense_weight,
            "weighted_bm25_weight": weighted_bm25_weight,
            "top_k_values": list(DEFAULT_TOP_K_VALUES),
        },
        "timing": {
            "corpus_indexing_ms": round(indexing_ms, 3),
            "query_embedding_median_ms": _latency_summary(query_embedding_latencies)[
                "median_ms"
            ],
        },
        "metric_definitions": {
            "page_recall_at_k": "Macro average, per query, of unique relevant labelled pages retrieved in the first k chunks divided by the number of relevant pages for that query.",
            "query_hit_rate_at_k": "Fraction of queries with at least one retrieved chunk from any relevant labelled page in the first k chunks.",
            "mrr_at_k": "Mean reciprocal rank of the first retrieved chunk from any relevant labelled page; zero when no relevant page is retrieved.",
            "answer_bearing_chunk_recall": "Not measured: the fixture contains relevant page labels, but no gold chunk IDs or answer-span annotations.",
        },
        "limitations": [
            "Page-level relevance is a proxy and does not prove that the retrieved chunk contains the answer-bearing passage.",
            "No answer generation, citation validation, claim entailment, or answer correctness is evaluated by this benchmark.",
            "BM25 latency excludes query embedding because lexical retrieval does not require it; dense and RRF timings include query embedding.",
            "RRF scores and BM25 scores are not cosine similarities and must not be substituted into the cosine score field.",
            "Cross-encoder reranking is not included in this first controlled comparison.",
        ],
        "strategies": strategy_reports,
    }
    _write_json_atomically(report_path, report)
    return report


def _restore_ranked_for_metrics(
    query_result: dict[str, Any],
    pipeline: HybridRetrievalPipeline,
) -> list[RankedCandidate]:
    """Resolve stored top-ten chunk IDs back to ranked source records."""
    by_id = {record.chunk_id: record for record in pipeline.vector_store.records}
    output: list[RankedCandidate] = []
    for entry in query_result["retrieved_top_10"]:
        record = by_id[entry["chunk_id"]]
        chunk = RetrievedChunk.model_validate(
            record.model_dump() | {"score": float(entry["cosine_score"])}
        )
        output.append(
            RankedCandidate(
                chunk=chunk,
                dense_rank=entry["dense_rank"],
                bm25_rank=entry["bm25_rank"],
                bm25_score=entry["bm25_score"],
                rrf_score=entry["rrf_score"],
            )
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Exact 40-character Hugging Face commit SHA; branch names are not accepted.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size-words", type=int, default=220)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument(
        "--dense-candidate-k", type=int, default=DEFAULT_DENSE_CANDIDATE_K
    )
    parser.add_argument(
        "--bm25-candidate-k", type=int, default=DEFAULT_BM25_CANDIDATE_K
    )
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--dense-weight", type=float, default=0.8)
    parser.add_argument("--bm25-weight", type=float, default=1.2)
    args = parser.parse_args(argv)

    dataset_path = (
        args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    )
    report_path = (
        args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    )
    try:
        report = run_comparison(
            dataset_path=dataset_path,
            report_path=report_path,
            model_name=args.model_name,
            model_revision=args.model_revision,
            device=args.device,
            chunk_size_words=args.chunk_size_words,
            overlap_words=args.overlap_words,
            dense_candidate_k=args.dense_candidate_k,
            bm25_candidate_k=args.bm25_candidate_k,
            rrf_k=args.rrf_k,
            weighted_dense_weight=args.dense_weight,
            weighted_bm25_weight=args.bm25_weight,
        )
    except (DatasetError, TypeError, ValueError, OSError, RuntimeError) as exc:
        print(
            f"ERROR: retrieval comparison failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Questions: {len(next(iter(report['strategies'].values()))['queries'])}")
    print("\nStrategy summary")
    for strategy, result in report["strategies"].items():
        metric = result["metrics"]
        latency = result["latency"]
        print(
            f"{strategy:12} "
            f"page_recall@5={metric['page_recall_at_5']:.3f} "
            f"hit@5={metric['query_hit_rate_at_5']:.3f} "
            f"MRR@10={metric['mrr_at_10']:.3f} "
            f"median_ms={latency['median_ms']:.2f}"
        )
    print(f"Report: {report_path}")
    print(
        "Answer-bearing chunk recall: NOT MEASURED (gold chunk/span labels are absent)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
