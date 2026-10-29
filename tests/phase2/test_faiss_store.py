"""Tests for the FAISS vector store and metadata persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.adapters.vectorstore.faiss_store import (
    FaissStoreIntegrityError,
    FaissVectorStore,
)
from app.schemas.retrieval import ChunkRecord, RetrievedChunk


def make_record(
    index: int,
    *,
    text: str | None = None,
) -> ChunkRecord:
    """Create a valid, independent source chunk."""
    chunk_text = text or f"source passage {index}"

    return ChunkRecord(
        chunk_id=f"chunk-{index}",
        document_id="doc-1",
        file_name="manual.pdf",
        page_number=index + 1,
        chunk_index=0,
        start_char=0,
        end_char=len(chunk_text),
        text=chunk_text,
    )


def make_vectors(count: int = 3) -> np.ndarray:
    """Return a small matrix of normalized two-dimensional vectors."""
    vectors = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.float32,
    )
    return vectors[:count].copy()


def make_store(count: int = 3) -> FaissVectorStore:
    """Create a populated store using deterministic vectors."""
    store = FaissVectorStore()
    store.add(
        [make_record(index) for index in range(count)],
        make_vectors(count),
    )
    return store


def test_new_store_has_no_index_or_dimension() -> None:
    store = FaissVectorStore()

    assert store.size == 0
    assert store.embedding_dimension is None
    assert store.records == ()


def test_add_preserves_vector_to_record_order() -> None:
    store = make_store()

    assert store.size == 3
    assert store.embedding_dimension == 2
    assert [record.chunk_id for record in store.records] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]


def test_search_returns_ranked_chunks_with_source_metadata() -> None:
    store = make_store()

    results = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=3,
    )

    assert all(isinstance(result, RetrievedChunk) for result in results)
    assert [result.chunk_id for result in results] == [
        "chunk-0",
        "chunk-2",
        "chunk-1",
    ]
    assert [result.page_number for result in results] == [1, 3, 2]
    assert all(result.file_name == "manual.pdf" for result in results)
    assert all(result.document_id == "doc-1" for result in results)
    assert [result.score for result in results] == pytest.approx(
        [1.0, 1.0 / np.sqrt(2.0), 0.0],
        abs=1e-6,
    )


def test_search_limits_results_to_top_k() -> None:
    store = make_store()

    results = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-0"


def test_search_returns_all_available_results_when_top_k_is_larger() -> None:
    store = make_store(count=2)

    results = store.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=10,
    )

    assert len(results) == 2


def test_rejects_empty_record_batch() -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError, match="at least one"):
        store.add([], np.empty((0, 2), dtype=np.float32))


def test_rejects_duplicate_chunk_ids_within_batch() -> None:
    store = FaissVectorStore()
    record = make_record(0)

    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        store.add(
            [record, record],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )


def test_rejects_chunk_id_already_in_index() -> None:
    store = make_store(count=1)

    with pytest.raises(ValueError, match="already exist"):
        store.add(
            [make_record(0)],
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )

    assert store.size == 1


def test_rejects_embedding_row_count_mismatch() -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError, match="row count"):
        store.add(
            [make_record(0), make_record(1)],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )

    assert store.size == 0


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        (
            np.asarray([1.0, 0.0], dtype=np.float32),
            "two-dimensional",
        ),
        (
            np.asarray([[np.nan, 0.0]], dtype=np.float32),
            "NaN or infinite",
        ),
        (
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            "zero-length",
        ),
        (
            np.asarray([[2.0, 0.0]], dtype=np.float32),
            "L2-normalized",
        ),
        (
            [["not", "numeric"]],
            "numeric matrix",
        ),
    ],
)
def test_rejects_invalid_embedding_matrices(
    embeddings: Any,
    message: str,
) -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError, match=message):
        store.add([make_record(0)], embeddings)

    assert store.size == 0


def test_rejects_dimension_change_between_additions() -> None:
    store = make_store(count=1)

    with pytest.raises(ValueError, match="dimension"):
        store.add(
            [make_record(1)],
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        )

    assert store.size == 1


def test_rejects_search_on_empty_store() -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError, match="empty"):
        store.search(np.asarray([1.0, 0.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("top_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_rejects_invalid_top_k(
    top_k: Any,
    error_type: type[Exception],
) -> None:
    store = make_store(count=1)

    with pytest.raises(error_type, match="top_k"):
        store.search(
            np.asarray([1.0, 0.0], dtype=np.float32),
            top_k=top_k,
        )


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            "one-dimensional",
        ),
        (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "dimension",
        ),
        (
            np.asarray([np.nan, 0.0], dtype=np.float32),
            "NaN or infinite",
        ),
        (
            np.asarray([0.0, 0.0], dtype=np.float32),
            "non-zero",
        ),
        (
            np.asarray([2.0, 0.0], dtype=np.float32),
            "L2-normalized",
        ),
    ],
)
def test_rejects_invalid_query_embeddings(
    query: Any,
    message: str,
) -> None:
    store = make_store(count=1)

    with pytest.raises(ValueError, match=message):
        store.search(query)


def test_save_and_load_preserve_results_and_metadata(
    tmp_path: Path,
) -> None:
    store = FaissVectorStore(
        embedding_model="test-embedding-model",
        embedding_revision="abc123",
    )
    store.add(
        [make_record(index) for index in range(3)],
        make_vectors(),
    )

    index_path = tmp_path / "retrieval.faiss"
    store.save(index_path)

    sidecar_path = tmp_path / "retrieval.faiss.json"
    assert index_path.is_file()
    assert sidecar_path.is_file()

    manifest = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert manifest["index_type"] == "IndexFlatIP"
    assert manifest["metric"] == "inner_product"
    assert manifest["embedding_model"] == "test-embedding-model"
    assert manifest["embedding_revision"] == "abc123"
    assert manifest["vector_count"] == 3
    assert manifest["embedding_dimension"] == 2
    assert len(manifest["index_sha256"]) == 64

    loaded = FaissVectorStore.load(
        index_path,
        embedding_model="test-embedding-model",
        embedding_revision="abc123",
    )

    assert loaded.size == store.size
    assert loaded.embedding_dimension == store.embedding_dimension
    assert loaded.records == store.records

    query = np.asarray([1.0, 0.0], dtype=np.float32)
    original_results = store.search(query, top_k=3)
    loaded_results = loaded.search(query, top_k=3)

    assert [item.chunk_id for item in loaded_results] == [
        item.chunk_id for item in original_results
    ]
    assert [item.score for item in loaded_results] == pytest.approx(
        [item.score for item in original_results],
        abs=1e-6,
    )


def test_load_rejects_different_embedding_model(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "retrieval.faiss"

    store = FaissVectorStore(embedding_model="model-A")
    store.add([make_record(0)], make_vectors(count=1))
    store.save(index_path)

    with pytest.raises(FaissStoreIntegrityError, match="embedding model"):
        FaissVectorStore.load(
            index_path,
            embedding_model="model-B",
        )


def test_load_rejects_different_embedding_revision(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "retrieval.faiss"

    store = FaissVectorStore(
        embedding_model="test-model",
        embedding_revision="revision-A",
    )
    store.add([make_record(0)], make_vectors(count=1))
    store.save(index_path)

    with pytest.raises(FaissStoreIntegrityError, match="revision"):
        FaissVectorStore.load(
            index_path,
            embedding_model="test-model",
            embedding_revision="revision-B",
        )


def test_load_rejects_missing_metadata_sidecar(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "retrieval.faiss"

    store = make_store(count=1)
    store.save(index_path)
    (tmp_path / "retrieval.faiss.json").unlink()

    with pytest.raises(FaissStoreIntegrityError, match="sidecar"):
        FaissVectorStore.load(index_path)


def test_load_rejects_index_checksum_mismatch(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "retrieval.faiss"
    make_store(count=1).save(index_path)

    index_path.write_bytes(index_path.read_bytes() + b"tampered")

    with pytest.raises(FaissStoreIntegrityError, match="checksum"):
        FaissVectorStore.load(index_path)


def test_load_rejects_metadata_vector_count_mismatch(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "retrieval.faiss"
    make_store(count=2).save(index_path)

    sidecar_path = tmp_path / "retrieval.faiss.json"
    manifest = json.loads(sidecar_path.read_text(encoding="utf-8"))
    manifest["vector_count"] = 100
    sidecar_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(FaissStoreIntegrityError, match="vector count"):
        FaissVectorStore.load(index_path)


def test_save_rejects_empty_store(tmp_path: Path) -> None:
    store = FaissVectorStore()

    with pytest.raises(ValueError, match="empty"):
        store.save(tmp_path / "empty.faiss")