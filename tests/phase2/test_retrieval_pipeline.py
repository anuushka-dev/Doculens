"""Tests for end-to-end retrieval orchestration without model downloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.adapters.embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
)
from app.adapters.vectorstore.faiss_store import (
    FaissStoreIntegrityError,
    FaissVectorStore,
)
from app.schemas.retrieval import ChunkRecord, RetrievalResponse
from app.services.retrieval_pipeline import (
    NoChunksToIndexError,
    RetrievalPipeline,
)


@dataclass(frozen=True)
class Page:
    """Small stand-in for an extracted Phase 1 page."""

    page_number: int
    text: str


class FakeEmbedder:
    """Deterministic fake embedding adapter; no model weights are downloaded."""

    model_name = DEFAULT_EMBEDDING_MODEL
    revision: str | None = None

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.text_batches: list[list[str]] = []
        self.query_calls: list[str] = []

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.lower()

        if "alpha" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "beta" in lowered:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        self.text_batches.append(list(texts))

        if self.failure is not None:
            raise self.failure

        return np.vstack([self._vector(text) for text in texts]).astype(
            np.float32,
            copy=False,
        )

    def embed_query(self, query: str) -> np.ndarray:
        self.query_calls.append(query)

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be blank")
        if self.failure is not None:
            raise self.failure

        return self._vector(query)


def make_pages() -> list[Page]:
    return [
        Page(1, "alpha apple amber"),
        Page(2, "beta berry birch"),
    ]


def make_pipeline(
    *,
    embedder: FakeEmbedder | None = None,
    top_k: int = 2,
) -> tuple[RetrievalPipeline, FakeEmbedder]:
    selected_embedder = embedder or FakeEmbedder()

    pipeline = RetrievalPipeline(
        embedder=selected_embedder,
        chunk_size_words=3,
        overlap_words=0,
        top_k=top_k,
    )

    return pipeline, selected_embedder


def test_pipeline_defaults_match_phase_two_baseline() -> None:
    pipeline = RetrievalPipeline()

    assert pipeline.chunk_size_words == 220
    assert pipeline.overlap_words == 40
    assert pipeline.top_k == 5
    assert pipeline.vector_store.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_index_pages_returns_chunks_and_preserves_page_metadata() -> None:
    pipeline, embedder = make_pipeline()

    records = pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    assert len(records) == 2
    assert all(isinstance(record, ChunkRecord) for record in records)
    assert [record.page_number for record in records] == [1, 2]
    assert [record.text for record in records] == [
        "alpha apple amber",
        "beta berry birch",
    ]
    assert [record.chunk_index for record in records] == [0, 0]
    assert pipeline.vector_store.size == 2
    assert len(embedder.text_batches) == 1
    assert embedder.text_batches[0] == [record.text for record in records]


def test_search_returns_valid_response_and_source_provenance() -> None:
    pipeline, _ = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    response = pipeline.search("alpha policy")

    assert isinstance(response, RetrievalResponse)
    assert response.query == "alpha policy"
    assert response.top_k == 2
    assert isinstance(response.results, tuple)
    assert response.results[0].page_number == 1
    assert response.results[0].file_name == "manual.pdf"
    assert response.results[0].document_id == "manual-1"
    assert response.results[0].text == "alpha apple amber"
    assert response.results[0].score == pytest.approx(1.0)


def test_search_allows_top_k_override() -> None:
    pipeline, _ = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    response = pipeline.search("beta", top_k=1)

    assert response.top_k == 1
    assert len(response.results) == 1
    assert response.results[0].page_number == 2


def test_search_preserves_original_nonblank_query() -> None:
    pipeline, embedder = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    response = pipeline.search(" alpha ")

    assert response.query == " alpha "
    assert embedder.query_calls == [" alpha "]


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_search_rejects_blank_query_without_embedding(query: str) -> None:
    pipeline, embedder = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )
    embedder.query_calls.clear()

    with pytest.raises(ValueError, match="query must not be blank"):
        pipeline.search(query)

    assert embedder.query_calls == []


def test_search_rejects_non_string_query() -> None:
    pipeline, embedder = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    with pytest.raises(TypeError, match="query must be a string"):
        pipeline.search(123)  # type: ignore[arg-type]

    assert embedder.query_calls == []


@pytest.mark.parametrize(
    ("top_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_search_rejects_invalid_top_k(
    top_k: Any,
    error_type: type[Exception],
) -> None:
    pipeline, _ = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    with pytest.raises(error_type, match="top_k"):
        pipeline.search("alpha", top_k=top_k)


def test_search_on_empty_index_does_not_invoke_embedder() -> None:
    pipeline, embedder = make_pipeline()

    with pytest.raises(ValueError, match="empty retrieval index"):
        pipeline.search("alpha")

    assert embedder.query_calls == []


def test_empty_or_whitespace_pages_do_not_create_vectors() -> None:
    pipeline, embedder = make_pipeline()
    pages = [Page(1, "  \n "), Page(2, "")]

    with pytest.raises(NoChunksToIndexError, match="no non-empty chunks"):
        pipeline.index_pages(
            pages,
            document_id="empty-doc",
            file_name="empty.pdf",
        )

    assert pipeline.vector_store.size == 0
    assert embedder.text_batches == []


def test_prevents_duplicate_indexing_before_embedding_again() -> None:
    pipeline, embedder = make_pipeline()
    pages = make_pages()

    pipeline.index_pages(
        pages,
        document_id="manual-1",
        file_name="manual.pdf",
    )
    batches_before_retry = len(embedder.text_batches)

    with pytest.raises(ValueError, match="already exist"):
        pipeline.index_pages(
            pages,
            document_id="manual-1",
            file_name="manual.pdf",
        )

    assert len(embedder.text_batches) == batches_before_retry
    assert pipeline.vector_store.size == 2


def test_allows_distinct_document_ids_to_be_indexed() -> None:
    pipeline, _ = make_pipeline()
    pages = [Page(1, "alpha apple amber")]

    first = pipeline.index_pages(
        pages,
        document_id="manual-1",
        file_name="manual.pdf",
    )
    second = pipeline.index_pages(
        pages,
        document_id="manual-2",
        file_name="another.pdf",
    )

    assert first[0].chunk_id != second[0].chunk_id
    assert pipeline.vector_store.size == 2


def test_embedding_failure_leaves_empty_index_unchanged() -> None:
    embedder = FakeEmbedder(failure=RuntimeError("encoder unavailable"))
    pipeline, _ = make_pipeline(embedder=embedder)

    with pytest.raises(RuntimeError, match="encoder unavailable"):
        pipeline.index_pages(
            make_pages(),
            document_id="manual-1",
            file_name="manual.pdf",
        )

    assert pipeline.vector_store.size == 0


@pytest.mark.parametrize(
    ("chunk_size_words", "overlap_words", "error_type"),
    [
        (0, 0, ValueError),
        (-1, 0, ValueError),
        (True, 0, TypeError),
        (3, -1, ValueError),
        (3, 3, ValueError),
        (3, True, TypeError),
    ],
)
def test_rejects_invalid_chunk_configuration(
    chunk_size_words: Any,
    overlap_words: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        RetrievalPipeline(
            embedder=FakeEmbedder(),
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )


@pytest.mark.parametrize(
    ("top_k", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_rejects_invalid_default_top_k(
    top_k: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        RetrievalPipeline(embedder=FakeEmbedder(), top_k=top_k)


def test_rejects_model_mismatch_between_embedder_and_store() -> None:
    store = FaissVectorStore(embedding_model="a-different-model")

    with pytest.raises(ValueError, match="embedding model"):
        RetrievalPipeline(
            embedder=FakeEmbedder(),
            vector_store=store,
        )


def test_rejects_revision_mismatch_between_embedder_and_store() -> None:
    class RevisionedFakeEmbedder(FakeEmbedder):
        revision = "revision-A"

    with pytest.raises(ValueError, match="embedding revision"):
        RetrievalPipeline(
            embedder=RevisionedFakeEmbedder(),
            vector_store=FaissVectorStore(
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                embedding_revision="revision-B",
            ),
        )


def test_save_and_load_preserve_search_results(tmp_path: Path) -> None:
    pipeline, _ = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    index_path = tmp_path / "pipeline.faiss"
    pipeline.save(index_path)

    loaded = RetrievalPipeline.load(
        index_path,
        embedder=FakeEmbedder(),
        chunk_size_words=3,
        overlap_words=0,
        top_k=2,
    )

    original = pipeline.search("alpha")
    restored = loaded.search("alpha")

    assert [result.chunk_id for result in restored.results] == [
        result.chunk_id for result in original.results
    ]
    assert [result.score for result in restored.results] == pytest.approx(
        [result.score for result in original.results]
    )
    assert restored.results[0].page_number == original.results[0].page_number


def test_load_rejects_embedder_with_different_model(
    tmp_path: Path,
) -> None:
    pipeline, _ = make_pipeline()

    pipeline.index_pages(
        make_pages(),
        document_id="manual-1",
        file_name="manual.pdf",
    )

    index_path = tmp_path / "pipeline.faiss"
    pipeline.save(index_path)

    class DifferentModelEmbedder(FakeEmbedder):
        model_name = "different/model"

    with pytest.raises(FaissStoreIntegrityError, match="embedding model"):
        RetrievalPipeline.load(
            index_path,
            embedder=DifferentModelEmbedder(),
        )