"""Orchestrate page chunking, local embeddings, FAISS indexing, and search."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from numpy.typing import NDArray

from app.adapters.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)
from app.adapters.vectorstore.faiss_store import FaissVectorStore
from app.schemas.retrieval import ChunkRecord, RetrievalResponse
from app.services.page_chunker import PageLike, chunk_pages

EmbeddingMatrix = NDArray
EmbeddingVector = NDArray


class EmbeddingAdapter(Protocol):
    """Minimum embedding interface needed by the retrieval pipeline."""

    model_name: str
    revision: str | None

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> EmbeddingMatrix:
        """Return one normalized embedding per input text."""

    def embed_query(self, query: str) -> EmbeddingVector:
        """Return one normalized query vector."""


class NoChunksToIndexError(ValueError):
    """No indexable text was found in the supplied pages."""


def _positive_integer(value: object, *, field_name: str) -> int:
    """Validate a positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be positive")
    return value


def _validate_chunk_configuration(
    chunk_size_words: object,
    overlap_words: object,
) -> tuple[int, int]:
    """Validate the same word-window constraints as the page chunker."""
    chunk_size = _positive_integer(
        chunk_size_words,
        field_name="chunk_size_words",
    )
    if isinstance(overlap_words, bool) or not isinstance(overlap_words, int):
        raise TypeError("overlap_words must be an integer")
    if overlap_words < 0 or overlap_words >= chunk_size:
        raise ValueError(
            "overlap_words must be >= 0 and smaller than chunk_size_words"
        )
    return chunk_size, overlap_words


class RetrievalPipeline:
    """Coordinate page-local chunking, embeddings, FAISS, and top-k search.

    PDF extraction remains Phase 1's responsibility. This class accepts the
    extracted page records and never reparses the source PDF.

    One vector is generated for each chunk, and the FAISS store preserves the
    one-to-one ordering between those vectors and their source metadata.
    """

    def __init__(
        self,
        *,
        embedder: EmbeddingAdapter | None = None,
        vector_store: FaissVectorStore | None = None,
        chunk_size_words: int = 220,
        overlap_words: int = 40,
        top_k: int = 5,
    ) -> None:
        self.chunk_size_words, self.overlap_words = (
            _validate_chunk_configuration(
                chunk_size_words,
                overlap_words,
            )
        )
        self.top_k = _positive_integer(top_k, field_name="top_k")

        selected_embedder = (
            embedder if embedder is not None else SentenceTransformerEmbedder()
        )

        if not isinstance(selected_embedder.model_name, str) or not (
            selected_embedder.model_name.strip()
        ):
            raise ValueError("embedder.model_name must be a non-blank string")

        if selected_embedder.revision is not None and (
            not isinstance(selected_embedder.revision, str)
            or not selected_embedder.revision.strip()
        ):
            raise ValueError("embedder.revision must be None or non-blank")

        selected_store = (
            vector_store
            if vector_store is not None
            else FaissVectorStore(
                embedding_model=selected_embedder.model_name,
                embedding_revision=selected_embedder.revision,
            )
        )

        if selected_store.embedding_model != selected_embedder.model_name:
            raise ValueError(
                "embedding model does not match the vector-store configuration"
            )
        if selected_store.embedding_revision != selected_embedder.revision:
            raise ValueError(
                "embedding revision does not match the vector-store configuration"
            )

        self.embedder = selected_embedder
        self.vector_store = selected_store

    def index_pages(
        self,
        pages: Sequence[PageLike],
        *,
        document_id: str,
        file_name: str,
    ) -> tuple[ChunkRecord, ...]:
        """Chunk extracted pages, embed each chunk, and add them to FAISS.

        Returns the exact chunk records submitted to the index. Empty or
        whitespace-only pages are skipped by the chunker; if every page is
        empty, this method raises and makes no embedding/indexing call.
        """
        records = chunk_pages(
            pages,
            document_id=document_id,
            file_name=file_name,
            chunk_size_words=self.chunk_size_words,
            overlap_words=self.overlap_words,
        )

        if not records:
            raise NoChunksToIndexError(
                "no non-empty chunks were produced from the supplied pages"
            )

        existing_ids = {
            record.chunk_id for record in self.vector_store.records
        }
        incoming_ids = {record.chunk_id for record in records}

        if existing_ids.intersection(incoming_ids):
            raise ValueError("one or more chunk IDs already exist in the index")

        vectors = self.embedder.embed_texts(
            [record.text for record in records]
        )
        self.vector_store.add(records, vectors)

        return tuple(records)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        """Embed a query and return ranked chunks with their source metadata."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be blank")

        result_count = (
            self.top_k
            if top_k is None
            else _positive_integer(top_k, field_name="top_k")
        )

        if self.vector_store.size == 0:
            raise ValueError("cannot search an empty retrieval index")

        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(
            query_vector,
            top_k=result_count,
        )

        return RetrievalResponse(
            query=query,
            top_k=result_count,
            results=tuple(results),
        )

    def save(self, index_path: str | Path) -> None:
        """Persist the vector index and its ordered metadata sidecar."""
        self.vector_store.save(index_path)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        *,
        embedder: EmbeddingAdapter | None = None,
        chunk_size_words: int = 220,
        overlap_words: int = 40,
        top_k: int = 5,
    ) -> RetrievalPipeline:
        """Load a saved index using an embedder with matching model settings."""
        selected_embedder = (
            embedder if embedder is not None else SentenceTransformerEmbedder()
        )

        store = FaissVectorStore.load(
            index_path,
            embedding_model=selected_embedder.model_name,
            embedding_revision=selected_embedder.revision,
        )

        return cls(
            embedder=selected_embedder,
            vector_store=store,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            top_k=top_k,
        )