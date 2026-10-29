"""Exact FAISS vector search with JSON-backed chunk provenance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.adapters.embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
)
from app.schemas.retrieval import ChunkRecord, RetrievedChunk

INDEX_FORMAT_VERSION = 1
INDEX_TYPE = "IndexFlatIP"
INDEX_METRIC = "inner_product"
UNIT_NORM_TOLERANCE = 1e-4
COSINE_SCORE_TOLERANCE = 1e-4

EmbeddingMatrix = NDArray[np.float32]
EmbeddingVector = NDArray[np.float32]


class FaissStoreError(RuntimeError):
    """Base exception for expected vector-store failures."""


class FaissDependencyError(FaissStoreError):
    """FAISS is not installed in the current environment."""


class FaissStoreIntegrityError(FaissStoreError):
    """The index or its metadata sidecar is invalid or inconsistent."""


class FaissStorePersistenceError(FaissStoreError):
    """The index and its metadata could not be persisted."""


def _load_faiss() -> Any:
    """Import FAISS only when an index operation actually needs it."""
    try:
        import faiss
    except ImportError as exc:
        raise FaissDependencyError(
            "Install faiss-cpu to use the DocuLens vector store."
        ) from exc

    return faiss


def _sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _sidecar_path(index_path: Path) -> Path:
    """Return the metadata sidecar path associated with an index file."""
    return index_path.with_name(f"{index_path.name}.json")


def _temporary_path(directory: Path, prefix: str) -> Path:
    """Reserve a temporary file beside the target for atomic replacement."""
    descriptor, name = tempfile.mkstemp(
        dir=directory,
        prefix=prefix,
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


class FaissVectorStore:
    """Store normalized embeddings in FAISS and source records in row order.

    Each FAISS row corresponds to the record at the same position in
    ``records``. This order is preserved in the JSON sidecar.

    Persisted indexes are accepted only when the manifest, checksum, model
    configuration, vector count, dimension, and metadata records agree.
    """

    def __init__(
        self,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_revision: str | None = None,
    ) -> None:
        if (
            not isinstance(embedding_model, str)
            or not embedding_model.strip()
        ):
            raise ValueError("embedding_model must be a non-blank string")

        if embedding_revision is not None and (
            not isinstance(embedding_revision, str)
            or not embedding_revision.strip()
        ):
            raise ValueError(
                "embedding_revision must be None or a non-blank string"
            )

        self.embedding_model = embedding_model.strip()
        self.embedding_revision = (
            embedding_revision.strip()
            if embedding_revision is not None
            else None
        )

        self._index: Any | None = None
        self._records: list[ChunkRecord] = []
        self._dimension: int | None = None

    @property
    def size(self) -> int:
        """Return the number of vectors currently indexed."""
        if self._index is None:
            return 0

        return int(self._index.ntotal)

    @property
    def embedding_dimension(self) -> int | None:
        """Return the vector dimension, or None before the first addition."""
        return self._dimension

    @property
    def records(self) -> tuple[ChunkRecord, ...]:
        """Expose the ordered metadata as an immutable tuple."""
        return tuple(self._records)

    @staticmethod
    def _validate_unit_norms(vectors: EmbeddingMatrix) -> None:
        """Reject vectors that cannot be interpreted as cosine embeddings."""
        norms = np.linalg.norm(vectors, axis=1)

        if not np.isfinite(norms).all():
            raise ValueError("Embedding vector norms must be finite")

        if np.any(norms <= 0.0):
            raise ValueError("Embedding vectors must not be zero-length")

        if not np.allclose(
            norms,
            1.0,
            rtol=UNIT_NORM_TOLERANCE,
            atol=UNIT_NORM_TOLERANCE,
        ):
            raise ValueError(
                "Embedding vectors must be L2-normalized before indexing"
            )

    @staticmethod
    def _validated_records(
        records: Sequence[ChunkRecord],
    ) -> list[ChunkRecord]:
        """Validate the incoming records without changing their source text."""
        if isinstance(records, (str, bytes)):
            raise TypeError("records must be a sequence of ChunkRecord objects")

        validated: list[ChunkRecord] = []

        for index, record in enumerate(records):
            try:
                chunk = (
                    record
                    if isinstance(record, ChunkRecord)
                    else ChunkRecord.model_validate(record)
                )
            except Exception as exc:
                raise ValueError(
                    f"records[{index}] is not a valid ChunkRecord"
                ) from exc

            validated.append(chunk)

        if not validated:
            raise ValueError("records must contain at least one chunk")

        chunk_ids = [record.chunk_id for record in validated]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("incoming records contain duplicate chunk IDs")

        return validated

    @staticmethod
    def _validated_vectors(
        embeddings: object,
        *,
        expected_rows: int,
    ) -> EmbeddingMatrix:
        """Convert input to contiguous float32 vectors and validate them."""
        try:
            vectors = np.asarray(embeddings, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embeddings must be a numeric matrix") from exc

        if vectors.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")

        if vectors.shape[0] != expected_rows:
            raise ValueError(
                "embedding row count must match the number of records"
            )

        if vectors.shape[1] < 1:
            raise ValueError("embedding dimension must be positive")

        if not np.isfinite(vectors).all():
            raise ValueError("embeddings contain NaN or infinite values")

        FaissVectorStore._validate_unit_norms(vectors)

        return np.ascontiguousarray(vectors, dtype=np.float32)

    def add(
        self,
        records: Sequence[ChunkRecord],
        embeddings: object,
    ) -> None:
        """Append records and their corresponding normalized vectors."""
        validated_records = self._validated_records(records)
        vectors = self._validated_vectors(
            embeddings,
            expected_rows=len(validated_records),
        )

        existing_ids = {record.chunk_id for record in self._records}
        incoming_ids = {record.chunk_id for record in validated_records}

        if existing_ids.intersection(incoming_ids):
            raise ValueError("chunk IDs already exist in the vector store")

        dimension = int(vectors.shape[1])

        if self._dimension is not None and dimension != self._dimension:
            raise ValueError(
                "embedding dimension does not match the existing index"
            )

        faiss = _load_faiss()

        target_index = self._index
        if target_index is None:
            target_index = faiss.IndexFlatIP(dimension)

        try:
            target_index.add(vectors)
        except Exception as exc:
            raise FaissStoreError(
                "FAISS failed while adding embedding vectors."
            ) from exc

        self._index = target_index
        self._dimension = dimension
        self._records.extend(validated_records)

    def search(
        self,
        query_embedding: object,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks with their original metadata and scores."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if self._index is None or self.size == 0:
            raise ValueError("cannot search an empty vector store")

        try:
            query = np.asarray(query_embedding, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("query embedding must be numeric") from exc

        if query.ndim != 1:
            raise ValueError("query embedding must be a one-dimensional vector")

        if query.shape[0] != self._dimension:
            raise ValueError("query dimension does not match the index")

        if not np.isfinite(query).all():
            raise ValueError("query embedding contains NaN or infinite values")

        query_norm = float(np.linalg.norm(query))

        if not np.isfinite(query_norm) or query_norm <= 0.0:
            raise ValueError("query embedding must be finite and non-zero")

        if not np.isclose(
            query_norm,
            1.0,
            rtol=UNIT_NORM_TOLERANCE,
            atol=UNIT_NORM_TOLERANCE,
        ):
            raise ValueError("query embedding must be L2-normalized")

        query_matrix = np.ascontiguousarray(query.reshape(1, -1))
        result_count = min(top_k, self.size)

        try:
            scores, row_ids = self._index.search(
                query_matrix,
                result_count,
            )
        except Exception as exc:
            raise FaissStoreError("FAISS search failed.") from exc

        ranked: list[tuple[float, ChunkRecord]] = []

        for raw_score, raw_row_id in zip(
            scores[0],
            row_ids[0],
            strict=True,
        ):
            row_id = int(raw_row_id)

            if row_id < 0:
                continue

            if row_id >= len(self._records):
                raise FaissStoreIntegrityError(
                    "FAISS returned a row without matching metadata"
                )

            score = float(raw_score)

            if (
                not np.isfinite(score)
                or score < -1.0 - COSINE_SCORE_TOLERANCE
                or score > 1.0 + COSINE_SCORE_TOLERANCE
            ):
                raise FaissStoreIntegrityError(
                    "cosine similarity score is outside the valid range"
                )

            # Clip tiny floating-point overshoots, not material score errors.
            score = float(np.clip(score, -1.0, 1.0))
            ranked.append((score, self._records[row_id]))

        # Stable secondary ordering for ties among returned candidates.
        ranked.sort(key=lambda item: (-item[0], item[1].chunk_id))

        results: list[RetrievedChunk] = []

        for score, record in ranked:
            result_data = record.model_dump()
            result_data["score"] = score
            results.append(RetrievedChunk.model_validate(result_data))

        return results

    def save(self, index_path: str | Path) -> None:
        """Persist the FAISS index and ordered metadata with a checksum."""
        if self._index is None or self.size == 0:
            raise ValueError("cannot save an empty vector store")

        target = Path(index_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        sidecar = _sidecar_path(target)

        temporary_index: Path | None = None
        temporary_manifest: Path | None = None

        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "index_type": INDEX_TYPE,
            "metric": INDEX_METRIC,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "embedding_dimension": self._dimension,
            "vector_count": self.size,
            "index_sha256": "",
            "records": [
                record.model_dump(mode="json")
                for record in self._records
            ],
        }

        try:
            temporary_index = _temporary_path(
                target.parent,
                f".{target.name}.",
            )
            temporary_manifest = _temporary_path(
                target.parent,
                f".{sidecar.name}.",
            )

            faiss = _load_faiss()
            faiss.write_index(self._index, str(temporary_index))
            manifest["index_sha256"] = _sha256_file(temporary_index)

            with temporary_manifest.open(
                "w",
                encoding="utf-8",
            ) as file_handle:
                json.dump(
                    manifest,
                    file_handle,
                    indent=2,
                    sort_keys=True,
                )
                file_handle.write("\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())

            # Publish the manifest last. A torn pair is rejected by checksum.
            os.replace(temporary_index, target)
            temporary_index = None

            os.replace(temporary_manifest, sidecar)
            temporary_manifest = None

        except Exception as exc:
            raise FaissStorePersistenceError(
                f"Could not persist the FAISS index at {target}."
            ) from exc

        finally:
            for temporary in (temporary_index, temporary_manifest):
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    @classmethod
    def load(
        cls,
        index_path: str | Path,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_revision: str | None = None,
    ) -> FaissVectorStore:
        """Load and validate a persisted index/metadata pair."""
        target = Path(index_path)
        sidecar = _sidecar_path(target)

        if not target.is_file():
            raise FileNotFoundError(f"FAISS index not found: {target}")

        if not sidecar.is_file():
            raise FaissStoreIntegrityError(
                f"metadata sidecar not found: {sidecar}"
            )

        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FaissStoreIntegrityError(
                "metadata sidecar is not readable valid JSON"
            ) from exc

        if not isinstance(manifest, dict):
            raise FaissStoreIntegrityError("metadata manifest must be an object")

        if manifest.get("format_version") != INDEX_FORMAT_VERSION:
            raise FaissStoreIntegrityError(
                "unsupported metadata format version"
            )

        if manifest.get("index_type") != INDEX_TYPE:
            raise FaissStoreIntegrityError("index type does not match baseline")

        if manifest.get("metric") != INDEX_METRIC:
            raise FaissStoreIntegrityError("index metric does not match baseline")

        if manifest.get("embedding_model") != embedding_model:
            raise FaissStoreIntegrityError(
                "embedding model does not match the requested model"
            )

        if manifest.get("embedding_revision") != embedding_revision:
            raise FaissStoreIntegrityError(
                "embedding revision does not match the requested revision"
            )

        expected_digest = manifest.get("index_sha256")

        if not isinstance(expected_digest, str) or not expected_digest:
            raise FaissStoreIntegrityError("index checksum is missing")

        try:
            actual_digest = _sha256_file(target)
        except OSError as exc:
            raise FaissStoreIntegrityError(
                "could not read the persisted FAISS index"
            ) from exc

        if actual_digest != expected_digest:
            raise FaissStoreIntegrityError(
                "index checksum does not match its metadata sidecar"
            )

        try:
            faiss = _load_faiss()
            index = faiss.read_index(str(target))
        except FaissStoreError:
            raise
        except Exception as exc:
            raise FaissStoreIntegrityError(
                "persisted file could not be read as a FAISS index"
            ) from exc

        if not isinstance(index, faiss.IndexFlatIP):
            raise FaissStoreIntegrityError(
                "persisted index is not a FAISS IndexFlatIP"
            )

        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise FaissStoreIntegrityError(
                "persisted index does not use inner-product similarity"
            )

        raw_records = manifest.get("records")
        if not isinstance(raw_records, list):
            raise FaissStoreIntegrityError(
                "metadata records must be a JSON array"
            )

        try:
            records = [
                ChunkRecord.model_validate(raw_record)
                for raw_record in raw_records
            ]
        except Exception as exc:
            raise FaissStoreIntegrityError(
                "metadata contains an invalid chunk record"
            ) from exc

        chunk_ids = [record.chunk_id for record in records]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise FaissStoreIntegrityError(
                "metadata contains duplicate chunk IDs"
            )

        vector_count = manifest.get("vector_count")
        dimension = manifest.get("embedding_dimension")

        if (
            isinstance(vector_count, bool)
            or not isinstance(vector_count, int)
            or vector_count != index.ntotal
            or len(records) != index.ntotal
        ):
            raise FaissStoreIntegrityError(
                "FAISS vector count does not match metadata"
            )

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
            or dimension != index.d
        ):
            raise FaissStoreIntegrityError(
                "FAISS dimension does not match metadata"
            )

        if index.ntotal < 1:
            raise FaissStoreIntegrityError(
                "persisted index must contain at least one vector"
            )

        store = cls(
            embedding_model=embedding_model,
            embedding_revision=embedding_revision,
        )
        store._index = index
        store._records = records
        store._dimension = dimension

        return store