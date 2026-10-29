"""Local sentence-transformer embeddings for DocuLens retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EmbeddingVector = NDArray[np.float32]
EmbeddingMatrix = NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Base class for expected embedding adapter failures."""


class EmbeddingDependencyError(EmbeddingError):
    """The sentence-transformers runtime is not installed."""


class EmbeddingModelLoadError(EmbeddingError):
    """The configured embedding model could not be loaded."""


class EmbeddingInferenceError(EmbeddingError):
    """The model failed while encoding input text."""


class EmbeddingOutputError(EmbeddingError):
    """The model returned an invalid embedding tensor."""


class SentenceEncoder(Protocol):
    """Small interface required from the sentence-transformers model."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        """Encode text into a two-dimensional matrix."""


ModelLoader = Callable[..., SentenceEncoder]


class SentenceTransformerEmbedder:
    """Load a sentence-transformer model lazily and return normalized vectors.

    A fake encoder can be injected for testing, avoiding model downloads.
    Successful outputs are finite, non-zero, L2-normalized float32 vectors.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        model: SentenceEncoder | None = None,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-blank string")

        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("revision must be None or a non-blank string")

        if device is not None and (
            not isinstance(device, str) or not device.strip()
        ):
            raise ValueError("device must be None or a non-blank string")

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        if model is not None and model_loader is not None:
            raise ValueError("provide either model or model_loader, not both")

        self.model_name = model_name.strip()
        self.revision = revision.strip() if revision is not None else None
        self.device = device.strip() if device is not None else None
        self.batch_size = batch_size

        self._model = model
        self._model_loader = model_loader
        self._embedding_dimension: int | None = None

    @property
    def embedding_dimension(self) -> int | None:
        """Return the observed output dimension after successful encoding."""
        return self._embedding_dimension

    def _get_model(self) -> SentenceEncoder:
        """Load the model once and report loading failures clearly."""
        if self._model is not None:
            return self._model

        loader = self._model_loader

        if loader is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingDependencyError(
                    "Install sentence-transformers to use local embeddings."
                ) from exc

            loader = SentenceTransformer

        kwargs: dict[str, str] = {}

        if self.revision is not None:
            kwargs["revision"] = self.revision

        if self.device is not None:
            kwargs["device"] = self.device

        try:
            self._model = loader(self.model_name, **kwargs)
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Could not load embedding model {self.model_name!r}."
            ) from exc

        return self._model

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> EmbeddingMatrix:
        """Encode one or more texts into a normalized float32 matrix."""
        if isinstance(texts, (str, bytes)):
            raise TypeError(
                "texts must be a sequence of strings, not one string"
            )

        if len(texts) == 0:
            raise ValueError("texts must contain at least one item")

        normalized_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] must be a string")

            if not text.strip():
                raise ValueError(f"texts[{index}] must not be blank")

            normalized_texts.append(text)

        selected_batch_size = (
            self.batch_size if batch_size is None else batch_size
        )
        self._validate_batch_size(selected_batch_size)

        try:
            raw_vectors = self._get_model().encode(
                normalized_texts,
                batch_size=selected_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingInferenceError(
                "The embedding model failed while encoding text."
            ) from exc

        try:
            vectors = np.asarray(raw_vectors, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingOutputError(
                "The embedding model did not return a numeric matrix."
            ) from exc

        if vectors.ndim != 2:
            raise EmbeddingOutputError(
                "Expected a 2D embedding matrix; "
                f"received {vectors.ndim}D output."
            )

        if vectors.shape[0] != len(normalized_texts):
            raise EmbeddingOutputError(
                "Embedding row count must match the number of input texts."
            )

        if vectors.shape[1] < 1:
            raise EmbeddingOutputError("Embedding dimension must be positive.")

        if not np.isfinite(vectors).all():
            raise EmbeddingOutputError(
                "Embedding output contains NaN or infinite values."
            )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        if not np.isfinite(norms).all():
            raise EmbeddingOutputError(
                "Embedding vector norms are not finite."
            )

        if np.any(norms <= 0.0):
            raise EmbeddingOutputError(
                "Embedding output contains a zero-length vector."
            )

        if (
            self._embedding_dimension is not None
            and vectors.shape[1] != self._embedding_dimension
        ):
            raise EmbeddingOutputError(
                "Embedding dimension changed between encode calls."
            )

        normalized_vectors = np.asarray(vectors / norms, dtype=np.float32)
        self._embedding_dimension = int(normalized_vectors.shape[1])

        return normalized_vectors

    def embed_query(self, query: str) -> EmbeddingVector:
        """Encode one non-blank search query into a normalized vector."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be blank")

        return self.embed_texts([query])[0]