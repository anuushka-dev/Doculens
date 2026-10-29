"""Tests for the sentence-transformer embedding adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.adapters.embeddings.sentence_transformer import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingInferenceError,
    EmbeddingModelLoadError,
    EmbeddingOutputError,
    SentenceTransformerEmbedder,
)


class FakeEncoder:
    """Small deterministic encoder that never downloads a model."""

    def __init__(
        self,
        output: Any = None,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
            }
        )

        if self.error is not None:
            raise self.error

        if self.output is not None:
            return self.output

        return np.tile(
            np.array([3.0, 4.0], dtype=np.float32),
            (len(sentences), 1),
        )


def test_default_model_name() -> None:
    embedder = SentenceTransformerEmbedder()

    assert embedder.model_name == DEFAULT_EMBEDDING_MODEL
    assert embedder.embedding_dimension is None


def test_returns_finite_l2_normalized_float32_vectors() -> None:
    model = FakeEncoder(
        output=np.array(
            [
                [3.0, 4.0],
                [5.0, 12.0],
            ],
            dtype=np.float64,
        )
    )
    embedder = SentenceTransformerEmbedder(model=model)

    vectors = embedder.embed_texts(["first document", "second document"])

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1),
        np.ones(2, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert embedder.embedding_dimension == 2


def test_preserves_input_order_and_encode_options() -> None:
    model = FakeEncoder()
    embedder = SentenceTransformerEmbedder(model=model, batch_size=16)

    embedder.embed_texts(["first", "second"], batch_size=7)

    assert len(model.calls) == 1
    assert model.calls[0] == {
        "sentences": ["first", "second"],
        "batch_size": 7,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
    }


def test_embed_query_returns_one_dimensional_vector() -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    vector = embedder.embed_query("find the relevant section")

    assert vector.shape == (2,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_model_is_loaded_lazily_and_only_once() -> None:
    model = FakeEncoder()
    loader_calls: list[tuple[str, dict[str, str]]] = []

    def loader(
        model_name: str,
        **kwargs: str,
    ) -> FakeEncoder:
        loader_calls.append((model_name, kwargs))
        return model

    embedder = SentenceTransformerEmbedder(
        model_name="example/model",
        revision="main",
        device="cpu",
        model_loader=loader,
    )

    assert loader_calls == []

    embedder.embed_texts(["first"])
    embedder.embed_texts(["second"])

    assert loader_calls == [
        (
            "example/model",
            {"revision": "main", "device": "cpu"},
        )
    ]
    assert len(model.calls) == 2


def test_rejects_empty_text_sequence() -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    with pytest.raises(ValueError, match="at least one"):
        embedder.embed_texts([])


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_rejects_blank_text(text: str) -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    with pytest.raises(ValueError, match="must not be blank"):
        embedder.embed_texts([text])


@pytest.mark.parametrize("texts", ["one string", b"one string", [123]])
def test_rejects_non_sequence_or_non_string_texts(texts: Any) -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    with pytest.raises(TypeError):
        embedder.embed_texts(texts)


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_rejects_invalid_constructor_batch_size(
    batch_size: Any,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        SentenceTransformerEmbedder(batch_size=batch_size)


@pytest.mark.parametrize(
    ("batch_size", "error_type"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_rejects_invalid_encode_batch_size(
    batch_size: Any,
    error_type: type[Exception],
) -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    with pytest.raises(error_type):
        embedder.embed_texts(["some text"], batch_size=batch_size)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (
            np.array([1.0, 2.0], dtype=np.float32),
            "2D",
        ),
        (
            np.ones((1, 3), dtype=np.float32),
            "row count",
        ),
        (
            np.array(
                [[np.nan, 1.0], [1.0, 2.0]],
                dtype=np.float32,
            ),
            "NaN or infinite",
        ),
        (
            np.array(
                [[np.inf, 1.0], [1.0, 2.0]],
                dtype=np.float32,
            ),
            "NaN or infinite",
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            "zero-length",
        ),
        (
            [["not", "numeric"], ["also", "not numeric"]],
            "numeric matrix",
        ),
    ],
)
def test_rejects_invalid_model_outputs(
    output: Any,
    message: str,
) -> None:
    embedder = SentenceTransformerEmbedder(
        model=FakeEncoder(output=output)
    )

    with pytest.raises(EmbeddingOutputError, match=message):
        embedder.embed_texts(["first", "second"])


def test_rejects_embedding_dimension_change() -> None:
    class ChangingEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode(
            self,
            sentences: list[str],
            *,
            batch_size: int,
            show_progress_bar: bool,
            convert_to_numpy: bool,
            normalize_embeddings: bool,
        ) -> object:
            self.calls += 1
            dimension = 2 if self.calls == 1 else 3
            return np.ones(
                (len(sentences), dimension),
                dtype=np.float32,
            )

    embedder = SentenceTransformerEmbedder(model=ChangingEncoder())

    embedder.embed_texts(["first"])

    with pytest.raises(EmbeddingOutputError, match="dimension changed"):
        embedder.embed_texts(["second"])


def test_wraps_model_loading_failure() -> None:
    def failing_loader(model_name: str, **kwargs: str) -> FakeEncoder:
        raise RuntimeError("model unavailable")

    embedder = SentenceTransformerEmbedder(model_loader=failing_loader)

    with pytest.raises(EmbeddingModelLoadError, match="Could not load") as exc:
        embedder.embed_texts(["document"])

    assert isinstance(exc.value.__cause__, RuntimeError)


def test_wraps_model_inference_failure() -> None:
    model = FakeEncoder(error=RuntimeError("inference failed"))
    embedder = SentenceTransformerEmbedder(model=model)

    with pytest.raises(EmbeddingInferenceError, match="failed while encoding"):
        embedder.embed_texts(["document"])


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_rejects_blank_query(query: str) -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    with pytest.raises(ValueError, match="must not be blank"):
        embedder.embed_query(query)


def test_rejects_non_string_query() -> None:
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    with pytest.raises(TypeError, match="query must be a string"):
        embedder.embed_query(123)  # type: ignore[arg-type]


def test_rejects_invalid_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        SentenceTransformerEmbedder(model_name="   ")


def test_rejects_blank_revision_and_device() -> None:
    with pytest.raises(ValueError, match="revision"):
        SentenceTransformerEmbedder(revision=" ")

    with pytest.raises(ValueError, match="device"):
        SentenceTransformerEmbedder(device=" ")