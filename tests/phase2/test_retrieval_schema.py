"""Contract tests for Phase 2 retrieval schemas."""

import math

import pytest
from pydantic import ValidationError

from app.schemas.retrieval import ChunkRecord, RetrievalResponse, RetrievedChunk


def make_chunk(**overrides: object) -> ChunkRecord:
    """Create a valid baseline chunk for focused schema tests."""
    payload: dict[str, object] = {
        "chunk_id": "chunk_1",
        "document_id": "doc",
        "file_name": "a.pdf",
        "page_number": 3,
        "chunk_index": 0,
        "start_char": 2,
        "end_char": 7,
        "text": "hello",
    }
    payload.update(overrides)
    return ChunkRecord.model_validate(payload)


def make_result(*, chunk_id: str = "chunk_1", score: float = 0.9) -> RetrievedChunk:
    """Create a valid ranked chunk, including required provenance."""
    return RetrievedChunk(
        **make_chunk(chunk_id=chunk_id).model_dump(),
        score=score,
    )


def test_chunk_record_preserves_page_identity_and_offsets() -> None:
    chunk = make_chunk()
    assert chunk.page_number == 3
    assert chunk.start_char == 2
    assert chunk.end_char == 7
    assert chunk.end_char - chunk.start_char == len(chunk.text)


def test_chunk_record_trims_identity_fields_but_not_source_text() -> None:
    chunk = make_chunk(
        chunk_id=" chunk_1 ",
        document_id=" doc ",
        file_name=" a.pdf ",
        text=" hello",
        start_char=2,
        end_char=8,
    )
    assert chunk.chunk_id == "chunk_1"
    assert chunk.document_id == "doc"
    assert chunk.file_name == "a.pdf"
    assert chunk.text == " hello"


def test_chunk_record_rejects_invalid_page_empty_text_and_bad_offsets() -> None:
    with pytest.raises(ValidationError):
        make_chunk(page_number=0)
    with pytest.raises(ValidationError):
        make_chunk(text=" ")
    with pytest.raises(ValidationError):
        make_chunk(start_char=0, end_char=4)
    with pytest.raises(ValidationError):
        make_chunk(start_char=7, end_char=7)


def test_chunk_record_rejects_blank_identity_and_path_filename() -> None:
    with pytest.raises(ValidationError):
        make_chunk(document_id="   ")
    with pytest.raises(ValidationError):
        make_chunk(file_name="docs/a.pdf")


def test_chunk_record_rejects_bool_where_integer_is_required() -> None:
    with pytest.raises(ValidationError):
        make_chunk(page_number=True)


def test_retrieved_chunk_reuses_chunk_validation_and_accepts_finite_score() -> None:
    result = make_result(score=0.75)
    assert result.score == 0.75
    assert result.page_number == 3
    assert result.text == "hello"


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf, 1.1, -1.1])
def test_retrieved_chunk_rejects_invalid_similarity_score(score: float) -> None:
    with pytest.raises(ValidationError):
        make_result(score=score)


def test_retrieval_response_accepts_ranked_results() -> None:
    response = RetrievalResponse(
        query="What does the document say?",
        top_k=2,
        results=(
            make_result(chunk_id="first", score=0.9),
            make_result(chunk_id="second", score=0.7),
        ),
    )
    assert len(response.results) == 2
    assert response.results[0].score > response.results[1].score


def test_retrieval_response_rejects_blank_query_too_many_or_duplicate_results() -> None:
    first = make_result(chunk_id="same", score=0.9)
    second = make_result(chunk_id="second", score=0.8)
    with pytest.raises(ValidationError):
        RetrievalResponse(query="  ", top_k=2, results=())
    with pytest.raises(ValidationError):
        RetrievalResponse(query="q", top_k=1, results=(first, second))
    with pytest.raises(ValidationError):
        RetrievalResponse(
            query="q",
            top_k=2,
            results=(first, first),
        )


def test_retrieval_response_rejects_unsorted_results() -> None:
    with pytest.raises(ValidationError):
        RetrievalResponse(
            query="q",
            top_k=2,
            results=(
                make_result(chunk_id="lower", score=0.2),
                make_result(chunk_id="higher", score=0.8),
            ),
        )


def test_schemas_are_immutable_and_reject_extra_fields() -> None:
    chunk = make_chunk()
    with pytest.raises(ValidationError):
        chunk.page_number = 4  # type: ignore[misc]
    with pytest.raises(ValidationError):
        make_chunk(unknown_field="not allowed")
