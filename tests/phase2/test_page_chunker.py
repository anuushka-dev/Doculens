from dataclasses import dataclass

import pytest

from app.services.page_chunker import chunk_pages


@dataclass(frozen=True)
class Page:
    page_number: int
    text: str


def test_chunks_are_deterministic_and_preserve_source_metadata() -> None:
    pages = [Page(1, "one two three four five six seven eight nine ten")]

    first = chunk_pages(
        pages,
        document_id="doc-1",
        file_name="guide.pdf",
        chunk_size_words=4,
        overlap_words=1,
    )
    second = chunk_pages(
        pages,
        document_id="doc-1",
        file_name="guide.pdf",
        chunk_size_words=4,
        overlap_words=1,
    )

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]
    assert [chunk.text for chunk in first] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert [chunk.chunk_index for chunk in first] == [0, 1, 2]
    assert all(chunk.page_number == 1 for chunk in first)
    assert all(
        chunk.document_id == "doc-1" and chunk.file_name == "guide.pdf"
        for chunk in first
    )


def test_chunk_offsets_slice_original_page_exactly() -> None:
    page_text = "  Alpha beta.\nGamma\tdelta  "

    chunks = chunk_pages(
        [Page(4, page_text)],
        document_id="d",
        file_name="x.pdf",
        chunk_size_words=2,
        overlap_words=0,
    )

    assert len(chunks) == 2

    for chunk in chunks:
        assert page_text[chunk.start_char:chunk.end_char] == chunk.text

    assert [chunk.page_number for chunk in chunks] == [4, 4]
    assert chunks[0].text == "Alpha beta."
    assert chunks[1].text == "Gamma\tdelta"


def test_empty_pages_are_skipped_without_renumbering() -> None:
    chunks = chunk_pages(
        [
            Page(1, "first page text"),
            Page(2, "   \n"),
            Page(3, "third page text"),
        ],
        document_id="d",
        file_name="x.pdf",
        chunk_size_words=10,
        overlap_words=0,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 3]


def test_chunks_never_cross_pages() -> None:
    chunks = chunk_pages(
        [Page(2, "alpha beta"), Page(3, "gamma delta")],
        document_id="d",
        file_name="x.pdf",
        chunk_size_words=3,
        overlap_words=0,
    )

    assert [(chunk.page_number, chunk.text) for chunk in chunks] == [
        (2, "alpha beta"),
        (3, "gamma delta"),
    ]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (3, 3),
        (3, -1),
        (True, 0),
        (3, False),
        (2.5, 0),
    ],
)
def test_rejects_invalid_chunk_configuration(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        chunk_pages(
            [Page(1, "some text here")],
            document_id="d",
            file_name="x.pdf",
            chunk_size_words=chunk_size,
            overlap_words=overlap,
        )


def test_rejects_duplicate_page_ids() -> None:
    with pytest.raises(ValueError, match="duplicate page_number"):
        chunk_pages(
            [Page(1, "first"), Page(1, "second")],
            document_id="d",
            file_name="x.pdf",
        )


def test_rejects_out_of_order_pages() -> None:
    with pytest.raises(ValueError, match="ascending page_number"):
        chunk_pages(
            [Page(2, "second"), Page(1, "first")],
            document_id="d",
            file_name="x.pdf",
        )


@pytest.mark.parametrize("page_number", [0, -1, True, 1.5])
def test_rejects_invalid_page_numbers(page_number: int) -> None:
    with pytest.raises(ValueError, match="page_number"):
        chunk_pages(
            [Page(page_number, "text")],
            document_id="d",
            file_name="x.pdf",
        )


def test_rejects_non_string_page_text() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        chunk_pages(
            [Page(1, None)],  # type: ignore[arg-type]
            document_id="d",
            file_name="x.pdf",
        )


@pytest.mark.parametrize("document_id", ["", "   ", None])
def test_rejects_blank_or_non_string_document_id(
    document_id: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="document_id"):
        chunk_pages(
            [Page(1, "text")],
            document_id=document_id,  # type: ignore[arg-type]
            file_name="x.pdf",
        )


@pytest.mark.parametrize(
    "file_name",
    [
        "",
        "   ",
        "../guide.pdf",
        "dir\\guide.pdf",
        "bad\nname.pdf",
    ],
)
def test_rejects_blank_or_unsafe_file_name(file_name: str) -> None:
    with pytest.raises(ValueError, match="file_name"):
        chunk_pages(
            [Page(1, "text")],
            document_id="d",
            file_name=file_name,
        )


def test_normalizes_identity_strings_before_creating_chunk_ids() -> None:
    page = [Page(1, "alpha beta")]

    normalized = chunk_pages(
        page,
        document_id="doc",
        file_name="guide.pdf",
        chunk_size_words=2,
        overlap_words=0,
    )
    padded = chunk_pages(
        page,
        document_id=" doc ",
        file_name=" guide.pdf ",
        chunk_size_words=2,
        overlap_words=0,
    )

    assert normalized == padded


def test_exact_chunk_size_produces_one_chunk() -> None:
    chunks = chunk_pages(
        [Page(7, "alpha beta gamma")],
        document_id="d",
        file_name="x.pdf",
        chunk_size_words=3,
        overlap_words=1,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "alpha beta gamma"


def test_rejects_invalid_source_filename_even_for_empty_pages() -> None:
    with pytest.raises(ValueError, match="file_name"):
        chunk_pages(
            [],
            document_id="d",
            file_name="../unsafe.pdf",
        )
