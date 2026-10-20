
import hashlib

import pymupdf
import pytest

from app.adapters.pdf import extractor
from app.adapters.pdf.extractor import (
    EmptyUploadError,
    EncryptedPDFError,
    InvalidFilenameError,
    InvalidPDFError,
    NoExtractableTextError,
    PDFPageExtractionError,
    PDFPageLimitError,
    PDFTooLargeError,
    UnsupportedFileTypeError,
    extract_pdf_bytes,
)

DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_MAX_PAGES = 10


def make_pdf(*page_texts: str, encrypted: bool = False) -> bytes:
    """Generate a small PDF fixture entirely in memory."""
    document = pymupdf.open()

    try:
        for text in page_texts:
            page = document.new_page()

            if text:
                page.insert_text((72, 72), text)

        if encrypted:
            return document.tobytes(
                encryption=pymupdf.PDF_ENCRYPT_AES_256,
                owner_pw="test-owner",
                user_pw="test-user",
            )

        return document.tobytes()
    finally:
        document.close()


def extract(
    content: bytes,
    filename: str = "sample.pdf",
    *,
    max_upload_bytes: int = DEFAULT_MAX_BYTES,
    max_pdf_pages: int = DEFAULT_MAX_PAGES,
):
    """Call the extractor with consistent test limits."""
    return extract_pdf_bytes(
        content,
        filename,
        max_upload_bytes=max_upload_bytes,
        max_pdf_pages=max_pdf_pages,
    )


def test_extracts_text_with_one_based_page_numbers_and_hash() -> None:
    content = make_pdf(
        "First page content.",
        "Second page content.",
    )

    result = extract(content)

    assert result.page_count == 2
    assert [page.page_number for page in result.pages] == [1, 2]
    assert "First page content." in result.pages[0].text
    assert "Second page content." in result.pages[1].text
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.filename == "sample.pdf"
    assert result.total_character_count > 0


def test_creates_a_unique_document_id() -> None:
    content = make_pdf("Some useful content.")

    first_result = extract(content)
    second_result = extract(content)

    assert first_result.document_id != second_result.document_id
    assert first_result.sha256 == second_result.sha256


def test_reduces_filename_path_to_display_basename() -> None:
    content = make_pdf("Useful content.")

    result = extract(content, r"..\private\notes.pdf")

    assert result.filename == "notes.pdf"


@pytest.mark.parametrize(
    "filename",
    ["", "   ", ".", "..", "notes.txt", "notes.pdf\x00"],
)
def test_rejects_invalid_filenames(filename: str) -> None:
    content = make_pdf("Useful content.")

    with pytest.raises(
        (InvalidFilenameError, UnsupportedFileTypeError)
    ):
        extract(content, filename)


def test_rejects_non_bytes_input() -> None:
    with pytest.raises(TypeError, match="pdf_bytes must be bytes"):
        extract("not bytes")  # type: ignore[arg-type]


def test_rejects_empty_upload() -> None:
    with pytest.raises(EmptyUploadError):
        extract(b"")


def test_rejects_content_without_pdf_header() -> None:
    with pytest.raises(InvalidPDFError):
        extract(b"This is not a PDF.", "sample.pdf")


def test_rejects_malformed_pdf() -> None:
    malformed = (
        b"%PDF-1.7\n"
        b"This is not a valid PDF structure.\n"
        b"%%EOF"
    )

    with pytest.raises(InvalidPDFError):
        extract(malformed)


def test_rejects_upload_over_size_limit() -> None:
    content = make_pdf("Content.")

    with pytest.raises(PDFTooLargeError):
        extract(content, max_upload_bytes=len(content) - 1)


@pytest.mark.parametrize(
    ("max_upload_bytes", "max_pdf_pages"),
    [(0, 10), (1_000, 0)],
)
def test_rejects_nonpositive_resource_limits(
    max_upload_bytes: int,
    max_pdf_pages: int,
) -> None:
    content = make_pdf("Content.")

    with pytest.raises(ValueError):
        extract(
            content,
            max_upload_bytes=max_upload_bytes,
            max_pdf_pages=max_pdf_pages,
        )


def test_rejects_document_over_page_limit() -> None:
    content = make_pdf("Page one.", "Page two.")

    with pytest.raises(PDFPageLimitError):
        extract(content, max_pdf_pages=1)


def test_rejects_password_protected_pdf() -> None:
    content = make_pdf("Protected content.", encrypted=True)

    with pytest.raises(EncryptedPDFError):
        extract(content)


def test_rejects_document_with_no_extractable_text() -> None:
    content = make_pdf("", "")

    with pytest.raises(NoExtractableTextError):
        extract(content)


def test_preserves_empty_pages_in_a_mixed_document() -> None:
    content = make_pdf(
        "Page one has content.",
        "",
        "Page three has content.",
    )

    result = extract(content)

    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert result.empty_page_numbers == (2,)
    assert result.pages[1].status == "empty_text"


def test_reports_near_empty_pages() -> None:
    content = make_pdf(
        "A",
        "This page contains enough useful text to pass the threshold.",
    )

    result = extract(content)

    assert result.near_empty_page_numbers == (1,)
    assert result.pages[0].status == "near_empty_text"
    assert result.pages[1].status == "ok"


def test_page_extraction_failure_has_safe_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Generate valid input before patching the parser.
    content = make_pdf("Valid source content.")

    class FakePage:
        def get_text(self, _mode: str) -> str:
            raise RuntimeError("internal parser detail")

    class FakeDocument:
        needs_pass = False
        page_count = 1

        def load_page(self, _index: int) -> FakePage:
            return FakePage()

        def close(self) -> None:
            pass

    def fake_open(**_kwargs: object) -> FakeDocument:
        return FakeDocument()

    monkeypatch.setattr(extractor.pymupdf, "open", fake_open)

    with pytest.raises(PDFPageExtractionError) as captured:
        extract(content)

    assert captured.value.page_number == 1
    assert "internal parser detail" not in str(captured.value)
