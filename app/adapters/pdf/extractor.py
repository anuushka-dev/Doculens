
import hashlib
import unicodedata
from uuid import uuid4

import pymupdf

from app.schemas.document import ExtractedPage, PDFExtractionResult


class PDFExtractionError(Exception):
    """Base class for expected PDF ingestion failures."""

    code = "pdf_extraction_error"


class InvalidFilenameError(PDFExtractionError):
    """The upload does not have a usable display filename."""

    code = "invalid_filename"


class EmptyUploadError(PDFExtractionError):
    """The uploaded file contains no bytes."""

    code = "empty_upload"


class UnsupportedFileTypeError(PDFExtractionError):
    """The submitted filename does not identify a PDF."""

    code = "unsupported_file_type"


class PDFTooLargeError(PDFExtractionError):
    """The uploaded PDF exceeds the configured byte limit."""

    code = "pdf_too_large"


class InvalidPDFError(PDFExtractionError):
    """The PDF header, structure, or metadata is invalid."""

    code = "invalid_pdf"


class EncryptedPDFError(PDFExtractionError):
    """The PDF requires a password."""

    code = "encrypted_pdf"


class PDFPageLimitError(PDFExtractionError):
    """The PDF exceeds the configured page limit."""

    code = "pdf_page_limit_exceeded"


class NoExtractableTextError(PDFExtractionError):
    """The PDF contains no usable selectable text."""

    code = "no_extractable_text"


class PDFPageExtractionError(PDFExtractionError):
    """Text extraction failed for an individual page."""

    code = "page_extraction_failed"

    def __init__(self, page_number: int) -> None:
        self.page_number = page_number
        super().__init__(
            f"Text extraction failed on page {page_number}."
        )


def _validate_resource_limit(value: int, name: str) -> None:
    """Reject invalid limits before processing the document."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 1:
        raise ValueError(f"{name} must be positive.")


def _normalize_filename(filename: str) -> str:
    """Return a safe display basename, never a filesystem path."""
    if not isinstance(filename, str) or not filename.strip():
        raise InvalidFilenameError("A filename is required.")

    if any(not character.isprintable() for character in filename):
        raise InvalidFilenameError(
            "The filename contains unsupported control characters."
        )

    # Handle both POSIX and Windows-style paths.
    basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    basename = basename.strip()

    if not basename or basename in {".", ".."}:
        raise InvalidFilenameError("A valid filename is required.")

    if not basename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError(
            "Only PDF files are supported."
        )

    # Bound display metadata while retaining the PDF extension.
    if len(basename) > 255:
        extension = basename[-4:]
        stem = basename[:-4]
        basename = f"{stem[: 255 - len(extension)]}{extension}"

    return basename


def _normalize_page_text(text: str) -> str:
    """Normalize text without merging page boundaries or collapsing layout."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\r\n", "\n")
    normalized = normalized.replace("\r", "\n")
    normalized = normalized.replace("\x0c", "\n")

    lines: list[str] = []
    previous_line_was_blank = False

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()

        if line.strip():
            # Preserve indentation and internal spacing, which may matter
            # for code samples, aligned text, and tables.
            lines.append(line)
            previous_line_was_blank = False
        elif lines and not previous_line_was_blank:
            # Keep one separator between text blocks, not long blank runs.
            lines.append("")
            previous_line_was_blank = True

    return "\n".join(lines).rstrip("\n")


def extract_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    *,
    max_upload_bytes: int,
    max_pdf_pages: int,
) -> PDFExtractionResult:
    """Validate a PDF and return immutable, page-aware extraction records.

    This function owns PDF-specific validation and parsing only.

    Raises:
        TypeError: Input bytes or resource limits have the wrong type.
        ValueError: A resource limit is not positive.
        PDFExtractionError: The upload cannot be processed as a supported PDF.
    """
    if not isinstance(pdf_bytes, bytes):
        raise TypeError("pdf_bytes must be bytes.")

    _validate_resource_limit(max_upload_bytes, "max_upload_bytes")
    _validate_resource_limit(max_pdf_pages, "max_pdf_pages")

    safe_filename = _normalize_filename(filename)

    if not pdf_bytes:
        raise EmptyUploadError("The uploaded file is empty.")

    # Check size before invoking the parser.
    if len(pdf_bytes) > max_upload_bytes:
        raise PDFTooLargeError(
            f"The PDF exceeds the {max_upload_bytes}-byte upload limit."
        )

    # The PDF header may occur within the first 1,024 bytes.
    # This quick check is followed by PyMuPDF's structural parsing.
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise InvalidPDFError(
            "The uploaded content does not have a valid PDF header."
        )

    document: pymupdf.Document | None = None

    try:
        try:
            document = pymupdf.open(
                stream=pdf_bytes,
                filetype="pdf",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidPDFError(
                "The PDF could not be opened or parsed."
            ) from exc

        # A malformed document can fail while exposing metadata as well
        # as while opening. Keep those failures inside this parser boundary.
        try:
            if document.needs_pass:
                raise EncryptedPDFError(
                    "Password-protected PDFs are not supported."
                )

            page_count = document.page_count
        except PDFExtractionError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise InvalidPDFError(
                "The PDF metadata could not be read."
            ) from exc

        if page_count < 1:
            raise InvalidPDFError("The PDF contains no pages.")

        # Reject excessive page counts before extracting page content.
        if page_count > max_pdf_pages:
            raise PDFPageLimitError(
                f"The PDF contains {page_count} pages; "
                f"the configured limit is {max_pdf_pages}."
            )

        pages: list[ExtractedPage] = []

        for page_index in range(page_count):
            page_number = page_index + 1

            try:
                page = document.load_page(page_index)
                extracted_text = page.get_text("text") or ""

                if not isinstance(extracted_text, str):
                    raise TypeError(
                        "The PDF parser returned a non-string page result."
                    )

            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                raise PDFPageExtractionError(page_number) from exc

            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text=_normalize_page_text(extracted_text),
                )
            )

        # Preserve empty pages in mixed documents so later source page IDs
        # do not shift. Reject only when the entire document has no text.
        if not any(page.text.strip() for page in pages):
            raise NoExtractableTextError(
                "No extractable text was found. This release supports "
                "text-selectable PDFs; OCR is not enabled."
            )

        return PDFExtractionResult(
            document_id=uuid4(),
            filename=safe_filename,
            sha256=hashlib.sha256(pdf_bytes).hexdigest(),
            pages=tuple(pages),
        )

    finally:
        if document is not None:
            document.close()
