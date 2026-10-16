
import hashlib
import unicodedata
from uuid import uuid4

import pymupdf

from app.schemas.document import ExtractedPage, PDFExtractionResult


class PDFExtractionError(Exception):
    """Base class for expected PDF ingestion failures."""

    code = "pdf_extraction_error"


class InvalidFilenameError(PDFExtractionError):
    code = "invalid_filename"


class EmptyUploadError(PDFExtractionError):
    code = "empty_upload"


class UnsupportedFileTypeError(PDFExtractionError):
    code = "unsupported_file_type"


class PDFTooLargeError(PDFExtractionError):
    code = "pdf_too_large"


class InvalidPDFError(PDFExtractionError):
    code = "invalid_pdf"


class EncryptedPDFError(PDFExtractionError):
    code = "encrypted_pdf"


class PDFPageLimitError(PDFExtractionError):
    code = "pdf_page_limit_exceeded"


class NoExtractableTextError(PDFExtractionError):
    code = "no_extractable_text"


class PDFPageExtractionError(PDFExtractionError):
    code = "page_extraction_failed"

    def __init__(self, page_number: int) -> None:
        self.page_number = page_number
        super().__init__(
            f"Text extraction failed on page {page_number}."
        )


def _normalize_filename(filename: str) -> str:
    """Normalize an uploaded filename for display, never for file access."""
    if not isinstance(filename, str) or not filename.strip():
        raise InvalidFilenameError("A filename is required.")

    if any(not character.isprintable() for character in filename):
        raise InvalidFilenameError(
            "The filename contains unsupported control characters."
        )

    # Browsers and operating systems may supply different path separators.
    # Keep only the basename; never use the supplied name to write a file.
    basename = filename.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    basename = basename.strip()

    if not basename or basename in {".", ".."}:
        raise InvalidFilenameError("A valid filename is required.")

    if not basename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError(
            "Only PDF files are supported."
        )

    # Bound metadata length while preserving the original extension.
    if len(basename) > 255:
        extension = basename[-4:]
        stem = basename[:-4]
        basename = f"{stem[: 255 - len(extension)]}{extension}"

    return basename


def _normalize_page_text(text: str) -> str:
    """Normalize Unicode and line endings while preserving layout spacing."""
    if not text:
        return ""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x0c", "\n")

    lines: list[str] = []
    previous_line_was_blank = False

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()

        if line.strip():
            # Keep indentation and internal spacing for code and tables.
            lines.append(line)
            previous_line_was_blank = False
        elif lines and not previous_line_was_blank:
            # Avoid excessively long runs of blank lines.
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
    """Extract validated, page-aware text from an uploaded PDF.

    Raises:
        TypeError: If pdf_bytes is not bytes.
        ValueError: If configured resource limits are invalid.
        PDFExtractionError: If the document cannot be ingested.
    """
    if not isinstance(pdf_bytes, bytes):
        raise TypeError("pdf_bytes must be bytes.")

    if max_upload_bytes < 1:
        raise ValueError("max_upload_bytes must be positive.")

    if max_pdf_pages < 1:
        raise ValueError("max_pdf_pages must be positive.")

    safe_filename = _normalize_filename(filename)

    if not pdf_bytes:
        raise EmptyUploadError("The uploaded file is empty.")

    # Check size before invoking the PDF parser.
    if len(pdf_bytes) > max_upload_bytes:
        raise PDFTooLargeError(
            f"The PDF exceeds the {max_upload_bytes}-byte upload limit."
        )

    # Reject obvious non-PDF input before parsing. The parser performs
    # the definitive structural validation below.
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

        if document.needs_pass:
            raise EncryptedPDFError(
                "Password-protected PDFs are not supported."
            )

        page_count = document.page_count

        if page_count < 1:
            raise InvalidPDFError("The PDF contains no pages.")

        # Reject oversized documents before extracting individual pages.
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
            except (OSError, RuntimeError, ValueError) as exc:
                raise PDFPageExtractionError(page_number) from exc

            pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text=_normalize_page_text(extracted_text),
                )
            )

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

    except PDFExtractionError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        # Convert expected parser failures into a stable application error.
        raise InvalidPDFError(
            "The PDF could not be processed."
        ) from exc
    finally:
        if document is not None:
            document.close()
