"""Application service for PDF ingestion."""

from app.adapters.pdf.extractor import extract_pdf_bytes
from app.core.config import get_settings
from app.schemas.document import PDFExtractionResult


def ingest_pdf_upload(
    pdf_bytes: bytes,
    filename: str,
) -> PDFExtractionResult:
    """Ingest an uploaded PDF using the validated application settings.

    This service coordinates configuration and PDF extraction. It does not
    implement PDF parsing or translate extraction failures into HTTP errors.

    Args:
        pdf_bytes: Raw bytes read from the uploaded file.
        filename: Original upload filename, used only as display metadata.

    Returns:
        A validated result containing document metadata and page-level text.

    Raises:
        TypeError: The input bytes or configured resource limits have an
            invalid type.
        ValueError: A resource limit is invalid.
        PDFExtractionError subclasses: The PDF is invalid, unsupported,
            encrypted, exceeds a limit, or contains no extractable text.
    """
    settings = get_settings()

    return extract_pdf_bytes(
        pdf_bytes=pdf_bytes,
        filename=filename,
        max_upload_bytes=settings.max_upload_bytes,
        max_pdf_pages=settings.max_pdf_pages,
    )
