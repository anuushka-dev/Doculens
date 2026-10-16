
from app.adapters.pdf.extractor import extract_pdf_bytes
from app.core.config import get_settings
from app.schemas.document import PDFExtractionResult


def ingest_pdf_upload(
    pdf_bytes: bytes,
    filename: str,
) -> PDFExtractionResult:
    """Ingest an uploaded PDF using the configured resource limits.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.
        filename: Original upload filename, used for display metadata.

    Returns:
        Validated extraction result with page-level text and provenance.

    Raises:
        PDFExtractionError: If the PDF is invalid, unsupported, or exceeds
            configured limits.
        ValueError: If the configured limits are invalid.
        TypeError: If the supplied PDF content is not bytes.
    """
    settings = get_settings()

    return extract_pdf_bytes(
        pdf_bytes,
        filename,
        max_upload_bytes=settings.max_upload_bytes,
        max_pdf_pages=settings.max_pdf_pages,
    )
