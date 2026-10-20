"""HTTP routes for document ingestion."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.adapters.pdf.extractor import (
    PDFExtractionError,
    PDFPageLimitError,
    PDFTooLargeError,
)
from app.core.config import get_settings
from app.schemas.document import PDFExtractionResult
from app.services.pdf_ingestion import ingest_pdf_upload

UPLOAD_READ_CHUNK_SIZE = 64 * 1024

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


async def _read_upload_limited(
    upload: UploadFile,
    max_upload_bytes: int,
) -> bytes:
    """Read an upload incrementally without exceeding the configured limit."""
    chunks: list[bytes] = []
    total_bytes = 0

    while True:
        chunk = await upload.read(UPLOAD_READ_CHUNK_SIZE)

        if not chunk:
            break

        total_bytes += len(chunk)

        if total_bytes > max_upload_bytes:
            raise PDFTooLargeError(
                f"The upload exceeds the {max_upload_bytes}-byte limit."
            )

        chunks.append(chunk)

    return b"".join(chunks)


def _error_detail(error: PDFExtractionError) -> dict[str, str]:
    """Return a stable error payload without exposing parser internals."""
    return {
        "code": error.code,
        "message": str(error),
    }


@router.post(
    "/extract",
    response_model=PDFExtractionResult,
    status_code=status.HTTP_200_OK,
    summary="Extract text from a PDF",
    description=(
        "Extracts selectable text from a PDF and returns page-level text "
        "with one-based page numbers. OCR is not enabled."
    ),
)
async def extract_document(
    file: Annotated[
        UploadFile,
        File(description="PDF file to extract"),
    ],
) -> PDFExtractionResult:
    """Validate and extract an uploaded PDF."""
    try:
        settings = get_settings()

        pdf_bytes = await _read_upload_limited(
            upload=file,
            max_upload_bytes=settings.max_upload_bytes,
        )

        return ingest_pdf_upload(
            pdf_bytes=pdf_bytes,
            filename=file.filename or "",
        )

    except (PDFTooLargeError, PDFPageLimitError) as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=_error_detail(exc),
        ) from exc

    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=_error_detail(exc),
        ) from exc

    finally:
        await file.close()
