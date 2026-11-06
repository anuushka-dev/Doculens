"""HTTP routes for grounded Q&A and structured document extraction."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, NoReturn

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.adapters.llm.ollama_client import (
    OllamaClient,
    OllamaClientError,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaOutputValidationError,
    OllamaProtocolError,
    OllamaTimeoutError,
)
from app.adapters.pdf.extractor import (
    PDFExtractionError,
    PDFPageLimitError,
    PDFTooLargeError,
)
from app.core.config import get_settings
from app.schemas.answer import AnswerResponse
from app.schemas.document import PDFExtractionResult
from app.schemas.extraction import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSchemaName,
)
from app.services.answer_generation import AnswerGenerationService
from app.services.pdf_ingestion import ingest_pdf_upload
from app.services.structured_extraction import (
    StructuredExtractionError,
    StructuredExtractionService,
)

UPLOAD_READ_CHUNK_SIZE = 64 * 1024
QUESTION_MAX_LENGTH = 2_000

router = APIRouter(
    prefix="/documents",
    tags=["document intelligence"],
)


def _create_ollama_client() -> OllamaClient:
    """Create an Ollama client using the application settings."""

    settings = get_settings()

    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.ollama_num_ctx,
        num_predict=settings.ollama_num_predict,
        temperature=settings.ollama_temperature,
        seed=settings.ollama_seed,
    )


async def _read_upload_limited(
    upload: UploadFile,
    max_upload_bytes: int,
) -> bytes:
    """Read an uploaded PDF without exceeding the configured size limit."""

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


async def _extract_uploaded_pdf(
    upload: UploadFile,
) -> PDFExtractionResult:
    """Use the existing Phase 1 ingestion pipeline for an uploaded PDF."""

    settings = get_settings()

    pdf_bytes = await _read_upload_limited(
        upload=upload,
        max_upload_bytes=settings.max_upload_bytes,
    )

    return ingest_pdf_upload(
        pdf_bytes=pdf_bytes,
        filename=upload.filename or "",
    )


def _raise_pdf_http_error(
    error: PDFExtractionError | PDFTooLargeError | PDFPageLimitError,
) -> NoReturn:
    """Map Phase 1 PDF errors to stable HTTP responses."""

    if isinstance(error, (PDFTooLargeError, PDFPageLimitError)):
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": error.code,
                "message": str(error),
            },
        ) from error

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail={
            "code": error.code,
            "message": str(error),
        },
    ) from error


def _raise_llm_http_error(
    error: OllamaClientError,
) -> NoReturn:
    """Map Ollama failures without exposing internal exception details."""

    if isinstance(error, OllamaTimeoutError):
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
        code = "OLLAMA_TIMEOUT"
        message = "The local language model did not respond in time."

    elif isinstance(error, OllamaConnectionError):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        code = "OLLAMA_UNAVAILABLE"
        message = "The local language model service is unavailable."

    elif isinstance(error, OllamaHTTPError):
        status_code = status.HTTP_502_BAD_GATEWAY
        code = "OLLAMA_HTTP_ERROR"
        message = "The local language model returned an unsuccessful response."

    elif isinstance(error, OllamaProtocolError):
        status_code = status.HTTP_502_BAD_GATEWAY
        code = "OLLAMA_PROTOCOL_ERROR"
        message = "The local language model returned an invalid response."

    elif isinstance(error, OllamaOutputValidationError):
        status_code = status.HTTP_502_BAD_GATEWAY
        code = "OLLAMA_OUTPUT_INVALID"
        message = "The generated output did not satisfy the required schema."

    else:
        status_code = status.HTTP_502_BAD_GATEWAY
        code = "OLLAMA_ERROR"
        message = "The local language model request failed."

    raise HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
    ) from error


async def get_answer_generation_service(
) -> AsyncIterator[AnswerGenerationService]:
    """Create an answer service using application settings."""

    settings = get_settings()

    service = AnswerGenerationService(
        llm_client=_create_ollama_client(),
        top_k=settings.retrieval_top_k,
        evidence_threshold=settings.evidence_threshold,
        max_context_chars=settings.max_context_chars,
    )

    try:
        yield service
    finally:
        await service.aclose()


async def get_structured_extraction_service(
) -> AsyncIterator[StructuredExtractionService]:
    """Create an extraction service using application settings."""

    settings = get_settings()

    service = StructuredExtractionService(
        llm_client=_create_ollama_client(),
        evidence_threshold=settings.evidence_threshold,
        max_context_chars=settings.max_context_chars,
        max_repair_attempts=settings.extraction_max_repair_attempts,
    )

    try:
        yield service
    finally:
        await service.aclose()


AnswerServiceDependency = Annotated[
    AnswerGenerationService,
    Depends(get_answer_generation_service),
]

ExtractionServiceDependency = Annotated[
    StructuredExtractionService,
    Depends(get_structured_extraction_service),
]


@router.post(
    "/ask",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about an uploaded PDF",
    description=(
        "Retrieves evidence from one uploaded PDF and generates an answer "
        "with verified citations, or returns insufficient_evidence."
    ),
)
async def ask_document(
    file: Annotated[
        UploadFile,
        File(description="PDF document to question"),
    ],
    question: Annotated[
        str,
        Form(
            min_length=1,
            max_length=QUESTION_MAX_LENGTH,
            description="Question to answer using the uploaded PDF",
        ),
    ],
    service: AnswerServiceDependency,
) -> AnswerResponse:
    """Ingest one PDF and answer a question using verified evidence."""

    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "EMPTY_QUESTION",
                "message": "The question must not be blank.",
            },
        )

    try:
        extraction = await _extract_uploaded_pdf(file)

    except (
        PDFTooLargeError,
        PDFPageLimitError,
        PDFExtractionError,
    ) as exc:
        _raise_pdf_http_error(exc)

    finally:
        await file.close()

    try:
        return await service.answer_pages(
            pages=extraction.pages,
            document_id=str(extraction.document_id),
            file_name=extraction.filename,
            question=question.strip(),
        )

    except OllamaClientError as exc:
        _raise_llm_http_error(exc)


@router.post(
    "/extract-structured",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract structured metadata from an uploaded PDF",
    description=(
        "Extracts TechnicalDocumentMetadata using a fixed Pydantic schema "
        "and verifies field-level citations against retrieved source text."
    ),
)
async def extract_structured_document(
    file: Annotated[
        UploadFile,
        File(description="PDF document for metadata extraction"),
    ],
    service: ExtractionServiceDependency,
    schema_name: Annotated[
        ExtractionSchemaName,
        Form(description="Supported extraction schema"),
    ] = ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA,
    top_k: Annotated[
        int | None,
        Form(
            ge=1,
            le=20,
            description="Maximum number of retrieved chunks",
        ),
    ] = None,
) -> ExtractionResponse:
    """Ingest one PDF and return verified structured metadata."""

    try:
        extraction = await _extract_uploaded_pdf(file)

    except (
        PDFTooLargeError,
        PDFPageLimitError,
        PDFExtractionError,
    ) as exc:
        _raise_pdf_http_error(exc)

    finally:
        await file.close()

    settings = get_settings()

    request = ExtractionRequest(
        schema_name=schema_name,
        top_k=(
            settings.retrieval_top_k
            if top_k is None
            else top_k
        ),
    )

    try:
        return await service.extract_pages(
            pages=extraction.pages,
            document_id=str(extraction.document_id),
            file_name=extraction.filename,
            request=request,
        )

    except StructuredExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "STRUCTURED_EXTRACTION_FAILED",
                "message": (
                    "The model could not produce valid, "
                    "evidence-backed metadata."
                ),
            },
        ) from exc

    except OllamaClientError as exc:
        _raise_llm_http_error(exc)
