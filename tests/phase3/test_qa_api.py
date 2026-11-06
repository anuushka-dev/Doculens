"""API tests for grounded Q&A and structured document extraction routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from httpx import Response

from app.adapters.llm.ollama_client import (
    OllamaClientError,
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaOutputValidationError,
    OllamaProtocolError,
    OllamaTimeoutError,
)
from app.adapters.pdf.extractor import PDFExtractionError, PDFTooLargeError
from app.api.routes import qa
from app.main import app
from app.schemas.answer import (
    AnswerResponse,
    AnswerStatus,
    Citation,
)
from app.schemas.document import ExtractedPage, PDFExtractionResult
from app.schemas.extraction import (
    ExtractionFieldCitation,
    ExtractionResponse,
    ExtractionSchemaName,
    ExtractionStatus,
    MetadataField,
    TechnicalDocumentMetadata,
)
from app.services.structured_extraction import StructuredExtractionError

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")
DOCUMENT_ID_TEXT = str(DOCUMENT_ID)
FILE_NAME = "technical-guide.pdf"
PDF_BYTES = b"%PDF-1.7\nexample test bytes\n"


def make_pdf_result() -> PDFExtractionResult:
    """Return a valid Phase 1 result for one uploaded PDF."""

    return PDFExtractionResult(
        document_id=DOCUMENT_ID,
        filename=FILE_NAME,
        sha256="a" * 64,
        pages=(
            ExtractedPage(
                page_number=1,
                text="Technical Guide for Evidence-Based Systems",
            ),
        ),
    )


def make_answer_response() -> AnswerResponse:
    """Return a valid answered response with a verified citation."""

    citation = Citation(
        citation_id="chunk-001",
        chunk_id="chunk-001",
        document_id=DOCUMENT_ID_TEXT,
        file_name=FILE_NAME,
        page_number=1,
        excerpt="Technical Guide for Evidence-Based Systems",
    )
    return AnswerResponse(
        answer="The document is a technical guide.",
        status=AnswerStatus.ANSWERED,
        citations=(citation,),
    )


def make_extraction_response() -> ExtractionResponse:
    """Return a valid partial metadata response with a field citation."""

    citation = Citation(
        citation_id="chunk-001",
        chunk_id="chunk-001",
        document_id=DOCUMENT_ID_TEXT,
        file_name=FILE_NAME,
        page_number=1,
        excerpt="Technical Guide for Evidence-Based Systems",
    )
    return ExtractionResponse(
        schema_name=ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA,
        status=ExtractionStatus.PARTIAL,
        data=TechnicalDocumentMetadata(
            document_title="Technical Guide for Evidence-Based Systems"
        ),
        field_citations=(
            ExtractionFieldCitation(
                field_name=MetadataField.DOCUMENT_TITLE,
                citation=citation,
            ),
        ),
    )


class FakeAnswerService:
    """Capture route arguments and optionally return or raise a result."""

    def __init__(
        self,
        *,
        result: AnswerResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or make_answer_response()
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def answer_pages(self, **kwargs: object) -> AnswerResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeExtractionService:
    """Capture extraction inputs and optionally return or raise a result."""

    def __init__(
        self,
        *,
        result: ExtractionResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or make_extraction_response()
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def extract_pages(self, **kwargs: object) -> ExtractionResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeLLMClient:
    """Minimal Ollama-client stand-in for dependency lifecycle tests."""

    def __init__(self) -> None:
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


@pytest.fixture(autouse=True)
def default_qa_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep route tests independent from environment-specific settings."""

    settings = SimpleNamespace(
        max_upload_bytes=20 * 1024 * 1024,
        retrieval_top_k=5,
    )
    monkeypatch.setattr(qa, "get_settings", lambda: settings)


@pytest.fixture

def client() -> Iterator[TestClient]:
    """Provide a TestClient and restore application dependency overrides."""

    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides.clear()
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


def override_answer_service(service: FakeAnswerService) -> None:
    async def dependency() -> AsyncIterator[FakeAnswerService]:
        yield service

    app.dependency_overrides[qa.get_answer_generation_service] = dependency


def override_extraction_service(service: FakeExtractionService) -> None:
    async def dependency() -> AsyncIterator[FakeExtractionService]:
        yield service

    app.dependency_overrides[qa.get_structured_extraction_service] = dependency


def patch_pdf_extraction(monkeypatch: pytest.MonkeyPatch) -> list[UploadFile]:
    """Replace PDF ingestion and retain uploads so closure can be asserted."""

    uploads: list[UploadFile] = []

    async def fake_extract(upload: UploadFile) -> PDFExtractionResult:
        uploads.append(upload)
        return make_pdf_result()

    monkeypatch.setattr(qa, "_extract_uploaded_pdf", fake_extract)
    return uploads


def post_file(
    client: TestClient,
    path: str,
    *,
    data: dict[str, str] | None = None,
) -> Response:
    """Post a small multipart PDF to one document endpoint."""

    return client.post(
        path,
        files={"file": (FILE_NAME, PDF_BYTES, "application/pdf")},
        data=data or {},
    )


class TestAskDocumentRoute:
    def test_success_returns_answer_and_passes_ingested_document_to_service(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        uploads = patch_pdf_extraction(monkeypatch)
        service = FakeAnswerService()
        override_answer_service(service)

        response = post_file(
            client,
            "/documents/ask",
            data={"question": "  What is this document?  "},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "answered"
        assert response.json()["citations"][0]["page_number"] == 1
        assert len(service.calls) == 1
        call = service.calls[0]
        assert call["pages"] == make_pdf_result().pages
        assert call["document_id"] == DOCUMENT_ID_TEXT
        assert call["file_name"] == FILE_NAME
        assert call["question"] == "What is this document?"
        assert uploads[0].file.closed is True

    def test_blank_question_returns_stable_422_without_ingestion(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FakeAnswerService()
        override_answer_service(service)

        async def unexpected_ingestion(_upload: UploadFile) -> PDFExtractionResult:
            pytest.fail("PDF ingestion must not run for a blank question")

        monkeypatch.setattr(qa, "_extract_uploaded_pdf", unexpected_ingestion)
        response = post_file(
            client,
            "/documents/ask",
            data={"question": "   "},
        )

        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "EMPTY_QUESTION"
        assert service.calls == []

    def test_question_over_maximum_length_is_rejected_by_form_validation(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FakeAnswerService()
        override_answer_service(service)
        uploads = patch_pdf_extraction(monkeypatch)

        response = post_file(
            client,
            "/documents/ask",
            data={"question": "q" * (qa.QUESTION_MAX_LENGTH + 1)},
        )

        assert response.status_code == 422
        assert uploads == []
        assert service.calls == []

    @pytest.mark.parametrize(
        ("error_type", "status_code", "error_code"),
        [
            (OllamaTimeoutError, 504, "OLLAMA_TIMEOUT"),
            (OllamaConnectionError, 503, "OLLAMA_UNAVAILABLE"),
            (OllamaHTTPError, 502, "OLLAMA_HTTP_ERROR"),
            (OllamaProtocolError, 502, "OLLAMA_PROTOCOL_ERROR"),
            (OllamaOutputValidationError, 502, "OLLAMA_OUTPUT_INVALID"),
            (OllamaClientError, 502, "OLLAMA_ERROR"),
        ],
    )
    def test_ollama_errors_map_to_stable_http_responses(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        error_type: type[OllamaClientError],
        status_code: int,
        error_code: str,
    ) -> None:
        patch_pdf_extraction(monkeypatch)
        service = FakeAnswerService(error=error_type("private internal detail"))
        override_answer_service(service)

        response = post_file(
            client,
            "/documents/ask",
            data={"question": "What is this document?"},
        )

        assert response.status_code == status_code
        assert response.json()["detail"]["code"] == error_code
        assert "private internal detail" not in response.text


class TestStructuredExtractionRoute:
    def test_success_uses_overridden_top_k_and_returns_field_citation(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_pdf_extraction(monkeypatch)
        monkeypatch.setattr(
            qa,
            "get_settings",
            lambda: SimpleNamespace(retrieval_top_k=7),
        )
        service = FakeExtractionService()
        override_extraction_service(service)

        response = post_file(
            client,
            "/documents/extract-structured",
            data={
                "schema_name": "technical_document_metadata",
                "top_k": "12",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["schema_name"] == "technical_document_metadata"
        assert body["status"] == "partial"
        assert body["data"]["document_title"] == (
            "Technical Guide for Evidence-Based Systems"
        )
        assert body["field_citations"][0]["field_name"] == "document_title"

        assert len(service.calls) == 1
        call = service.calls[0]
        request = call["request"]
        assert call["pages"] == make_pdf_result().pages
        assert call["document_id"] == DOCUMENT_ID_TEXT
        assert call["file_name"] == FILE_NAME
        assert request.schema_name == ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
        assert request.top_k == 12

    def test_default_schema_and_configured_top_k_are_used_when_top_k_omitted(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_pdf_extraction(monkeypatch)
        monkeypatch.setattr(
            qa,
            "get_settings",
            lambda: SimpleNamespace(retrieval_top_k=9),
        )
        service = FakeExtractionService()
        override_extraction_service(service)

        response = post_file(client, "/documents/extract-structured")

        assert response.status_code == 200
        request = service.calls[0]["request"]
        assert request.schema_name == ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
        assert request.top_k == 9

    @pytest.mark.parametrize("top_k", ["0", "21"])
    def test_top_k_outside_supported_range_is_rejected(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        top_k: str,
    ) -> None:
        service = FakeExtractionService()
        override_extraction_service(service)
        uploads = patch_pdf_extraction(monkeypatch)

        response = post_file(
            client,
            "/documents/extract-structured",
            data={"top_k": top_k},
        )

        assert response.status_code == 422
        assert uploads == []
        assert service.calls == []

    def test_unsupported_schema_name_is_rejected_before_ingestion(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FakeExtractionService()
        override_extraction_service(service)
        uploads = patch_pdf_extraction(monkeypatch)

        response = post_file(
            client,
            "/documents/extract-structured",
            data={"schema_name": "custom_schema"},
        )

        assert response.status_code == 422
        assert uploads == []
        assert service.calls == []

    def test_structured_extraction_failure_maps_to_502_without_leaking_detail(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        patch_pdf_extraction(monkeypatch)
        service = FakeExtractionService(
            error=StructuredExtractionError("private repair trace")
        )
        override_extraction_service(service)

        response = post_file(client, "/documents/extract-structured")

        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "STRUCTURED_EXTRACTION_FAILED"
        assert "private repair trace" not in response.text

    @pytest.mark.parametrize(
        ("error_type", "status_code", "error_code"),
        [
            (OllamaTimeoutError, 504, "OLLAMA_TIMEOUT"),
            (OllamaConnectionError, 503, "OLLAMA_UNAVAILABLE"),
            (OllamaHTTPError, 502, "OLLAMA_HTTP_ERROR"),
            (OllamaProtocolError, 502, "OLLAMA_PROTOCOL_ERROR"),
            (OllamaOutputValidationError, 502, "OLLAMA_OUTPUT_INVALID"),
            (OllamaClientError, 502, "OLLAMA_ERROR"),
        ],
    )
    def test_ollama_errors_map_to_stable_http_responses(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        error_type: type[OllamaClientError],
        status_code: int,
        error_code: str,
    ) -> None:
        patch_pdf_extraction(monkeypatch)
        service = FakeExtractionService(error=error_type("private internal detail"))
        override_extraction_service(service)

        response = post_file(client, "/documents/extract-structured")

        assert response.status_code == status_code
        assert response.json()["detail"]["code"] == error_code
        assert "private internal detail" not in response.text


class TestPDFUploadHandling:
    @pytest.mark.parametrize(
        ("error_type", "status_code"),
        [
            (PDFTooLargeError, 413),
            (PDFExtractionError, 422),
        ],
    )
    def test_pdf_errors_map_to_stable_http_responses_and_close_upload(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        error_type: type[PDFExtractionError],
        status_code: int,
    ) -> None:
        service = FakeAnswerService()
        override_answer_service(service)
        uploads: list[UploadFile] = []

        async def failing_extract(upload: UploadFile) -> PDFExtractionResult:
            uploads.append(upload)
            raise error_type("test PDF failure")

        monkeypatch.setattr(qa, "_extract_uploaded_pdf", failing_extract)
        response = post_file(
            client,
            "/documents/ask",
            data={"question": "What is this document?"},
        )

        assert response.status_code == status_code
        assert response.json()["detail"]["message"] == "test PDF failure"
        assert "code" in response.json()["detail"]
        assert uploads[0].file.closed is True
        assert service.calls == []

    def test_upload_size_limit_is_enforced_before_pdf_ingestion(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = FakeAnswerService()
        override_answer_service(service)
        monkeypatch.setattr(
            qa,
            "get_settings",
            lambda: SimpleNamespace(max_upload_bytes=3),
        )
        monkeypatch.setattr(
            qa,
            "ingest_pdf_upload",
            lambda **_kwargs: pytest.fail("oversized upload must not be ingested"),
        )

        response = client.post(
            "/documents/ask",
            files={"file": (FILE_NAME, b"1234", "application/pdf")},
            data={"question": "What is this document?"},
        )

        assert response.status_code == 413
        assert "code" in response.json()["detail"]
        assert service.calls == []

    def test_upload_reader_accepts_payload_at_exact_limit(self) -> None:
        upload = UploadFile(filename=FILE_NAME, file=BytesIO(b"1234"))
        result = asyncio.run(qa._read_upload_limited(upload, 4))

        assert result == b"1234"
        asyncio.run(upload.close())

    def test_upload_reader_rejects_payload_over_limit(self) -> None:
        upload = UploadFile(filename=FILE_NAME, file=BytesIO(b"12345"))

        with pytest.raises(PDFTooLargeError):
            asyncio.run(qa._read_upload_limited(upload, 4))

        asyncio.run(upload.close())


class TestDependencyLifecycle:
    def test_answer_service_dependency_uses_settings_and_closes_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = SimpleNamespace(
            retrieval_top_k=11,
            evidence_threshold=0.31,
            max_context_chars=8_000,
        )
        llm_client = FakeLLMClient()
        monkeypatch.setattr(qa, "get_settings", lambda: settings)
        monkeypatch.setattr(qa, "_create_ollama_client", lambda: llm_client)

        async def exercise() -> None:
            dependency = qa.get_answer_generation_service()
            service = await anext(dependency)
            assert service.top_k == 11
            assert service.evidence_threshold == 0.31
            assert service.max_context_chars == 8_000
            await dependency.aclose()

        asyncio.run(exercise())
        assert llm_client.close_count == 1

    def test_extraction_service_dependency_uses_settings_and_closes_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = SimpleNamespace(
            evidence_threshold=0.37,
            max_context_chars=7_000,
            extraction_max_repair_attempts=0,
        )
        llm_client = FakeLLMClient()
        monkeypatch.setattr(qa, "get_settings", lambda: settings)
        monkeypatch.setattr(qa, "_create_ollama_client", lambda: llm_client)

        async def exercise() -> None:
            dependency = qa.get_structured_extraction_service()
            service = await anext(dependency)
            assert service.evidence_threshold == 0.37
            assert service.max_context_chars == 7_000
            assert service.max_repair_attempts == 0
            await dependency.aclose()

        asyncio.run(exercise())
        assert llm_client.close_count == 1


class TestApplicationRouteRegistration:
    def test_phase_one_and_phase_three_routes_remain_registered(self) -> None:
        paths = app.openapi()["paths"]

        assert "get" in paths["/health"]
        assert "post" in paths["/documents/extract"]
        assert "post" in paths["/documents/ask"]
        assert "post" in paths["/documents/extract-structured"]
