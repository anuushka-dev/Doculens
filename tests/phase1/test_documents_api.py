from types import SimpleNamespace

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import documents
from app.api.routes.documents import router
from app.services import pdf_ingestion


def make_pdf(*page_texts: str) -> bytes:
    """Generate a small PDF fixture in memory."""
    document = pymupdf.open()

    try:
        for text in page_texts:
            page = document.new_page()

            if text:
                page.insert_text((72, 72), text)

        return document.tobytes()
    finally:
        document.close()


@pytest.fixture
def client() -> TestClient:
    """Create an isolated application for route tests."""
    application = FastAPI()
    application.include_router(router)

    return TestClient(application)


def test_extract_endpoint_returns_page_aware_result(
    client: TestClient,
) -> None:
    content = make_pdf(
        "DocuLens extracts source evidence.",
        "Page two contains more evidence.",
    )

    response = client.post(
        "/documents/extract",
        files={"file": ("example.pdf", content, "application/pdf")},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["filename"] == "example.pdf"
    assert result["page_count"] == 2
    assert [page["page_number"] for page in result["pages"]] == [1, 2]
    assert "source evidence" in result["pages"][0]["text"]
    assert len(result["sha256"]) == 64
    assert result["document_id"]


def test_extract_endpoint_rejects_unsupported_extension(
    client: TestClient,
) -> None:
    response = client.post(
        "/documents/extract",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 422

    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_file_type"
    assert detail["message"] == "Only PDF files are supported."


def test_extract_endpoint_rejects_malformed_pdf(
    client: TestClient,
) -> None:
    content = b"%PDF-1.7\nnot a valid PDF structure\n%%EOF"

    response = client.post(
        "/documents/extract",
        files={"file": ("broken.pdf", content, "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_pdf"


def test_extract_endpoint_rejects_document_without_text(
    client: TestClient,
) -> None:
    content = make_pdf("")

    response = client.post(
        "/documents/extract",
        files={"file": ("blank.pdf", content, "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "no_extractable_text"


def test_extract_endpoint_rejects_upload_over_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        max_upload_bytes=32,
        max_pdf_pages=10,
    )
    monkeypatch.setattr(documents, "get_settings", lambda: settings)

    content = make_pdf("This PDF exceeds the configured upload limit.")

    response = client.post(
        "/documents/extract",
        files={"file": ("large.pdf", content, "application/pdf")},
    )

    assert response.status_code == 413

    detail = response.json()["detail"]
    assert detail["code"] == "pdf_too_large"
    assert detail["message"] == "The upload exceeds the 32-byte limit."


def test_extract_endpoint_rejects_page_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        max_upload_bytes=2_000_000,
        max_pdf_pages=1,
    )
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    monkeypatch.setattr(pdf_ingestion, "get_settings", lambda: settings)

    content = make_pdf("Page one.", "Page two.")

    response = client.post(
        "/documents/extract",
        files={"file": ("multiple-pages.pdf", content, "application/pdf")},
    )

    assert response.status_code == 413

    detail = response.json()["detail"]
    assert detail["code"] == "pdf_page_limit_exceeded"
    assert detail["message"] == (
        "The PDF contains 2 pages; the configured limit is 1."
    )
