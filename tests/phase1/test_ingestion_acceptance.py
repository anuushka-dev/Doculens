from types import SimpleNamespace

import pymupdf
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import documents
from app.api.routes.documents import router
from app.services import pdf_ingestion


def make_pdf(*page_texts: str, encrypted: bool = False) -> bytes:
    """Create a deterministic PDF fixture entirely in memory."""
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


@pytest.fixture
def client() -> TestClient:
    """Create an isolated application for acceptance tests."""
    application = FastAPI()
    application.include_router(router)
    return TestClient(application)


def test_ac02_preserves_one_based_page_identity(
    client: TestClient,
) -> None:
    content = make_pdf(
        "First page contains enough useful source text.",
        "",
        "Third page contains different source evidence.",
    )

    response = client.post(
        "/documents/extract",
        files={"file": ("acceptance.pdf", content, "application/pdf")},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["page_count"] == 3
    assert [page["page_number"] for page in result["pages"]] == [1, 2, 3]
    assert "First page" in result["pages"][0]["text"]
    assert result["pages"][1]["text"] == ""
    assert "Third page" in result["pages"][2]["text"]
    assert result["empty_page_numbers"] == [2]
    assert result["near_empty_page_numbers"] == []
    assert len(result["sha256"]) == 64
    assert result["document_id"]


def test_ac01_rejects_unsupported_file_with_clear_message(
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


def test_ac01_rejects_upload_over_configured_byte_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        max_upload_bytes=32,
        max_pdf_pages=10,
    )
    monkeypatch.setattr(documents, "get_settings", lambda: settings)

    content = make_pdf("This PDF is larger than thirty-two bytes.")

    response = client.post(
        "/documents/extract",
        files={"file": ("large.pdf", content, "application/pdf")},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "pdf_too_large"
    assert detail["message"] == "The upload exceeds the 32-byte limit."


def test_ac01_rejects_pdf_over_page_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = SimpleNamespace(
        max_upload_bytes=2_000_000,
        max_pdf_pages=1,
    )
    monkeypatch.setattr(documents, "get_settings", lambda: settings)
    monkeypatch.setattr(pdf_ingestion, "get_settings", lambda: settings)

    content = make_pdf("Page one has enough text.", "Page two has enough text.")

    response = client.post(
        "/documents/extract",
        files={"file": ("two-pages.pdf", content, "application/pdf")},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert detail["code"] == "pdf_page_limit_exceeded"
    assert detail["message"] == (
        "The PDF contains 2 pages; the configured limit is 1."
    )


def test_ac01_rejects_password_protected_pdf(
    client: TestClient,
) -> None:
    content = make_pdf("Protected content is not available.", encrypted=True)

    response = client.post(
        "/documents/extract",
        files={"file": ("protected.pdf", content, "application/pdf")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "encrypted_pdf"
    assert detail["message"] == "Password-protected PDFs are not supported."


def test_ac01_rejects_document_without_extractable_text(
    client: TestClient,
) -> None:
    content = make_pdf("")

    response = client.post(
        "/documents/extract",
        files={"file": ("blank.pdf", content, "application/pdf")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "no_extractable_text"
    assert detail["message"] == (
        "No extractable text was found. This release supports "
        "text-selectable PDFs; OCR is not enabled."
    )


def test_ac01_rejects_malformed_pdf_without_exposing_parser_details(
    client: TestClient,
) -> None:
    content = b"%PDF-1.7\nnot a valid PDF structure\n%%EOF"

    response = client.post(
        "/documents/extract",
        files={"file": ("broken.pdf", content, "application/pdf")},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_pdf"
    assert detail["message"] == "The PDF could not be opened or parsed."
    assert "Traceback" not in detail["message"]


def test_ac01_rejects_empty_upload_with_stable_error(
    client: TestClient,
) -> None:
    response = client.post(
        "/documents/extract",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "empty_upload",
        "message": "The uploaded file is empty.",
    }


def test_ac01_rejects_invalid_filename_with_stable_error(
    client: TestClient,
) -> None:
    response = client.post(
        "/documents/extract",
        files={
            "file": (
                " ",
                make_pdf("Text for filename validation."),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "invalid_filename",
        "message": "A filename is required.",
    }