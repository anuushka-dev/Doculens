
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "doculens-api",
    }


def test_openapi_documents_extraction_endpoint() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/documents/extract" in paths
    assert "post" in paths["/documents/extract"]


def test_interactive_api_docs_are_available() -> None:
    response = client.get("/docs")

    assert response.status_code == 200
