
"""DocuLens FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes.documents import router as documents_router
from app.api.routes.qa import router as qa_router
from app.schemas.health import HealthResponse

app = FastAPI(
    title="DocuLens API",
    description=(
        "Evidence-first technical PDF ingestion and document intelligence."
    ),
    version="0.1.0",
)

# Phase 1: PDF ingestion and extraction.
app.include_router(documents_router)

# Phase 3: Grounded Q&A and structured extraction.
app.include_router(qa_router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Check API health",
)
def health_check() -> HealthResponse:
    """Report whether the API process is responding."""

    return HealthResponse(
        status="ok",
        service="doculens-api",
    )
