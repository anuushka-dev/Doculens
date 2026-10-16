# DocuLens

DocuLens is an evidence-first assistant for answering questions about
user-supplied technical PDF documents.

Its core design goal is to make answers verifiable by connecting them
to actual document pages and source excerpts.

## Product goals

The planned minimum viable product (MVP) will:

- Accept English-language, text-selectable technical PDFs.
- Extract text page by page and preserve page numbers.
- Retrieve relevant passages for a user's question.
- Generate answers grounded in retrieved document evidence.
- Show source page references and supporting excerpts.
- Abstain when the document does not provide sufficient evidence.
- Extract a limited set of fields into schema-validated JSON.
- Measure retrieval quality, citation correctness, abstention,
  structured-output validity, and latency.

These are product goals, not claims that every feature is implemented.

## Current implementation status

The project currently has:

- A Python 3.12 development environment managed by uv.
- A configured Python package rooted at `app/`.
- Core backend dependencies installed and synchronized.
- Docker Desktop available from the Ubuntu WSL2 environment.
- Initial application package directories.

The PDF ingestion, vector retrieval, grounded Q&A, citation
validation, and structured extraction workflows still need to be
implemented and tested.

## Planned architecture

- **API:** FastAPI.
- **PDF extraction:** PyMuPDF.
- **Configuration and schemas:** Pydantic.
- **Embeddings:** A locally runnable embedding model, subject to
  feasibility testing.
- **Vector storage:** ChromaDB, subject to feasibility testing.
- **Answer generation:** An LLM provider behind a replaceable adapter.
- **Testing:** pytest and HTTPX.
- **Interface:** A minimal Streamlit interface for the first demo.

Technology choices that have not passed feasibility checks remain
provisional.

## Initial scope boundaries

The MVP focuses on a small collection of ordinary technical PDFs.

Scanned-document OCR, advanced figure and diagram reasoning,
two-document comparison, and agentic multi-step workflows are
deferred until the core retrieval and evidence checks pass.

## Engineering principles

1. Source citations must come from actual retrieved document metadata.
2. Missing evidence must not be replaced with a guessed answer.
3. Uploaded PDF text is untrusted input.
4. Secrets must not be committed to Git.
5. Features must pass tests and documented acceptance checks before
   being described as working.

## Development status

This is an actively developed prototype. Production readiness,
benchmark performance, and deployment reliability have not yet
been established.
