# DocuLens

DocuLens is an evidence-first technical document intelligence prototype. Its goal is to answer questions about PDF documents using retrieved source evidence, page-level citations, explicit abstention when evidence is insufficient, and schema-validated structured output.

**Current status:** Phase 0 feasibility work and the initial PDF ingestion API are implemented. Retrieval, grounded question answering, structured extraction, and the user interface are not yet complete.

## Current capabilities

- FastAPI application with a health endpoint.
- PDF upload and page-wise text extraction using PyMuPDF.
- Validation of upload size, page count, malformed PDFs, encrypted PDFs, and documents with no usable extracted text.
- Stable, one-based page numbering.
- SHA-256 document digest and generated document identifier.
- Automated tests and Ruff linting.

## Supported input and limitations

The initial target is one English, text-selectable PDF per session.

- Maximum upload size: 20 MiB.
- Maximum PDF length: 250 pages.
- Scanned/image-only PDFs requiring OCR are not supported.
- Reliable table structure extraction and general diagram understanding are not guaranteed.
- Retrieval, answer citations, abstention, and structured JSON generation remain planned capabilities until their implementation and tests are complete.

DocuLens is a prototype, not a production-ready service.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Git

Ollama and the `qwen3.5:2b` model are additionally required to run the local Phase 0 model benchmark.

## Setup

From the repository root:

```bash
uv sync --locked
```

## Run the API

```bash
uv run uvicorn app.main:app --reload
```

The local service is available at `http://127.0.0.1:8000`.

- Health check: `GET /health`
- PDF extraction: `POST /documents/extract`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Use the interactive API documentation to inspect the upload field, request format, response schema, and error responses.

## Tests and lint

```bash
uv run pytest -q
uv run ruff check .
```

## Phase 0 experiments

The project contains PDF extraction and quality-audit scripts under `scripts/phase0/`.

The local-model benchmark uses the NIST PDF fixtures in `sample_docs/`:

```bash
uv run python scripts/phase0/ollama_f5_benchmark.py
```

The benchmark report and rendered audit images are generated under `data/phase0/`. These generated artifacts are local and are intentionally excluded from Git.

Benchmark request success and latency do not, by themselves, establish answer correctness. Review each answer against its evidence before drawing quality conclusions.

## Project roadmap

1. Feasibility validation.
2. Ingestion and parsing.
3. Page-aware chunking and retrieval.
4. Grounded answers, citations, abstention, and structured output.
5. Reliability, safety, and guardrails.
6. Minimal demo, reproducible benchmarks, and deployment handoff.

The phase plan and boundary specification are maintained in `docs/`.