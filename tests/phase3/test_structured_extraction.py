"""Tests for evidence-grounded structured metadata extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import pytest
from pydantic import BaseModel

from app.adapters.llm.ollama_client import (
    OllamaOutputValidationError,
    OllamaTimeoutError,
)
from app.schemas.extraction import (
    ExtractionRequest,
    ExtractionSchemaName,
    ExtractionStatus,
    MetadataField,
    TechnicalDocumentMetadata,
)
from app.schemas.retrieval import RetrievalResponse, RetrievedChunk
from app.services.retrieval_pipeline import NoChunksToIndexError
from app.services.structured_extraction import (
    StructuredExtractionError,
    StructuredExtractionService,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

DOCUMENT_ID = "doc-001"
FILE_NAME = "nist_sp_800_63_4.pdf"
PAGES = [object()]
FIELD_VALUES = {
    "document_title": "Digital Identity Guidelines",
    "issuing_organization": "National Institute of Standards and Technology",
    "document_identifier": "NIST SP 800-63-4",
    "publication_date": "July 2025",
    "version": "4",
}
SOURCE_TEXT = (
    "Document title: Digital Identity Guidelines. "
    "Issuing organization: National Institute of Standards and Technology. "
    "Document identifier: NIST SP 800-63-4. "
    "Publication date: July 2025. Version: 4."
)


@pytest.fixture
def anyio_backend() -> str:
    """Run async service tests on asyncio, matching the service implementation."""

    return "asyncio"


def make_chunk(
    text: str = SOURCE_TEXT,
    *,
    chunk_id: str = "chunk-001",
    document_id: str = DOCUMENT_ID,
    file_name: str = FILE_NAME,
    page_number: int = 1,
    score: float = 0.85,
) -> RetrievedChunk:
    """Build a retrieval result with internally consistent source offsets."""

    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=file_name,
        page_number=page_number,
        chunk_index=0,
        start_char=0,
        end_char=len(text),
        text=text,
        score=score,
    )


def make_metadata(
    overrides: dict[str, str | None] | None = None,
) -> TechnicalDocumentMetadata:
    """Build fixed-schema metadata, optionally leaving fields unpopulated."""

    values = FIELD_VALUES.copy()
    if overrides is not None:
        values.update(overrides)
    return TechnicalDocumentMetadata(**values)


def make_draft(
    overrides: dict[str, str | None] | None = None,
) -> TechnicalDocumentMetadata:
    """Build candidate metadata for the fake structured generator."""

    return make_metadata(overrides)

class FakePipeline:
    """Synchronous retrieval fake used by asyncio.to_thread in the service."""

    def __init__(
        self,
        chunks: Sequence[RetrievedChunk] = (),
        *,
        index_error: Exception | None = None,
    ) -> None:
        self.chunks = tuple(chunks)
        self.index_error = index_error
        self.index_calls: list[tuple[Sequence[object], str, str]] = []
        self.search_calls: list[tuple[str, int | None]] = []

    def index_pages(
        self,
        pages: Sequence[object],
        *,
        document_id: str,
        file_name: str,
    ) -> tuple[RetrievedChunk, ...]:
        self.index_calls.append((pages, document_id, file_name))

        if self.index_error is not None:
            raise self.index_error

        return self.chunks

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> RetrievalResponse:
        self.search_calls.append((query, top_k))
        return RetrievalResponse(
            query=query,
            top_k=top_k or 5,
            results=self.chunks[:top_k],
        )


class FakeGenerator:
    """Async structured generator fake with deterministic outputs/errors."""

    def __init__(self, outputs: Sequence[BaseModel | Exception]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, type[BaseModel]]] = []
        self.closed = False

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_model: type[ModelT],
    ) -> ModelT:
        self.calls.append((prompt, response_model))
        if not self.outputs:
            raise AssertionError("generator was called more times than expected")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output

    async def aclose(self) -> None:
        self.closed = True


def make_service(
    generator: FakeGenerator,
    pipeline: FakePipeline,
    **overrides: object,
) -> StructuredExtractionService:
    """Construct the service with injected deterministic dependencies."""

    return StructuredExtractionService(
        llm_client=generator,
        pipeline_factory=lambda: pipeline,
        **overrides,
    )


async def run_extraction(
    service: StructuredExtractionService,
    *,
    request: ExtractionRequest | None = None,
) -> object:
    """Call the public service entry point with a single placeholder page."""

    return await service.extract_pages(
        pages=PAGES,
        document_id=DOCUMENT_ID,
        file_name=FILE_NAME,
        request=request or ExtractionRequest(top_k=3),
    )


class TestStructuredExtractionConfiguration:
    def test_default_configuration(self):
        generator = FakeGenerator([])
        pipeline = FakePipeline()
        service = make_service(generator, pipeline)

        assert service.evidence_threshold == 0.25
        assert service.max_context_chars == 12_000
        assert service.max_repair_attempts == 1

    @pytest.mark.parametrize("threshold", [-1.01, 1.01])
    def test_evidence_threshold_must_be_in_cosine_range(self, threshold):
        with pytest.raises(ValueError, match="evidence_threshold"):
            StructuredExtractionService(
                llm_client=FakeGenerator([]),
                pipeline_factory=FakePipeline,
                evidence_threshold=threshold,
            )

    @pytest.mark.parametrize("max_context_chars", [0, 499])
    def test_context_limit_must_be_at_least_five_hundred(
        self,
        max_context_chars,
    ):
        with pytest.raises(ValueError, match="max_context_chars"):
            StructuredExtractionService(
                llm_client=FakeGenerator([]),
                pipeline_factory=FakePipeline,
                max_context_chars=max_context_chars,
            )

    @pytest.mark.parametrize("attempts", [-1, 2, 3])
    def test_repair_attempts_are_limited_to_zero_or_one(self, attempts):
        with pytest.raises(ValueError, match="max_repair_attempts"):
            StructuredExtractionService(
                llm_client=FakeGenerator([]),
                pipeline_factory=FakePipeline,
                max_repair_attempts=attempts,
            )

    @pytest.mark.parametrize("attempts", [0, 1])
    def test_supported_repair_attempt_values_are_accepted(self, attempts):
        service = StructuredExtractionService(
            llm_client=FakeGenerator([]),
            pipeline_factory=FakePipeline,
            max_repair_attempts=attempts,
        )

        assert service.max_repair_attempts == attempts


class TestStructuredExtractionRetrieval:
    @pytest.mark.anyio
    async def test_indexes_pages_and_searches_for_metadata_with_requested_top_k(
        self,
    ):
        pipeline = FakePipeline([make_chunk()])
        generator = FakeGenerator([make_draft()])
        service = make_service(generator, pipeline)

        response = await run_extraction(
            service,
            request=ExtractionRequest(top_k=7),
        )

        assert pipeline.index_calls == [(PAGES, DOCUMENT_ID, FILE_NAME)]
        assert pipeline.search_calls == [
            (
                (
                    "document title issuing organization document identifier "
                    "publication date version metadata"
                ),
                7,
            )
        ]
        assert response.status == ExtractionStatus.EXTRACTED

    @pytest.mark.anyio
    async def test_empty_retrieval_returns_insufficient_evidence_without_generation(
        self,
    ):
        generator = FakeGenerator([])
        service = make_service(generator, FakePipeline())

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.INSUFFICIENT_EVIDENCE
        assert response.data == TechnicalDocumentMetadata()
        assert response.field_citations == ()
        assert generator.calls == []
        assert len(response.request_id) == 32

    @pytest.mark.anyio
    async def test_no_chunks_to_index_error_returns_empty_response(self):
        generator = FakeGenerator([])
        pipeline = FakePipeline(index_error=NoChunksToIndexError("no text chunks"))
        service = make_service(generator, pipeline)

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.INSUFFICIENT_EVIDENCE
        assert response.field_citations == ()
        assert pipeline.search_calls == []
        assert generator.calls == []

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("chunk_document_id", "chunk_file_name"),
        [
            ("another-document", FILE_NAME),
            (DOCUMENT_ID, "another-file.pdf"),
        ],
    )
    async def test_foreign_document_or_filename_fails_closed(
        self,
        chunk_document_id,
        chunk_file_name,
    ):
        generator = FakeGenerator([])
        pipeline = FakePipeline(
            [
                make_chunk(
                    document_id=chunk_document_id,
                    file_name=chunk_file_name,
                )
            ]
        )
        service = make_service(generator, pipeline)

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.INSUFFICIENT_EVIDENCE
        assert response.data == TechnicalDocumentMetadata()
        assert response.field_citations == ()
        assert generator.calls == []

    @pytest.mark.anyio
    async def test_score_below_threshold_abstains_without_generation(self):
        generator = FakeGenerator([])
        pipeline = FakePipeline(
            [make_chunk(score=0.249, page_number=4)]
        )
        service = make_service(
            generator,
            pipeline,
            evidence_threshold=0.25,
        )

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.INSUFFICIENT_EVIDENCE
        assert generator.calls == []


    @pytest.mark.anyio
    async def test_score_equal_to_threshold_is_eligible_for_extraction(self):
        generator = FakeGenerator([make_draft()])
        pipeline = FakePipeline(
            [make_chunk(score=0.25, page_number=4)]
        )
        service = make_service(
            generator,
            pipeline,
            evidence_threshold=0.25,
        )

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.EXTRACTED
        assert len(generator.calls) == 1


class TestStructuredExtractionPromptAndContext:
    def test_prompt_contains_fixed_schema_and_untrusted_source_warning(self):
        service = StructuredExtractionService(
            llm_client=FakeGenerator([]),
            pipeline_factory=FakePipeline,
        )
        prompt = service._build_prompt(
            schema_name=ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA,
            chunks=[make_chunk()],
        )

        assert "Requested schema: technical_document_metadata" in prompt
        assert "document_title" in prompt
        assert "issuing_organization" in prompt
        assert "document_identifier" in prompt
        assert "publication_date" in prompt
        assert "version" in prompt
        assert "untrusted source data" in prompt
        assert "chunk_id=chunk-001" in prompt
        assert "page=1" in prompt

    def test_context_is_bounded_and_truncates_long_passage(self):
        long_text = "A" * 2_000
        chunk = make_chunk(long_text)
        service = StructuredExtractionService(
            llm_client=FakeGenerator([]),
            pipeline_factory=FakePipeline,
            max_context_chars=500,
        )

        context = service._build_context([chunk])

        assert len(context) == 500
        assert context.startswith("[chunk_id=chunk-001 | page=1]")

    def test_context_stops_after_budget_is_exhausted(self):
        first = make_chunk("A" * 600, chunk_id="chunk-001")
        second = make_chunk("Second passage", chunk_id="chunk-002")
        service = StructuredExtractionService(
            llm_client=FakeGenerator([]),
            pipeline_factory=FakePipeline,
            max_context_chars=500,
        )

        context = service._build_context([first, second])

        assert len(context) == 500
        assert "chunk-002" not in context


class TestStructuredExtractionResults:
    @pytest.mark.anyio
    async def test_complete_metadata_returns_one_field_citation_per_field(self):
        generator = FakeGenerator([make_draft()])
        service = make_service(generator, FakePipeline([make_chunk()]))

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.EXTRACTED
        assert response.data == make_metadata()
        assert {item.field_name for item in response.field_citations} == set(
            MetadataField
        )
        assert len(response.field_citations) == 5
        assert all(
            item.citation.document_id == DOCUMENT_ID
            and item.citation.file_name == FILE_NAME
            and item.citation.page_number == 1
            for item in response.field_citations
        )

    @pytest.mark.anyio
    async def test_partial_metadata_returns_partial_status_and_matching_citation(
        self,
    ):
        partial_values = {
            "issuing_organization": None,
            "document_identifier": None,
            "publication_date": None,
            "version": None,
        }
        generator = FakeGenerator([make_draft(overrides=partial_values)])
        service = make_service(generator, FakePipeline([make_chunk()]))

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.PARTIAL
        assert response.data.document_title == FIELD_VALUES["document_title"]
        assert len(response.field_citations) == 1
        assert response.field_citations[0].field_name == MetadataField.DOCUMENT_TITLE

    @pytest.mark.anyio
    async def test_all_null_metadata_returns_insufficient_evidence_status(self):
        empty_values = {field: None for field in FIELD_VALUES}
        generator = FakeGenerator([make_draft(overrides=empty_values)])
        service = make_service(generator, FakePipeline([make_chunk()]))

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.INSUFFICIENT_EVIDENCE
        assert response.data == TechnicalDocumentMetadata()
        assert response.field_citations == ()

    @pytest.mark.anyio
    async def test_unsupported_schema_is_rejected_before_retrieval(self):
        pipeline = FakePipeline([make_chunk()])
        generator = FakeGenerator([])
        service = make_service(generator, pipeline)
        invalid_request = ExtractionRequest.model_construct(
            schema_name="unsupported_schema",
            top_k=5,
        )

        with pytest.raises(ValueError, match="Unsupported extraction schema"):
            await run_extraction(service, request=invalid_request)

        assert pipeline.index_calls == []
        assert generator.calls == []

class TestStructuredExtractionRepair:
    @pytest.mark.anyio
    async def test_metadata_only_output_builds_citations_from_source(self):
        metadata = make_draft()
        chunk = make_chunk()

        generator = FakeGenerator([metadata])
        service = make_service(
            generator,
            FakePipeline([chunk]),
        )

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.EXTRACTED
        assert len(generator.calls) == 1

        # The model receives a metadata-only response schema.
        assert generator.calls[0][1] is TechnicalDocumentMetadata

        # Citations are constructed from actual source chunks.
        assert len(response.field_citations) == 5

        for field_citation in response.field_citations:
            assert field_citation.citation.chunk_id == chunk.chunk_id
            assert field_citation.citation.document_id == DOCUMENT_ID
            assert field_citation.citation.page_number == chunk.page_number

    @pytest.mark.anyio
    async def test_unsupported_metadata_value_becomes_null_without_citation(self):
        metadata = make_draft(
            overrides={
                "document_title": "A fabricated document title",
            }
        )

        generator = FakeGenerator([metadata])
        service = make_service(
            generator,
            FakePipeline([make_chunk()]),
        )

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.PARTIAL
        assert response.data.document_title is None

        cited_fields = {
            item.field_name
            for item in response.field_citations
        }

        assert MetadataField.DOCUMENT_TITLE not in cited_fields
        assert len(response.field_citations) == 4

    @pytest.mark.anyio
    async def test_ollama_output_validation_error_is_repaired(self):
        generator = FakeGenerator(
            [
                OllamaOutputValidationError(
                    "invalid JSON schema output"
                ),
                make_draft(),
            ]
        )

        service = make_service(
            generator,
            FakePipeline([make_chunk()]),
        )

        response = await run_extraction(service)

        assert response.status == ExtractionStatus.EXTRACTED
        assert len(generator.calls) == 2

        assert all(
            response_model is TechnicalDocumentMetadata
            for _, response_model in generator.calls
        )

        assert (
            "invalid JSON schema output"
            in generator.calls[1][0]
        )

    @pytest.mark.anyio
    async def test_invalid_result_after_repair_raises_structured_extraction_error(
        self,
    ):
        generator = FakeGenerator(
            [
                OllamaOutputValidationError("invalid first response"),
                OllamaOutputValidationError("invalid second response"),
            ]
        )

        service = make_service(
            generator,
            FakePipeline([make_chunk()]),
        )

        with pytest.raises(StructuredExtractionError) as exc_info:
            await run_extraction(service)

        assert len(generator.calls) == 2
        assert isinstance(
            exc_info.value.__cause__,
            OllamaOutputValidationError,
        )

    @pytest.mark.anyio
    async def test_zero_repair_attempts_fails_after_first_invalid_result(self):
        generator = FakeGenerator(
            [
                OllamaOutputValidationError("invalid JSON schema output"),
                make_draft(),
            ]
        )

        service = make_service(
            generator,
            FakePipeline([make_chunk()]),
            max_repair_attempts=0,
        )

        with pytest.raises(StructuredExtractionError):
            await run_extraction(service)

        assert len(generator.calls) == 1
        assert len(generator.outputs) == 1

    @pytest.mark.anyio
    async def test_timeout_propagates_without_repair_retry(self):
        generator = FakeGenerator(
            [
                OllamaTimeoutError("Ollama generation timed out"),
                make_draft(),
            ]
        )

        service = make_service(
            generator,
            FakePipeline([make_chunk()]),
        )

        with pytest.raises(OllamaTimeoutError):
            await run_extraction(service)

        assert len(generator.calls) == 1
        assert len(generator.outputs) == 1


class TestStructuredExtractionLifecycle:
    @pytest.mark.anyio
    async def test_aclose_delegates_to_generator(self):
        generator = FakeGenerator([])
        service = make_service(generator, FakePipeline())

        await service.aclose()

        assert generator.closed is True
