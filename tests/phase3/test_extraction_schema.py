"""Tests for structured document extraction schemas."""

import pytest
from pydantic import ValidationError

from app.schemas.answer import Citation
from app.schemas.extraction import (
    ExtractionDraft,
    ExtractionEvidenceDraft,
    ExtractionFieldCitation,
    ExtractionRequest,
    ExtractionResponse,
    ExtractionSchemaName,
    ExtractionStatus,
    MetadataField,
    TechnicalDocumentMetadata,
)


def make_metadata(
    **overrides: str | None,
) -> TechnicalDocumentMetadata:
    """Build valid document metadata with optional field overrides."""

    values = {
        "document_title": "Digital Identity Guidelines",
        "issuing_organization": "National Institute of Standards and Technology",
        "document_identifier": "NIST SP 800-63-4",
        "publication_date": "July 2025",
        "version": "4",
    }
    values.update(overrides)

    return TechnicalDocumentMetadata(**values)


def make_evidence(
    field_name: MetadataField = MetadataField.DOCUMENT_TITLE,
    *,
    chunk_id: str = "chunk-001",
    quote: str = "Digital Identity Guidelines",
) -> ExtractionEvidenceDraft:
    """Build a field-specific evidence draft."""

    return ExtractionEvidenceDraft(
        field_name=field_name,
        chunk_id=chunk_id,
        quote=quote,
    )


def make_citation(
    *,
    citation_id: str = "citation-001",
    chunk_id: str = "chunk-001",
    page_number: int = 1,
) -> Citation:
    """Build an application-owned source citation."""

    return Citation(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id="doc-001",
        file_name="nist_sp_800_63_4.pdf",
        page_number=page_number,
        excerpt="Digital Identity Guidelines",
    )


def make_field_citation(
    field_name: MetadataField,
    *,
    citation_id: str,
    chunk_id: str,
    page_number: int = 1,
) -> ExtractionFieldCitation:
    """Build a citation associated with one metadata field."""

    return ExtractionFieldCitation(
        field_name=field_name,
        citation=make_citation(
            citation_id=citation_id,
            chunk_id=chunk_id,
            page_number=page_number,
        ),
    )


def make_response(
    metadata: TechnicalDocumentMetadata,
    status: ExtractionStatus,
) -> ExtractionResponse:
    """Build a response with one citation for each populated field."""

    citations = []

    for index, field_name in enumerate(MetadataField, start=1):
        if getattr(metadata, field_name.value) is not None:
            citations.append(
                make_field_citation(
                    field_name,
                    citation_id=f"citation-{index:03d}",
                    chunk_id=f"chunk-{index:03d}",
                    page_number=index,
                )
            )

    return ExtractionResponse(
        schema_name=ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA,
        status=status,
        data=metadata,
        field_citations=tuple(citations),
        request_id="request-001",
    )


class TestExtractionSchemaName:
    def test_supported_schema_name(self):
        assert (
            ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA.value
            == "technical_document_metadata"
        )

    def test_unknown_schema_name_is_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionRequest(
                schema_name="unknown_schema",
                top_k=5,
            )


class TestExtractionRequest:
    def test_valid_extraction_request(self):
        request = ExtractionRequest(
            schema_name=(
                ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
            ),
            top_k=5,
        )

        assert (
            request.schema_name
            == ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
        )
        assert request.top_k == 5

    def test_default_top_k(self):
        request = ExtractionRequest(
            schema_name=(
                ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
            )
        )

        assert request.top_k == 5

    @pytest.mark.parametrize("top_k", [0, -1, 21])
    def test_invalid_top_k_is_rejected(self, top_k):
        with pytest.raises(ValidationError):
            ExtractionRequest(
                schema_name=(
                    ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
                ),
                top_k=top_k,
            )

    @pytest.mark.parametrize("top_k", [True, 5.5])
    def test_non_integer_top_k_is_rejected(self, top_k):
        with pytest.raises(ValidationError):
            ExtractionRequest(
                schema_name=(
                    ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
                ),
                top_k=top_k,
            )

    def test_extra_request_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionRequest(
                schema_name=(
                    ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
                ),
                top_k=5,
                custom_schema={"title": "Anything"},
            )


class TestTechnicalDocumentMetadata:
    def test_all_five_metadata_fields(self):
        metadata = make_metadata()

        assert metadata.document_title == "Digital Identity Guidelines"
        assert (
            metadata.issuing_organization
            == "National Institute of Standards and Technology"
        )
        assert metadata.document_identifier == "NIST SP 800-63-4"
        assert metadata.publication_date == "July 2025"
        assert metadata.version == "4"

    def test_all_metadata_fields_can_be_null(self):
        metadata = TechnicalDocumentMetadata(
            document_title=None,
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        assert metadata.document_title is None
        assert metadata.issuing_organization is None
        assert metadata.document_identifier is None
        assert metadata.publication_date is None
        assert metadata.version is None

    def test_partial_metadata_is_valid(self):
        metadata = TechnicalDocumentMetadata(
            document_title="Digital Identity Guidelines",
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        assert metadata.document_title is not None
        assert metadata.document_identifier is None

    @pytest.mark.parametrize(
        "field_name",
        [
            "document_title",
            "issuing_organization",
            "document_identifier",
            "publication_date",
            "version",
        ],
    )
    def test_blank_metadata_values_are_rejected(self, field_name):
        with pytest.raises(ValidationError):
            make_metadata(**{field_name: "   "})

    def test_extra_metadata_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            TechnicalDocumentMetadata(
                document_title="Digital Identity Guidelines",
                issuing_organization="NIST",
                document_identifier="NIST SP 800-63-4",
                publication_date="July 2025",
                version="4",
                author="Unexpected field",
            )

    def test_non_string_metadata_values_are_rejected(self):
        with pytest.raises(ValidationError):
            make_metadata(document_title=123)


class TestExtractionEvidenceDraft:
    def test_valid_field_evidence(self):
        evidence = make_evidence()

        assert evidence.field_name == MetadataField.DOCUMENT_TITLE
        assert evidence.chunk_id == "chunk-001"
        assert evidence.quote == "Digital Identity Guidelines"

    def test_all_supported_metadata_fields_are_valid(self):
        for field_name in MetadataField:
            evidence = make_evidence(
                field_name=field_name,
                chunk_id=f"chunk-{field_name.value}",
            )

            assert evidence.field_name == field_name

    def test_unknown_metadata_field_is_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionEvidenceDraft(
                field_name="author",
                chunk_id="chunk-001",
                quote="An author name",
            )

    def test_blank_quote_is_rejected(self):
        with pytest.raises(ValidationError):
            make_evidence(quote="   ")

    def test_empty_chunk_id_is_rejected(self):
        with pytest.raises(ValidationError):
            make_evidence(chunk_id="")

    def test_extra_evidence_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionEvidenceDraft(
                field_name=MetadataField.DOCUMENT_TITLE,
                chunk_id="chunk-001",
                quote="Digital Identity Guidelines",
                page_number=1,
            )


class TestExtractionDraft:
    def test_valid_complete_extraction_draft(self):
        metadata = make_metadata()

        evidence = (
            make_evidence(
                MetadataField.DOCUMENT_TITLE,
                chunk_id="chunk-001",
                quote="Digital Identity Guidelines",
            ),
            make_evidence(
                MetadataField.ISSUING_ORGANIZATION,
                chunk_id="chunk-002",
                quote="National Institute of Standards and Technology",
            ),
            make_evidence(
                MetadataField.DOCUMENT_IDENTIFIER,
                chunk_id="chunk-003",
                quote="NIST SP 800-63-4",
            ),
            make_evidence(
                MetadataField.PUBLICATION_DATE,
                chunk_id="chunk-004",
                quote="July 2025",
            ),
            make_evidence(
                MetadataField.VERSION,
                chunk_id="chunk-005",
                quote="4",
            ),
        )

        draft = ExtractionDraft(
            metadata=metadata,
            evidence=evidence,
        )

        assert len(draft.evidence) == 5

    def test_partial_draft_requires_only_populated_field_evidence(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        draft = ExtractionDraft(
            metadata=metadata,
            evidence=(
                make_evidence(
                    MetadataField.DOCUMENT_TITLE,
                ),
            ),
        )

        assert len(draft.evidence) == 1

    def test_missing_evidence_for_populated_field_is_rejected(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        with pytest.raises(ValidationError):
            ExtractionDraft(
                metadata=metadata,
                evidence=(),
            )

    def test_evidence_for_null_field_is_rejected(self):
        metadata = make_metadata(
            document_title=None,
        )

        with pytest.raises(ValidationError):
            ExtractionDraft(
                metadata=metadata,
                evidence=(
                    make_evidence(
                        MetadataField.DOCUMENT_TITLE,
                    ),
                    make_evidence(
                        MetadataField.ISSUING_ORGANIZATION,
                        chunk_id="chunk-002",
                    ),
                    make_evidence(
                        MetadataField.DOCUMENT_IDENTIFIER,
                        chunk_id="chunk-003",
                    ),
                    make_evidence(
                        MetadataField.PUBLICATION_DATE,
                        chunk_id="chunk-004",
                    ),
                    make_evidence(
                        MetadataField.VERSION,
                        chunk_id="chunk-005",
                        quote="4",
                    ),
                ),
            )

    def test_duplicate_evidence_for_same_field_is_rejected(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        with pytest.raises(ValidationError):
            ExtractionDraft(
                metadata=metadata,
                evidence=(
                    make_evidence(
                        MetadataField.DOCUMENT_TITLE,
                        chunk_id="chunk-001",
                    ),
                    make_evidence(
                        MetadataField.DOCUMENT_TITLE,
                        chunk_id="chunk-002",
                    ),
                ),
            )

    def test_empty_metadata_and_empty_evidence_are_valid(self):
        metadata = make_metadata(
            document_title=None,
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        draft = ExtractionDraft(
            metadata=metadata,
            evidence=(),
        )

        assert not any(
            getattr(draft.metadata, field.value) is not None
            for field in MetadataField
        )
        assert draft.evidence == ()


class TestExtractionFieldCitation:
    def test_valid_field_citation(self):
        item = make_field_citation(
            MetadataField.DOCUMENT_TITLE,
            citation_id="citation-001",
            chunk_id="chunk-001",
        )

        assert item.field_name == MetadataField.DOCUMENT_TITLE
        assert item.citation.chunk_id == "chunk-001"
        assert item.citation.page_number == 1

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValidationError):
            ExtractionFieldCitation(
                field_name="author",
                citation=make_citation(),
            )


class TestExtractionStatus:
    def test_supported_status_values(self):
        assert ExtractionStatus.EXTRACTED.value == "extracted"
        assert ExtractionStatus.PARTIAL.value == "partial"
        assert (
            ExtractionStatus.INSUFFICIENT_EVIDENCE.value
            == "insufficient_evidence"
        )


class TestExtractionResponse:
    def test_complete_extraction_requires_all_field_citations(self):
        response = make_response(
            make_metadata(),
            ExtractionStatus.EXTRACTED,
        )

        assert response.status == ExtractionStatus.EXTRACTED
        assert len(response.field_citations) == 5

    def test_partial_extraction_requires_populated_field_citation(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        response = make_response(
            metadata,
            ExtractionStatus.PARTIAL,
        )

        assert response.status == ExtractionStatus.PARTIAL
        assert len(response.field_citations) == 1

    def test_empty_extraction_has_no_citations(self):
        metadata = make_metadata(
            document_title=None,
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        response = make_response(
            metadata,
            ExtractionStatus.INSUFFICIENT_EVIDENCE,
        )

        assert (
            response.status
            == ExtractionStatus.INSUFFICIENT_EVIDENCE
        )
        assert response.field_citations == ()

    def test_missing_citation_for_populated_field_is_rejected(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        with pytest.raises(ValidationError):
            ExtractionResponse(
                schema_name=(
                    ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
                ),
                status=ExtractionStatus.PARTIAL,
                data=metadata,
                field_citations=(),
            )

    def test_citation_for_null_field_is_rejected(self):
        metadata = make_metadata(
            document_title=None,
        )

        citations = (
            make_field_citation(
                MetadataField.DOCUMENT_TITLE,
                citation_id="citation-001",
                chunk_id="chunk-001",
            ),
            make_field_citation(
                MetadataField.ISSUING_ORGANIZATION,
                citation_id="citation-002",
                chunk_id="chunk-002",
            ),
            make_field_citation(
                MetadataField.DOCUMENT_IDENTIFIER,
                citation_id="citation-003",
                chunk_id="chunk-003",
            ),
            make_field_citation(
                MetadataField.PUBLICATION_DATE,
                citation_id="citation-004",
                chunk_id="chunk-004",
            ),
            make_field_citation(
                MetadataField.VERSION,
                citation_id="citation-005",
                chunk_id="chunk-005",
            ),
        )

        with pytest.raises(ValidationError):
            ExtractionResponse(
                schema_name=(
                    ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
                ),
                status=ExtractionStatus.PARTIAL,
                data=metadata,
                field_citations=citations,
            )

    def test_partial_metadata_cannot_use_extracted_status(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        response = make_response(
            metadata,
            ExtractionStatus.PARTIAL,
        )
        payload = response.model_dump()
        payload["status"] = ExtractionStatus.EXTRACTED

        with pytest.raises(ValidationError):
            ExtractionResponse.model_validate(payload)

    def test_complete_metadata_cannot_use_partial_status(self):
        response = make_response(
            make_metadata(),
            ExtractionStatus.EXTRACTED,
        )
        payload = response.model_dump()
        payload["status"] = ExtractionStatus.PARTIAL

        with pytest.raises(ValidationError):
            ExtractionResponse.model_validate(payload)

    def test_empty_metadata_cannot_use_partial_status(self):
        metadata = make_metadata(
            document_title=None,
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        response = make_response(
            metadata,
            ExtractionStatus.INSUFFICIENT_EVIDENCE,
        )
        payload = response.model_dump()
        payload["status"] = ExtractionStatus.PARTIAL

        with pytest.raises(ValidationError):
            ExtractionResponse.model_validate(payload)

    def test_non_empty_metadata_cannot_use_insufficient_status(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        response = make_response(
            metadata,
            ExtractionStatus.PARTIAL,
        )
        payload = response.model_dump()
        payload["status"] = ExtractionStatus.INSUFFICIENT_EVIDENCE

        with pytest.raises(ValidationError):
            ExtractionResponse.model_validate(payload)

    def test_extra_response_fields_are_rejected(self):
        metadata = make_metadata(
            issuing_organization=None,
            document_identifier=None,
            publication_date=None,
            version=None,
        )

        response = make_response(
            metadata,
            ExtractionStatus.PARTIAL,
        )

        payload = response.model_dump()
        payload["internal_debug_data"] = "not allowed"

        with pytest.raises(ValidationError):
            ExtractionResponse.model_validate(payload)
