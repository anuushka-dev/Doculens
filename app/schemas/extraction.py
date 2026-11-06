
"""Strict contracts for schema-driven document extraction."""

from __future__ import annotations

from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.answer import Citation


class ExtractionSchemaName(StrEnum):
    """Supported structured extraction schemas."""

    TECHNICAL_DOCUMENT_METADATA = "technical_document_metadata"


class ExtractionStatus(StrEnum):
    """Possible outcomes of a structured extraction request."""

    EXTRACTED = "extracted"
    PARTIAL = "partial"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class MetadataField(StrEnum):
    """Fields supported by TechnicalDocumentMetadata."""

    DOCUMENT_TITLE = "document_title"
    ISSUING_ORGANIZATION = "issuing_organization"
    DOCUMENT_IDENTIFIER = "document_identifier"
    PUBLICATION_DATE = "publication_date"
    VERSION = "version"


METADATA_FIELDS = tuple(MetadataField)


class ExtractionRequest(BaseModel):
    """Validated options for a structured extraction request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: ExtractionSchemaName = (
        ExtractionSchemaName.TECHNICAL_DOCUMENT_METADATA
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        strict=True,
        description="Maximum number of retrieved chunks supplied to extraction.",
    )


class TechnicalDocumentMetadata(BaseModel):
    """The first fixed extraction schema for technical documents."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_title: str | None = Field(
        default=None,
        max_length=500,
    )

    issuing_organization: str | None = Field(
        default=None,
        max_length=300,
    )

    document_identifier: str | None = Field(
        default=None,
        max_length=200,
    )

    publication_date: str | None = Field(
        default=None,
        max_length=100,
    )

    version: str | None = Field(
        default=None,
        max_length=100,
    )

    @model_validator(mode="after")
    def reject_blank_values(self) -> TechnicalDocumentMetadata:
        """Prevent empty strings from representing extracted values."""

        for field_name in METADATA_FIELDS:
            value = getattr(self, field_name.value)

            if value is not None and not value.strip():
                raise ValueError(
                    f"{field_name.value} must be non-blank or null"
                )

        return self


class ExtractionEvidenceDraft(BaseModel):
    """Untrusted model-provided evidence for one extracted field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: MetadataField

    chunk_id: str = Field(
        min_length=1,
        max_length=80,
    )

    quote: str = Field(
        min_length=1,
        max_length=4_000,
    )

    @field_validator("quote")
    @classmethod
    def quote_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence quote must not be blank")
        return value


class ExtractionDraft(BaseModel):
    """Untrusted model output before evidence verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metadata: TechnicalDocumentMetadata

    evidence: tuple[ExtractionEvidenceDraft, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )

    @model_validator(mode="after")
    def validate_field_evidence(self) -> ExtractionDraft:
        """Require exactly one evidence item for each non-null field."""

        evidence_fields = [
            item.field_name for item in self.evidence
        ]

        if len(evidence_fields) != len(set(evidence_fields)):
            raise ValueError(
                "each metadata field can have only one evidence item"
            )

        for field_name in METADATA_FIELDS:
            has_value = (
                getattr(self.metadata, field_name.value) is not None
            )

            has_evidence = field_name in evidence_fields

            if has_value and not has_evidence:
                raise ValueError(
                    f"{field_name.value} requires source evidence"
                )

            if has_evidence and not has_value:
                raise ValueError(
                    f"{field_name.value} has evidence but no extracted value"
                )

        return self


class ExtractionFieldCitation(BaseModel):
    """Verified citation associated with one extracted metadata field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    field_name: MetadataField
    citation: Citation


class ExtractionResponse(BaseModel):
    """Final structured extraction response returned by the API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: ExtractionSchemaName

    status: ExtractionStatus

    data: TechnicalDocumentMetadata

    field_citations: tuple[ExtractionFieldCitation, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )

    request_id: str | None = Field(
        default=None,
        max_length=64,
    )

    @model_validator(mode="after")
    def validate_extraction_result(self) -> ExtractionResponse:
        """Keep extracted values, citations, and status consistent."""

        cited_fields = [
            item.field_name for item in self.field_citations
        ]

        if len(cited_fields) != len(set(cited_fields)):
            raise ValueError(
                "each metadata field can have only one citation"
            )

        populated_fields = {
            field_name
            for field_name in METADATA_FIELDS
            if getattr(self.data, field_name.value) is not None
        }

        if set(cited_fields) != populated_fields:
            raise ValueError(
                "every non-null metadata field must have exactly one citation"
            )

        extracted_count = len(populated_fields)
        total_fields = len(METADATA_FIELDS)

        if extracted_count == 0:
            expected_status = ExtractionStatus.INSUFFICIENT_EVIDENCE
        elif extracted_count == total_fields:
            expected_status = ExtractionStatus.EXTRACTED
        else:
            expected_status = ExtractionStatus.PARTIAL

        if self.status != expected_status:
            raise ValueError(
                f"status must be '{expected_status.value}' "
                "for the supplied extraction data"
            )

        return self
