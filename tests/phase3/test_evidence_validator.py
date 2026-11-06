"""Tests for deterministic citation and field-evidence provenance checks."""

import pytest
from pydantic import ValidationError

from app.schemas.answer import EvidenceQuoteDraft
from app.schemas.extraction import (
    ExtractionEvidenceDraft,
    MetadataField,
)
from app.schemas.retrieval import RetrievedChunk
from app.services.evidence_validator import (
    EvidenceValidationError,
    EvidenceValidator,
)


def make_chunk(
    *,
    chunk_id: str = "chunk-001",
    document_id: str = "doc-001",
    file_name: str = "technical-guide.pdf",
    page_number: int = 7,
    chunk_index: int = 0,
    text: str = "The technical guide describes the system.",
    score: float = 0.85,
) -> RetrievedChunk:
    """Build a retrieved chunk with offsets matching its source text."""

    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=file_name,
        page_number=page_number,
        chunk_index=chunk_index,
        start_char=0,
        end_char=len(text),
        text=text,
        score=score,
    )


def make_answer_evidence(
    *,
    chunk_id: str = "chunk-001",
    quote: str = "technical guide",
) -> EvidenceQuoteDraft:
    """Build an answer evidence draft."""

    return EvidenceQuoteDraft(chunk_id=chunk_id, quote=quote)


def make_extraction_evidence(
    *,
    field_name: MetadataField = MetadataField.DOCUMENT_TITLE,
    chunk_id: str = "chunk-001",
    quote: str = "Digital Identity Guidelines",
) -> ExtractionEvidenceDraft:
    """Build a field-specific extraction evidence draft."""

    return ExtractionEvidenceDraft(
        field_name=field_name,
        chunk_id=chunk_id,
        quote=quote,
    )


class TestEvidenceValidatorInitialization:
    def test_empty_retrieval_result_is_valid(self):
        validator = EvidenceValidator(())

        assert validator.build_references() == ()

    def test_duplicate_chunk_ids_are_rejected(self):
        first = make_chunk(chunk_id="duplicate")
        second = make_chunk(
            chunk_id="duplicate",
            document_id="doc-002",
            page_number=8,
            text="Different source text.",
        )

        with pytest.raises(
            EvidenceValidationError,
            match="duplicate chunk IDs",
        ):
            EvidenceValidator((first, second))


class TestQuoteVerification:
    def test_exact_quote_is_verified_against_retrieved_chunk(self):
        chunk = make_chunk(text="The technical guide describes the system.")
        validator = EvidenceValidator((chunk,))

        verified = validator.verify_quote(
            chunk_id="chunk-001",
            quote="technical guide",
        )

        assert verified.chunk == chunk
        assert verified.excerpt == "technical guide"

    @pytest.mark.parametrize(
        ("source", "quote", "expected_excerpt"),
        [
            (
                "The TECHNICAL GUIDE describes the system.",
                "technical guide",
                "TECHNICAL GUIDE",
            ),
            (
                "The technical\tguide\n describes the system.",
                "TECHNICAL guide",
                "technical\tguide",
            ),
            (
                "The oﬃcial technical guide.",
                "OFFICIAL TECHNICAL GUIDE",
                "oﬃcial technical guide",
            ),
            (
                "The Straße technical guide.",
                "STRASSE TECHNICAL GUIDE",
                "Straße technical guide",
            ),
            (
                "Cafe\u0301 technical guide.",
                "CAFÉ TECHNICAL GUIDE",
                "Cafe\u0301 technical guide",
            ),
        ],
    )
    def test_unicode_case_and_whitespace_normalization_preserves_source_excerpt(
        self,
        source,
        quote,
        expected_excerpt,
    ):
        validator = EvidenceValidator((make_chunk(text=source),))

        verified = validator.verify_quote(
            chunk_id="chunk-001",
            quote=quote,
        )

        assert verified.excerpt == expected_excerpt

    def test_repeated_quote_uses_first_source_occurrence(self):
        source = "Title: Digital Guide. Archive: Digital Guide."
        validator = EvidenceValidator((make_chunk(text=source),))

        verified = validator.verify_quote(
            chunk_id="chunk-001",
            quote="Digital Guide",
        )

        assert verified.excerpt == "Digital Guide"
        assert source.index(verified.excerpt) == source.index("Digital Guide")

    def test_unknown_chunk_id_is_rejected(self):
        validator = EvidenceValidator((make_chunk(),))

        with pytest.raises(
            EvidenceValidationError,
            match="outside this retrieval result",
        ):
            validator.verify_quote(
                chunk_id="not-retrieved",
                quote="technical guide",
            )

    def test_quote_absent_from_source_is_rejected(self):
        validator = EvidenceValidator((make_chunk(),))

        with pytest.raises(
            EvidenceValidationError,
            match="does not match the retrieved chunk",
        ):
            validator.verify_quote(
                chunk_id="chunk-001",
                quote="a fabricated source statement",
            )

    def test_quote_blank_after_normalization_is_rejected(self):
        validator = EvidenceValidator((make_chunk(),))

        with pytest.raises(
            EvidenceValidationError,
            match="blank after normalization",
        ):
            validator.verify_quote(
                chunk_id="chunk-001",
                quote=" \t\n ",
            )


class TestCitationConstruction:
    def test_citation_uses_retrieved_chunk_metadata(self):
        chunk = make_chunk(
            chunk_id="source-chunk-17",
            document_id="source-document-4",
            file_name="nist-guide.pdf",
            page_number=12,
            text="NIST SP 800-63-4 defines digital identity guidance.",
        )
        validator = EvidenceValidator((chunk,))

        citation = validator.build_citation(
            make_answer_evidence(
                chunk_id="source-chunk-17",
                quote="NIST SP 800-63-4",
            )
        )

        assert citation.citation_id == "source-chunk-17"
        assert citation.chunk_id == "source-chunk-17"
        assert citation.document_id == "source-document-4"
        assert citation.file_name == "nist-guide.pdf"
        assert citation.page_number == 12
        assert citation.excerpt == "NIST SP 800-63-4"

    def test_citation_preserves_original_source_formatting(self):
        source = "The OFFICIAL\tGUIDE was issued in 2025."
        validator = EvidenceValidator((make_chunk(text=source),))

        citation = validator.build_citation(
            make_answer_evidence(quote="official guide")
        )

        assert citation.excerpt == "OFFICIAL\tGUIDE"

    def test_citation_rejects_a_forged_chunk_id(self):
        validator = EvidenceValidator((make_chunk(),))

        with pytest.raises(EvidenceValidationError):
            validator.build_citation(
                make_answer_evidence(
                    chunk_id="forged-chunk",
                    quote="technical guide",
                )
            )

    def test_citation_rejects_a_quote_not_in_the_source(self):
        validator = EvidenceValidator((make_chunk(),))

        with pytest.raises(EvidenceValidationError):
            validator.build_citation(
                make_answer_evidence(quote="unsupported claim")
            )


class TestEvidenceReferences:
    def test_references_preserve_retrieval_metadata_and_order(self):
        first = make_chunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            file_name="first.pdf",
            page_number=3,
            score=0.91,
        )
        second = make_chunk(
            chunk_id="chunk-002",
            document_id="doc-002",
            file_name="second.pdf",
            page_number=9,
            chunk_index=1,
            text="The second technical source.",
            score=0.72,
        )
        validator = EvidenceValidator((first, second))

        references = validator.build_references()

        assert len(references) == 2
        assert references[0].chunk_id == "chunk-001"
        assert references[0].document_id == "doc-001"
        assert references[0].file_name == "first.pdf"
        assert references[0].page_number == 3
        assert references[0].score == 0.91
        assert references[1].chunk_id == "chunk-002"
        assert references[1].document_id == "doc-002"
        assert references[1].file_name == "second.pdf"
        assert references[1].page_number == 9
        assert references[1].score == 0.72

    def test_references_are_immutable_schema_objects(self):
        validator = EvidenceValidator((make_chunk(),))
        reference = validator.build_references()[0]

        with pytest.raises(ValidationError):
            reference.page_number = 99


class TestFieldCitationConstruction:
    def test_field_citation_requires_value_in_verified_source_quote(self):
        source = "Document title: Digital Identity Guidelines."
        validator = EvidenceValidator((make_chunk(text=source),))
        draft = make_extraction_evidence(
            quote="Digital Identity Guidelines"
        )

        result = validator.build_field_citation(
            draft,
            expected_value="Digital Identity Guidelines",
        )

        assert result.field_name == MetadataField.DOCUMENT_TITLE
        assert result.citation.chunk_id == "chunk-001"
        assert result.citation.document_id == "doc-001"
        assert result.citation.file_name == "technical-guide.pdf"
        assert result.citation.page_number == 7
        assert result.citation.excerpt == "Digital Identity Guidelines"

    def test_field_value_matching_uses_case_and_whitespace_normalization(self):
        source = "Document ID: NIST   SP 800-63-4."
        validator = EvidenceValidator((make_chunk(text=source),))
        draft = make_extraction_evidence(
            quote="NIST SP 800-63-4",
            field_name=MetadataField.DOCUMENT_IDENTIFIER,
        )

        result = validator.build_field_citation(
            draft,
            expected_value="nist   sp 800-63-4",
        )

        assert result.field_name == MetadataField.DOCUMENT_IDENTIFIER
        assert result.citation.excerpt == "NIST   SP 800-63-4"

    @pytest.mark.parametrize("expected_value", [None, "", "   ", "\t\n"])
    def test_null_or_blank_expected_value_is_rejected(self, expected_value):
        validator = EvidenceValidator((make_chunk(),))
        draft = make_extraction_evidence(quote="technical guide")

        with pytest.raises(
            EvidenceValidationError,
            match="non-blank extracted value",
        ):
            validator.build_field_citation(
                draft,
                expected_value=expected_value,
            )

    def test_value_absent_from_verified_quote_is_rejected(self):
        source = "Document title: Digital Identity Guidelines."
        validator = EvidenceValidator((make_chunk(text=source),))
        draft = make_extraction_evidence(
            quote="Digital Identity Guidelines"
        )

        with pytest.raises(
            EvidenceValidationError,
            match="not present in its supporting quote",
        ):
            validator.build_field_citation(
                draft,
                expected_value="Digital Identity 2030",
            )

    def test_value_cannot_be_borrowed_from_another_chunk(self):
        first = make_chunk(
            chunk_id="chunk-title",
            text="Document title: Digital Identity Guidelines.",
        )
        second = make_chunk(
            chunk_id="chunk-date",
            page_number=8,
            chunk_index=1,
            text="Publication date: July 2025.",
        )
        validator = EvidenceValidator((first, second))
        draft = make_extraction_evidence(
            chunk_id="chunk-title",
            quote="Document title: Digital Identity Guidelines.",
        )

        with pytest.raises(
            EvidenceValidationError,
            match="not present in its supporting quote",
        ):
            validator.build_field_citation(
                draft,
                expected_value="July 2025",
            )

    def test_field_evidence_cannot_reference_an_unretrieved_chunk(self):
        validator = EvidenceValidator((make_chunk(),))
        draft = make_extraction_evidence(
            chunk_id="unretrieved-chunk",
            quote="technical guide",
        )

        with pytest.raises(
            EvidenceValidationError,
            match="outside this retrieval result",
        ):
            validator.build_field_citation(
                draft,
                expected_value="technical guide",
            )


def test_curly_quote_normalization_keeps_the_original_source_excerpt():
    source = 'The “official” guide is current.'
    validator = EvidenceValidator((make_chunk(text=source),))

    verified = validator.verify_quote(
        chunk_id="chunk-001",
        quote='The "official" guide',
    )

    assert verified.excerpt == 'The “official” guide'


def test_metadata_title_match_preserves_original_punctuation():
    source = "Document title: NIST SP 800-63-4: Digital Identity Guidelines."
    validator = EvidenceValidator((make_chunk(text=source),))

    result = validator.build_field_citation_for_value(
        field_name=MetadataField.DOCUMENT_TITLE,
        expected_value="NIST SP 800-63-4: Digital Identity Guidelines",
    )

    assert result is not None
    assert "NIST SP 800-63-4: Digital Identity Guidelines" in result.citation.excerpt
    assert ":" in result.citation.excerpt


def test_superseded_document_date_is_not_cited_as_publication_date():
    source = (
        "NASA SP-2016-6105 Rev2 supersedes SP-2007-6105 Rev 1 "
        "dated December, 2007."
    )
    validator = EvidenceValidator((make_chunk(text=source),))

    result = validator.build_field_citation_for_value(
        field_name=MetadataField.PUBLICATION_DATE,
        expected_value="December, 2007",
    )

    assert result is None
