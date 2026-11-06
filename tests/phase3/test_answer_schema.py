"""Tests for strict Phase 3 answer and citation response contracts."""

import pytest
from pydantic import ValidationError

from app.schemas.answer import (
    AnswerDraft,
    AnswerResponse,
    AnswerStatus,
    Citation,
    EvidenceQuoteDraft,
    EvidenceReference,
)


def make_citation(**overrides):
    values = {
        "citation_id": "chunk-001",
        "chunk_id": "chunk-001",
        "document_id": "doc-001",
        "file_name": "guide.pdf",
        "page_number": 2,
        "excerpt": "The source excerpt.",
    }
    values.update(overrides)
    return Citation(**values)


def make_reference(**overrides):
    values = {
        "chunk_id": "chunk-001",
        "document_id": "doc-001",
        "file_name": "guide.pdf",
        "page_number": 2,
        "score": 0.83,
    }
    values.update(overrides)
    return EvidenceReference(**values)


class TestAnswerStatus:
    def test_supported_status_values(self):
        assert AnswerStatus.ANSWERED.value == "answered"
        assert AnswerStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"

    def test_unknown_status_is_rejected(self):
        with pytest.raises(ValidationError):
            AnswerResponse(answer="No evidence.", status="unknown")


class TestEvidenceQuoteDraft:
    def test_valid_evidence_quote(self):
        item = EvidenceQuoteDraft(chunk_id="chunk-001", quote="Exact source text")
        assert item.chunk_id == "chunk-001"
        assert item.quote == "Exact source text"

    @pytest.mark.parametrize("quote", ["", " ", "\t\n"])
    def test_blank_quote_is_rejected(self, quote):
        with pytest.raises(ValidationError):
            EvidenceQuoteDraft(chunk_id="chunk-001", quote=quote)

    def test_empty_chunk_id_is_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceQuoteDraft(chunk_id="", quote="Source")

    def test_extra_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            EvidenceQuoteDraft(
                chunk_id="chunk-001",
                quote="Source",
                page_number=3,
            )


class TestAnswerDraft:
    def test_valid_answer_draft(self):
        draft = AnswerDraft(
            answer="The answer is supported.",
            evidence=(EvidenceQuoteDraft(chunk_id="chunk-001", quote="Support"),),
        )
        assert draft.abstain is False
        assert len(draft.evidence) == 1

    def test_default_abstain_is_false(self):
        draft = AnswerDraft(answer="Supported answer")
        assert draft.abstain is False

    @pytest.mark.parametrize("answer", ["", " ", "\n\t"])
    def test_blank_answer_is_rejected(self, answer):
        with pytest.raises(ValidationError):
            AnswerDraft(answer=answer)

    def test_abstention_with_evidence_is_rejected(self):
        with pytest.raises(ValidationError, match="must not cite evidence"):
            AnswerDraft(
                answer="Insufficient evidence.",
                abstain=True,
                evidence=(EvidenceQuoteDraft(chunk_id="c1", quote="text"),),
            )

    def test_abstention_without_evidence_is_valid(self):
        draft = AnswerDraft(answer="Insufficient evidence.", abstain=True)
        assert draft.evidence == ()

    def test_too_many_evidence_quotes_are_rejected(self):
        evidence = tuple(
            EvidenceQuoteDraft(chunk_id=f"c{i}", quote=f"quote {i}")
            for i in range(9)
        )
        with pytest.raises(ValidationError):
            AnswerDraft(answer="Answer", evidence=evidence)


class TestEvidenceReference:
    def test_valid_reference(self):
        reference = make_reference()
        assert reference.score == pytest.approx(0.83)

    @pytest.mark.parametrize("page", [0, -1])
    def test_page_number_must_be_positive(self, page):
        with pytest.raises(ValidationError):
            make_reference(page_number=page)

    def test_page_number_must_be_an_integer(self):
        with pytest.raises(ValidationError):
            make_reference(page_number=2.5)

    @pytest.mark.parametrize("score", [-1.1, 1.1, float("inf"), -float("inf"), float("nan")])
    def test_invalid_similarity_score_is_rejected(self, score):
        with pytest.raises(ValidationError):
            make_reference(score=score)

    def test_cosine_similarity_score_is_preserved(self):
        reference = make_reference(score=0.251234)
        assert reference.score == pytest.approx(0.251234)


class TestCitation:
    def test_valid_citation(self):
        citation = make_citation()
        assert citation.page_number == 2
        assert citation.excerpt == "The source excerpt."

    @pytest.mark.parametrize("excerpt", ["", " ", "\n\t"])
    def test_blank_excerpt_is_rejected(self, excerpt):
        with pytest.raises(ValidationError):
            make_citation(excerpt=excerpt)

    @pytest.mark.parametrize("page", [0, -1])
    def test_page_number_must_be_positive(self, page):
        with pytest.raises(ValidationError):
            make_citation(page_number=page)

    def test_page_number_must_be_strict_integer(self):
        with pytest.raises(ValidationError):
            make_citation(page_number="2")

    def test_extra_fields_are_rejected(self):
        with pytest.raises(ValidationError):
            make_citation(source_url="https://example.invalid")


class TestAnswerResponse:
    def test_answered_response_requires_verified_citation(self):
        with pytest.raises(ValidationError, match="require at least one"):
            AnswerResponse(answer="An answer", status=AnswerStatus.ANSWERED)

    def test_answered_response_with_citation_is_valid(self):
        response = AnswerResponse(
            answer="The document says this.",
            status=AnswerStatus.ANSWERED,
            citations=(make_citation(),),
            evidence_references=(make_reference(),),
        )
        assert response.citations[0].page_number == 2

    def test_non_answered_response_must_not_contain_citations(self):
        with pytest.raises(ValidationError, match="must not contain citations"):
            AnswerResponse(
                answer="Cannot verify.",
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                citations=(make_citation(),),
            )

    def test_citation_ids_must_be_unique(self):
        duplicate = make_citation()
        with pytest.raises(ValidationError, match="must be unique"):
            AnswerResponse(
                answer="Supported.",
                status=AnswerStatus.ANSWERED,
                citations=(duplicate, duplicate),
            )

    def test_response_is_immutable(self):
        response = AnswerResponse(
            answer="Not established.",
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        )
        with pytest.raises(ValidationError):
            response.answer = "Changed"
