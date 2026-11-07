"""Tests for the production Q&A retrieval pipeline selection."""

from app.services.answer_generation import _default_pipeline_factory
from app.services.hybrid_retrieval import HybridRetrievalPipeline


def test_default_factory_uses_benchmarked_equal_weight_rrf() -> None:
    pipeline = _default_pipeline_factory()

    assert isinstance(pipeline, HybridRetrievalPipeline)
    assert pipeline.dense_candidate_k == 20
    assert pipeline.lexical_candidate_k == 20
    assert pipeline.rrf_k == 60
    assert pipeline.top_k == 5
