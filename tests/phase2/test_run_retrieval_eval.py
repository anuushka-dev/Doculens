"""Tests for the Phase 2 retrieval evaluation runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.phase2.run_retrieval_eval as runner
from scripts.phase2.validate_retrieval_dataset import DatasetError


@dataclass(frozen=True)
class Page:
    page_number: int
    text: str
    status: str = "ok"


@dataclass(frozen=True)
class Retrieved:
    document_id: str
    file_name: str
    page_number: int
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    score: float
    text: str


def make_rows(
    *,
    annotation_status: str = "labelled",
    page: int = 1,
) -> list[dict[str, Any]]:
    return [
        {
            "question_id": f"Q{index:03d}",
            "query": f"Which requirement is described in section {index}?",
            "document_path": "sample_docs/reference.pdf",
            "relevant_page_numbers": [page],
            "annotation_status": annotation_status,
            "annotation_note": "Verified against the rendered source PDF page.",
        }
        for index in range(1, 31)
    ]


def write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_resolve_pdf_accepts_repository_local_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    pdf = root / "sample_docs" / "source.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-test")

    monkeypatch.setattr(runner, "ROOT", root)

    assert runner.resolve_pdf("sample_docs/source.pdf") == pdf


def test_resolve_pdf_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "repo").resolve()
    root.mkdir()

    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-test")

    monkeypatch.setattr(runner, "ROOT", root)

    with pytest.raises(ValueError, match="inside the repository"):
        runner.resolve_pdf("../outside.pdf")


def test_resolve_pdf_rejects_non_pdf_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    source = root / "sample_docs" / "source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("not a PDF", encoding="utf-8")

    monkeypatch.setattr(runner, "ROOT", root)

    with pytest.raises(ValueError, match="not a PDF"):
        runner.resolve_pdf("sample_docs/source.txt")


def test_runner_refuses_pending_labels_before_extracting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    dataset = root / "tests" / "phase2" / "fixtures" / "retrieval_eval.jsonl"

    write_dataset(dataset, make_rows(annotation_status="pending"))
    monkeypatch.setattr(runner, "ROOT", root)

    def unexpected_extraction(**_kwargs: Any) -> None:
        pytest.fail("PDF extraction must not start with pending labels")

    with pytest.raises(DatasetError, match="not ready"):
        runner.run_benchmark(
            dataset_path=dataset,
            report_path=root / "data" / "report.json",
            extraction_fn=unexpected_extraction,
        )


def test_runner_rejects_expected_pages_outside_source_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    dataset = root / "tests" / "phase2" / "fixtures" / "retrieval_eval.jsonl"

    write_dataset(dataset, make_rows(page=2))

    pdf = root / "sample_docs" / "reference.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-test")

    monkeypatch.setattr(runner, "ROOT", root)

    def fake_extract(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            filename="reference.pdf",
            pages=(Page(1, "Source content"),),
            page_count=1,
            empty_page_numbers=(),
            near_empty_page_numbers=(),
        )

    def unexpected_pipeline(**_kwargs: Any) -> None:
        pytest.fail("The model/index must not load for invalid page labels")

    with pytest.raises(ValueError, match="outside the 1-page source PDF"):
        runner.run_benchmark(
            dataset_path=dataset,
            report_path=root / "data" / "report.json",
            extraction_fn=fake_extract,
            pipeline_factory=unexpected_pipeline,
        )


def test_runner_writes_recall_and_top_five_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    dataset = root / "tests" / "phase2" / "fixtures" / "retrieval_eval.jsonl"
    report_path = root / "data" / "phase2" / "report.json"

    pdf = root / "sample_docs" / "reference.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-test")

    write_dataset(dataset, make_rows())
    monkeypatch.setattr(runner, "ROOT", root)

    source_text = "Useful source content for evaluation."

    def fake_extract(*, pdf_bytes: bytes, filename: str) -> Any:
        assert pdf_bytes == b"%PDF-test"

        return SimpleNamespace(
            filename=filename,
            pages=(Page(1, source_text),),
            page_count=1,
            empty_page_numbers=(),
            near_empty_page_numbers=(),
        )

    def fake_embedder_factory(
        *,
        model_name: str,
        revision: str | None,
        device: str,
    ) -> Any:
        return SimpleNamespace(
            model_name=model_name,
            revision=revision,
            device=device,
            embedding_dimension=3,
        )

    class FakePipeline:
        def __init__(
            self,
            *,
            embedder: Any,
            chunk_size_words: int,
            overlap_words: int,
            top_k: int,
        ) -> None:
            self.embedder = embedder
            self.chunk_size_words = chunk_size_words
            self.overlap_words = overlap_words
            self.top_k = top_k
            self.document_ids: list[str] = []

        def index_pages(
            self,
            pages: tuple[Page, ...],
            *,
            document_id: str,
            file_name: str,
        ) -> tuple[object, ...]:
            assert pages[0].page_number == 1
            self.document_ids.append(document_id)
            return (object(),)

        def search(
            self,
            query: str,
            *,
            top_k: int | None = None,
        ) -> Any:
            assert query.startswith("Which requirement")

            return SimpleNamespace(
                results=(
                    Retrieved(
                        document_id=self.document_ids[0],
                        file_name="reference.pdf",
                        page_number=1,
                        chunk_id="reference-page-1-chunk-0",
                        chunk_index=0,
                        start_char=0,
                        end_char=len(source_text),
                        score=0.9,
                        text=source_text,
                    ),
                )
            )

        def save(self, _index_path: Path) -> None:
            return None

    report = runner.run_benchmark(
        dataset_path=dataset,
        report_path=report_path,
        extraction_fn=fake_extract,
        embedder_factory=fake_embedder_factory,
        pipeline_factory=FakePipeline,
    )

    assert report_path.is_file()
    assert report["metrics"]["query_count"] == 30
    assert report["metrics"]["hits"] == 30
    assert report["metrics"]["misses"] == 0
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["target_met"] is True
    assert report["index"]["chunk_count"] == 1
    assert report["queries"][0]["retrieved_evidence"][0]["text"] == source_text

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["dataset_sha256"] == report["dataset_sha256"]


def test_report_path_cannot_overwrite_dataset_or_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    dataset = root / "tests" / "phase2" / "fixtures" / "retrieval_eval.jsonl"

    pdf = root / "sample_docs" / "reference.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-test")

    write_dataset(dataset, make_rows())
    monkeypatch.setattr(runner, "ROOT", root)

    with pytest.raises(ValueError, match="must not overwrite"):
        runner.run_benchmark(
            dataset_path=dataset,
            report_path=dataset,
            extraction_fn=lambda **_kwargs: SimpleNamespace(
                filename="reference.pdf",
                pages=(Page(1, "Source content"),),
                page_count=1,
                empty_page_numbers=(),
                near_empty_page_numbers=(),
            ),
        )