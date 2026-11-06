"""Tests for the Phase 3 grounded-Q&A evaluation runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.schemas.answer import AnswerResponse, AnswerStatus, Citation
from scripts.phase3 import run_qa_eval as qa_eval

DOCUMENT_ID = UUID("12345678-1234-5678-1234-567812345678")


def make_example(
    *,
    case_id: str = "Q001",
    question: str = "What is the document title?",
    answerable: bool = True,
    expected_pages: tuple[int, ...] = (1,),
    review_status: str = "reviewed",
    review_note: str = "Checked against the source PDF.",
    pdf_path: str | None = "source.pdf",
    line_number: int = 1,
) -> qa_eval.QAExample:
    """Create a valid evaluation example."""

    return qa_eval.QAExample(
        case_id=case_id,
        question=question,
        answerable=answerable,
        expected_pages=expected_pages,
        review_status=review_status,
        review_note=review_note,
        pdf_path=pdf_path,
        line_number=line_number,
    )


def make_settings() -> SimpleNamespace:
    """Return the settings consumed by run_evaluation."""

    return SimpleNamespace(
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3.5:2b",
        ollama_timeout_seconds=120.0,
        ollama_num_ctx=8192,
        ollama_num_predict=1024,
        ollama_temperature=0.0,
        ollama_seed=42,
        retrieval_top_k=5,
        evidence_threshold=0.25,
        max_context_chars=12_000,
    )


def make_answered_response(page_number: int = 1) -> AnswerResponse:
    """Return a valid answered response with one citation."""

    return AnswerResponse(
        answer="The document title is Systems Guide.",
        status=AnswerStatus.ANSWERED,
        citations=(
            Citation(
                citation_id="chunk-001",
                chunk_id="chunk-001",
                document_id="doc-001",
                file_name="source.pdf",
                page_number=page_number,
                excerpt="Systems Guide",
            ),
        ),
        request_id="request-001",
    )


def make_abstained_response() -> AnswerResponse:
    """Return a valid insufficient-evidence response."""

    return AnswerResponse(
        answer="The uploaded document does not establish this fact.",
        status=AnswerStatus.INSUFFICIENT_EVIDENCE,
        request_id="request-002",
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> bytes:
    """Write JSONL rows and return the exact bytes written."""

    raw = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(raw)
    return raw


class TestAnswerabilityParsing:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            ("answerable", True),
            (" TRUE ", True),
            ("yes", True),
            ("1", True),
            ("unanswerable", False),
            ("FALSE", False),
            ("no", False),
            ("0", False),
        ],
    )
    def test_supported_answerability_labels(self, value, expected):
        assert qa_eval._parse_answerability({"answerable": value}, 3) is expected

    @pytest.mark.parametrize("value", [None, 1, 0, [], "maybe", ""])
    def test_invalid_answerability_labels_are_rejected(self, value):
        with pytest.raises(ValueError, match="Line 4"):
            qa_eval._parse_answerability({"answerable": value}, 4)


class TestExpectedPageParsing:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (3, (3,)),
            ([3, 1, 3], (1, 3)),
            ("3, 1,3", (1, 3)),
            ("", ()),
            (None, ()),
            ([], ()),
        ],
    )
    def test_supported_page_formats_are_normalized(self, value, expected):
        assert qa_eval._parse_expected_pages({"expected_pages": value}, 2) == expected

    @pytest.mark.parametrize("value", [True, 1.5, {}, "one,2"])
    def test_invalid_page_container_or_value_is_rejected(self, value):
        with pytest.raises((TypeError, ValueError)):
            qa_eval._parse_expected_pages({"expected_pages": value}, 7)

    @pytest.mark.parametrize("value", [[0], [-1], [True], [1.5]])
    def test_page_numbers_must_be_positive_integers(self, value):
        with pytest.raises(ValueError, match="positive integers"):
            qa_eval._parse_expected_pages({"expected_pages": value}, 8)


class TestExampleParsing:
    def test_parses_aliases_and_trims_text(self):
        example = qa_eval._parse_example(
            {
                "question_id": "  Q-9  ",
                "question": "  What is the version?  ",
                "answerability": "answerable",
                "gold_pages": [2, 1, 2],
                "review_status": " REVIEWED ",
                "reviewer_note": "  Checked on page 1.  ",
                "document_path": "docs/source.pdf",
            },
            9,
        )

        assert example.case_id == "Q-9"
        assert example.question == "What is the version?"
        assert example.answerable is True
        assert example.expected_pages == (1, 2)
        assert example.review_status == "reviewed"
        assert example.review_note == "Checked on page 1."
        assert example.pdf_path == "docs/source.pdf"
        assert example.line_number == 9

    def test_answerable_example_requires_expected_pages(self):
        with pytest.raises(ValueError, match="answerable cases require"):
            qa_eval._parse_example(
                {"id": "Q1", "question": "Question?", "answerable": True},
                1,
            )

    def test_unanswerable_example_cannot_have_expected_pages(self):
        with pytest.raises(ValueError, match="must not have"):
            qa_eval._parse_example(
                {
                    "id": "Q1",
                    "question": "Question?",
                    "answerable": False,
                    "expected_pages": [1],
                },
                1,
            )

    @pytest.mark.parametrize(
        "row",
        [
            {"question": "Question?", "answerable": True, "expected_pages": [1]},
            {"id": "Q1", "answerable": True, "expected_pages": [1]},
            {"id": "Q1", "question": " ", "answerable": True, "expected_pages": [1]},
        ],
    )
    def test_missing_case_id_or_question_is_rejected(self, row):
        with pytest.raises(ValueError):
            qa_eval._parse_example(row, 5)

    def test_review_note_must_be_a_string(self):
        with pytest.raises(TypeError, match="review note must be a string"):
            qa_eval._parse_example(
                {
                    "id": "Q1",
                    "question": "Question?",
                    "answerable": True,
                    "expected_pages": [1],
                    "review_note": 42,
                },
                1,
            )

    def test_pdf_path_must_be_a_string_when_supplied(self):
        with pytest.raises(ValueError, match="PDF path must be a string"):
            qa_eval._parse_example(
                {
                    "id": "Q1",
                    "question": "Question?",
                    "answerable": True,
                    "expected_pages": [1],
                    "pdf_path": 123,
                },
                1,
            )


class TestDatasetLoading:
    def test_loads_jsonl_skips_blank_lines_and_returns_sha256(self, tmp_path):
        dataset = tmp_path / "qa.jsonl"
        raw = (
            json.dumps(
                {
                    "id": "Q1",
                    "question": "Question one?",
                    "answerable": True,
                    "expected_pages": [1],
                }
            )
            + "\n\n"
            + json.dumps(
                {
                    "id": "Q2",
                    "question": "Question two?",
                    "answerable": False,
                }
            )
            + "\n"
        ).encode("utf-8")
        dataset.write_bytes(raw)

        examples, digest = qa_eval.load_dataset(dataset)

        assert [item.case_id for item in examples] == ["Q1", "Q2"]
        assert digest == hashlib.sha256(raw).hexdigest()
        assert examples[1].line_number == 3

    def test_missing_dataset_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            qa_eval.load_dataset(tmp_path / "missing.jsonl")

    def test_empty_dataset_is_rejected(self, tmp_path):
        dataset = tmp_path / "empty.jsonl"
        dataset.write_text("\n  \n", encoding="utf-8")
        with pytest.raises(ValueError, match="contains no examples"):
            qa_eval.load_dataset(dataset)

    def test_invalid_json_reports_line_number(self, tmp_path):
        dataset = tmp_path / "invalid.jsonl"
        dataset.write_text(
            '{"id": "Q1", "question": "Question?", '
            '"answerable": true, "expected_pages": [1]}\n'
            '{invalid}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Line 2: invalid JSON"):
            qa_eval.load_dataset(dataset)

    def test_non_object_jsonl_row_is_rejected(self, tmp_path):
        dataset = tmp_path / "array.jsonl"
        dataset.write_text("[]\n", encoding="utf-8")
        with pytest.raises(TypeError, match="each JSONL row must be an object"):
            qa_eval.load_dataset(dataset)

    def test_duplicate_case_ids_are_rejected(self, tmp_path):
        dataset = tmp_path / "duplicates.jsonl"
        row = {
            "id": "Q1",
            "question": "Question?",
            "answerable": True,
            "expected_pages": [1],
        }
        write_jsonl(dataset, [row, row])
        with pytest.raises(ValueError, match="Duplicate evaluation IDs: Q1"):
            qa_eval.load_dataset(dataset)


class TestDatasetValidation:
    def test_accepts_expected_counts_and_reviewed_labels(self):
        examples = [
            make_example(case_id="Q1", answerable=True, expected_pages=(1,)),
            make_example(
                case_id="Q2",
                answerable=False,
                expected_pages=(),
            ),
        ]

        qa_eval.validate_dataset(
            examples,
            expected_total=2,
            expected_answerable=1,
            expected_unanswerable=1,
        )

    def test_rejects_inconsistent_expected_count_arguments(self):
        with pytest.raises(ValueError, match="counts are inconsistent"):
            qa_eval.validate_dataset(
                [make_example()],
                expected_total=2,
                expected_answerable=1,
                expected_unanswerable=0,
            )

    def test_rejects_wrong_total_or_class_balance(self):
        examples = [
            make_example(case_id="Q1", answerable=True, expected_pages=(1,)),
            make_example(case_id="Q2", answerable=False, expected_pages=()),
        ]
        with pytest.raises(ValueError, match="Expected 3 examples, found 2"):
            qa_eval.validate_dataset(
                examples,
                expected_total=3,
                expected_answerable=2,
                expected_unanswerable=1,
            )

        with pytest.raises(ValueError, match="Expected 2 answerable examples"):
            qa_eval.validate_dataset(
                examples,
                expected_total=2,
                expected_answerable=2,
                expected_unanswerable=0,
            )

    @pytest.mark.parametrize(
        ("review_status", "review_note"),
        [("pending_review", "Checked"), ("reviewed", ""), ("needs_review", "")],
    )
    def test_blocks_gold_rows_without_completed_review(self, review_status, review_note):
        examples = [
            make_example(
                review_status=review_status,
                review_note=review_note,
            )
        ]
        with pytest.raises(ValueError, match="not fully reviewed"):
            qa_eval.validate_dataset(
                examples,
                expected_total=1,
                expected_answerable=1,
                expected_unanswerable=0,
            )


class TestPDFPathResolution:
    def test_prefers_project_relative_file_when_both_locations_exist(
        self,
        tmp_path,
        monkeypatch,
    ):
        project_root = tmp_path / "project"
        dataset_dir = tmp_path / "dataset"
        (project_root / "docs").mkdir(parents=True)
        (dataset_dir / "docs").mkdir(parents=True)
        project_file = project_root / "docs" / "source.pdf"
        dataset_file = dataset_dir / "docs" / "source.pdf"
        project_file.write_bytes(b"project")
        dataset_file.write_bytes(b"dataset")
        monkeypatch.setattr(qa_eval, "PROJECT_ROOT", project_root)

        resolved = qa_eval.resolve_pdf_path(
            make_example(pdf_path="docs/source.pdf"),
            default_pdf=None,
            dataset_path=dataset_dir / "qa.jsonl",
        )

        assert resolved == project_file.resolve()

    def test_uses_dataset_relative_fallback(self, tmp_path, monkeypatch):
        project_root = tmp_path / "project"
        dataset_dir = tmp_path / "fixtures"
        dataset_dir.mkdir()
        source = dataset_dir / "source.pdf"
        source.write_bytes(b"pdf")
        monkeypatch.setattr(qa_eval, "PROJECT_ROOT", project_root)

        resolved = qa_eval.resolve_pdf_path(
            make_example(pdf_path="source.pdf"),
            default_pdf=None,
            dataset_path=dataset_dir / "qa.jsonl",
        )

        assert resolved == source.resolve()

    def test_uses_default_pdf_when_example_has_no_path(self, tmp_path):
        source = tmp_path / "default.pdf"
        source.write_bytes(b"pdf")
        resolved = qa_eval.resolve_pdf_path(
            make_example(pdf_path=None),
            default_pdf=source,
            dataset_path=tmp_path / "qa.jsonl",
        )
        assert resolved == source.resolve()

    def test_requires_pdf_path_or_default(self, tmp_path):
        with pytest.raises(ValueError, match="has no pdf_path"):
            qa_eval.resolve_pdf_path(
                make_example(pdf_path=None),
                default_pdf=None,
                dataset_path=tmp_path / "qa.jsonl",
            )

    def test_missing_pdf_raises_with_case_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qa_eval, "PROJECT_ROOT", tmp_path / "project")
        with pytest.raises(FileNotFoundError, match="case Q001"):
            qa_eval.resolve_pdf_path(
                make_example(pdf_path="absent.pdf"),
                default_pdf=None,
                dataset_path=tmp_path / "qa.jsonl",
            )


class TestReportHelpers:
    def test_build_output_path_sanitizes_model_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qa_eval, "PROJECT_ROOT", tmp_path)
        output = qa_eval.build_output_path("qwen3.5:2b/production")

        assert output.parent == tmp_path / "data" / "phase3"
        assert re.fullmatch(
            r"qa_eval_\d{8}_qwen3_5_2b_production\.json",
            output.name,
        )

    def test_parse_args_accepts_dataset_pdf_output_and_validation_counts(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_qa_eval.py",
                "--dataset",
                "qa.jsonl",
                "--pdf",
                "source.pdf",
                "--output",
                "report.json",
                "--expected-total",
                "2",
                "--expected-answerable",
                "1",
                "--expected-unanswerable",
                "1",
                "--validate-only",
            ],
        )

        args = qa_eval.parse_args()

        assert args.dataset == Path("qa.jsonl")
        assert args.pdf == Path("source.pdf")
        assert args.output == Path("report.json")
        assert args.expected_total == 2
        assert args.expected_answerable == 1
        assert args.expected_unanswerable == 1
        assert args.validate_only is True


class TestRunEvaluation:
    def test_builds_report_caches_pdf_and_closes_service(
        self,
        tmp_path,
        monkeypatch,
    ):
        pdf = tmp_path / "source.pdf"
        pdf.write_bytes(b"mock pdf bytes")
        examples = [
            make_example(case_id="Q1", answerable=True, expected_pages=(1,), pdf_path=str(pdf)),
            make_example(
                case_id="Q2",
                answerable=False,
                expected_pages=(),
                pdf_path=str(pdf),
            ),
        ]
        settings = make_settings()
        monkeypatch.setattr(qa_eval, "get_settings", lambda: settings)

        client_instances = []

        class FakeClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                client_instances.append(self)

        monkeypatch.setattr(qa_eval, "OllamaClient", FakeClient)

        class FakeService:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.calls = []
                self.closed = False
                self.responses = [make_answered_response(1), make_abstained_response()]
                self.client = kwargs["llm_client"]
                fake_services.append(self)

            async def answer_pages(self, **kwargs):
                self.calls.append(kwargs)
                return self.responses.pop(0)

            async def aclose(self):
                self.closed = True

        fake_services = []
        monkeypatch.setattr(qa_eval, "AnswerGenerationService", FakeService)

        ingested = []

        def fake_ingest(pdf_bytes, *, filename):
            ingested.append((pdf_bytes, filename))
            return SimpleNamespace(
                pages=(object(),),
                document_id=DOCUMENT_ID,
                filename=filename,
            )

        monkeypatch.setattr(qa_eval, "ingest_pdf_upload", fake_ingest)

        report = asyncio.run(qa_eval.run_evaluation(
            examples,
            dataset_hash="a" * 64,
            dataset_path=tmp_path / "qa.jsonl",
            default_pdf=None,
        ))

        assert len(client_instances) == 1
        assert client_instances[0].kwargs["model"] == "qwen3.5:2b"
        service = fake_services[0]
        assert service.closed is True
        assert service.kwargs["top_k"] == settings.retrieval_top_k
        assert service.kwargs["evidence_threshold"] == settings.evidence_threshold
        assert len(ingested) == 1
        assert ingested[0] == (b"mock pdf bytes", "source.pdf")
        assert len(service.calls) == 2
        assert service.calls[0]["question"] == examples[0].question
        assert service.calls[0]["document_id"] == str(DOCUMENT_ID)

        assert report["schema_version"] == 1
        assert report["evaluation"] == "phase3_grounded_qa"
        assert report["dataset"] == {
            "path": str(tmp_path / "qa.jsonl"),
            "sha256": "a" * 64,
            "total_examples": 2,
            "answerable_examples": 1,
            "unanswerable_examples": 1,
        }
        assert report["model"]["name"] == "qwen3.5:2b"
        assert report["configuration"]["retrieval_top_k"] == 5
        assert report["metrics"]["classification_accuracy"] == 1.0
        assert report["metrics"]["answerable_answer_rate"] == 1.0
        assert report["metrics"]["unanswerable_abstention_rate"] == 1.0
        assert report["metrics"]["citation_page_accuracy"] == 1.0
        assert report["metrics"]["answerable_citation_coverage"] == 1.0
        assert report["metrics"]["confusion_matrix"] == {
            "true_answered": 1,
            "true_abstained": 1,
            "false_answered": 0,
            "false_abstained": 0,
        }
        assert len(report["results"]) == 2
        assert report["results"][0]["citation_page_matches"] == 1
        assert report["results"][1]["citation_count"] == 0
        assert report["metrics"]["median_latency_ms"] >= 0
        assert report["metrics"]["p95_latency_ms"] >= 0

    def test_service_is_closed_when_evaluation_fails(
        self,
        tmp_path,
        monkeypatch,
    ):
        pdf = tmp_path / "source.pdf"
        pdf.write_bytes(b"pdf")
        monkeypatch.setattr(qa_eval, "get_settings", make_settings)
        monkeypatch.setattr(qa_eval, "OllamaClient", lambda **kwargs: object())
        instances = []

        class FakeService:
            def __init__(self, **kwargs):
                self.closed = False
                instances.append(self)

            async def answer_pages(self, **kwargs):
                raise RuntimeError("synthetic service failure")

            async def aclose(self):
                self.closed = True

        monkeypatch.setattr(qa_eval, "AnswerGenerationService", FakeService)
        monkeypatch.setattr(
            qa_eval,
            "ingest_pdf_upload",
            lambda pdf_bytes, *, filename: SimpleNamespace(
                pages=(object(),), document_id=DOCUMENT_ID, filename=filename
            ),
        )

        with pytest.raises(RuntimeError, match="synthetic service failure"):
            asyncio.run(qa_eval.run_evaluation(
                [make_example(pdf_path=str(pdf))],
                dataset_hash="b" * 64,
                dataset_path=tmp_path / "qa.jsonl",
                default_pdf=None,
            ))

        assert instances[0].closed is True


class TestMainCLI:
    def _args(self, dataset: Path, **overrides) -> Namespace:
        values = {
            "dataset": dataset,
            "pdf": None,
            "output": None,
            "expected_total": 1,
            "expected_answerable": 1,
            "expected_unanswerable": 0,
            "validate_only": True,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_validate_only_prints_hash_and_skips_inference(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        dataset = tmp_path / "qa.jsonl"
        raw = write_jsonl(
            dataset,
            [
                {
                    "id": "Q1",
                    "question": "What title?",
                    "answerable": True,
                    "expected_pages": [1],
                    "review_status": "reviewed",
                    "review_note": "Verified against page 1.",
                }
            ],
        )
        monkeypatch.setattr(qa_eval, "parse_args", lambda: self._args(dataset))
        monkeypatch.setattr(
            qa_eval,
            "run_evaluation",
            lambda *args, **kwargs: pytest.fail("validation-only must not infer"),
        )

        assert qa_eval.main() == 0
        output = capsys.readouterr().out
        assert "QA dataset validation passed." in output
        assert "Examples: 1" in output
        assert hashlib.sha256(raw).hexdigest() in output
        assert "no Ollama requests were sent" in output

    def test_review_gate_failure_returns_one_and_writes_stderr(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        dataset = tmp_path / "pending.jsonl"
        write_jsonl(
            dataset,
            [
                {
                    "id": "Q1",
                    "question": "What title?",
                    "answerable": True,
                    "expected_pages": [1],
                    "review_status": "pending_review",
                    "review_note": "",
                }
            ],
        )
        monkeypatch.setattr(qa_eval, "parse_args", lambda: self._args(dataset))

        assert qa_eval.main() == 1
        captured = capsys.readouterr()
        assert "not fully reviewed" in captured.err

    def test_full_cli_run_writes_json_report(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        dataset = tmp_path / "qa.jsonl"
        write_jsonl(
            dataset,
            [
                {
                    "id": "Q1",
                    "question": "What title?",
                    "answerable": True,
                    "expected_pages": [1],
                    "review_status": "reviewed",
                    "review_note": "Verified against page 1.",
                    "pdf_path": "source.pdf",
                }
            ],
        )
        pdf = tmp_path / "source.pdf"
        pdf.write_bytes(b"pdf")
        output = tmp_path / "reports" / "qa.json"
        monkeypatch.setattr(
            qa_eval,
            "parse_args",
            lambda: self._args(dataset, output=output, validate_only=False),
        )
        observed = {}

        async def fake_run_evaluation(examples, **kwargs):
            observed["examples"] = examples
            observed.update(kwargs)
            return {
                "schema_version": 1,
                "model": {"name": "fake-model"},
                "metrics": {
                    "total_cases": 1,
                    "classification_accuracy": 1.0,
                    "answerable_answer_rate": 1.0,
                    "unanswerable_abstention_rate": None,
                    "citation_page_accuracy": 1.0,
                    "answerable_citation_coverage": 1.0,
                    "citation_pages_correct": 1,
                    "citation_pages_total": 1,
                    "confusion_matrix": {
                        "true_answered": 1,
                        "true_abstained": 0,
                        "false_answered": 0,
                        "false_abstained": 0,
                    },
                    "median_latency_ms": 12.0,
                    "p95_latency_ms": 12.0,
                    "semantic_answer_correctness": None,
                    "semantic_answer_correctness_note": (
                        "Not automatically assessed. Requires reviewed "
                        "answer-content labels or human evaluation."
                    ),
                },
                "results": [],
            }

        monkeypatch.setattr(qa_eval, "run_evaluation", fake_run_evaluation)

        assert qa_eval.main() == 0
        written = json.loads(output.read_text(encoding="utf-8"))
        assert written["model"]["name"] == "fake-model"
        assert written["metrics"]["classification_accuracy"] == 1.0
        assert len(observed["examples"]) == 1
        assert observed["default_pdf"] is None
        assert "Report saved to:" in capsys.readouterr().out

    def test_cli_errors_return_one(self, tmp_path, monkeypatch, capsys):
        missing_dataset = tmp_path / "missing.jsonl"
        monkeypatch.setattr(
            qa_eval,
            "parse_args",
            lambda: self._args(missing_dataset),
        )

        assert qa_eval.main() == 1
        assert "QA evaluation failed:" in capsys.readouterr().err
