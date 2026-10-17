from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_REPORT = Path("data/phase0/nomic_embedding_smoke.json")
TIMEOUT_SECONDS = 120

SAME_TEXT = (
    "search_document: A DevOps engineer maintains CI/CD pipelines, cloud "
    "infrastructure, and deployment automation."
)
RELATED_TEXT = (
    "search_document: The role requires a bachelor's degree in Computer "
    "Science, Information Technology, or a related field."
)
UNRELATED_TEXT = "search_document: The capital city of Japan is Tokyo."
QUERY_TEXT = "search_query: What educational qualification does the role require?"


def post_json(url: str, payload: dict, timeout: int = TIMEOUT_SECONDS) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding returned a zero vector.")
    return dot / (left_norm * right_norm)


def write_report(report_path: Path, report: dict) -> None:
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline smoke test for a locally running Ollama embedding model."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    report: dict = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "base_url": base_url,
        "endpoint": "/api/embeddings",
        "paid_api_calls": 0,
        "tests": [],
        "limitations": [
            "This is a connectivity/dimensionality/repeatability smoke test, not a retrieval benchmark.",
            "The single toy similarity comparison is diagnostic only and is not a quality score.",
            "No vector index is built or evaluated by this script.",
        ],
    }

    print(f"Ollama base URL: {base_url}")
    print(f"Embedding model: {args.model}")
    print("Endpoint: /api/embeddings")
    print("Paid API calls: 0\n")

    try:
        started = time.perf_counter()
        # GET /api/tags is used to verify the model is available locally.
        with urlopen(f"{base_url}/api/tags", timeout=10) as response:
            tags = json.loads(response.read().decode("utf-8"))
        names = [item.get("name", "") for item in tags.get("models", [])]
        model_present = any(
            name == args.model or name.startswith(f"{args.model}:")
            for name in names
        )
        report["available_models"] = names
        report["model_present"] = model_present
        report["model_inventory_seconds"] = round(time.perf_counter() - started, 3)
        if not model_present:
            report["passed"] = False
            report["error"] = f"Model '{args.model}' was not found in Ollama tags."
            write_report(args.report, report)
            print(report["error"])
            print("Run `ollama list` and pass the exact model name with --model.")
            return 1

        inputs = [SAME_TEXT, SAME_TEXT, RELATED_TEXT, UNRELATED_TEXT, QUERY_TEXT]
        vectors: list[list[float]] = []
        for index, text in enumerate(inputs, start=1):
            started = time.perf_counter()
            payload = post_json(
                f"{base_url}/api/embeddings",
                {"model": args.model, "prompt": text},
            )
            latency = round(time.perf_counter() - started, 3)
            vector = payload.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"Request {index} returned no embedding vector.")
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in vector
            ):
                raise ValueError(f"Request {index} returned a non-finite vector value.")
            vectors.append([float(value) for value in vector])
            test_record = {
                "input_label": [
                    "repeat_a_cold_or_first",
                    "repeat_a_warm",
                    "qualification_document",
                    "unrelated_document",
                    "qualification_query",
                ][index - 1],
                "latency_seconds": latency,
                "dimensions": len(vector),
                "vector_nonzero": any(value != 0 for value in vector),
            }
            report["tests"].append(test_record)
            print(
                f"[{index}/{len(inputs)}] {test_record['input_label']}: "
                f"{latency:.2f}s, dimensions={len(vector)}"
            )

        dimensions = {len(vector) for vector in vectors}
        repeat_similarity = cosine_similarity(vectors[0], vectors[1])
        relevant_similarity = cosine_similarity(vectors[4], vectors[2])
        unrelated_similarity = cosine_similarity(vectors[4], vectors[3])
        latencies = [test["latency_seconds"] for test in report["tests"]]
        stable_dimensions = len(dimensions) == 1
        stable_repeat = repeat_similarity >= 0.999
        finite_nonzero = all(test["vector_nonzero"] for test in report["tests"])
        basic_pass = stable_dimensions and stable_repeat and finite_nonzero

        report.update({
            "dimensions": next(iter(dimensions)) if len(dimensions) == 1 else sorted(dimensions),
            "stable_dimensions": stable_dimensions,
            "same_input_cosine_similarity": round(repeat_similarity, 8),
            "same_input_repeatability_pass": stable_repeat,
            "qualification_query_similarity": round(relevant_similarity, 6),
            "unrelated_document_similarity": round(unrelated_similarity, 6),
            "toy_relevance_ordering_pass": relevant_similarity > unrelated_similarity,
            "median_latency_seconds": round(statistics.median(latencies), 3),
            "max_latency_seconds": round(max(latencies), 3),
            "basic_smoke_pass": basic_pass,
            "passed": basic_pass,
        })
        write_report(args.report, report)

        print("\nSUMMARY")
        print(json.dumps({
            key: report[key]
            for key in (
                "model_present",
                "dimensions",
                "stable_dimensions",
                "same_input_cosine_similarity",
                "same_input_repeatability_pass",
                "qualification_query_similarity",
                "unrelated_document_similarity",
                "toy_relevance_ordering_pass",
                "median_latency_seconds",
                "max_latency_seconds",
                "basic_smoke_pass",
            )
        }, indent=2))
        print(
            "\nInterpretation: a basic pass only confirms local embedding access, "
            "consistent dimensions, finite/nonzero vectors, and repeatability. "
            "It does not establish retrieval quality."
        )
        return 0 if basic_pass else 1

    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        write_report(args.report, report)
        print(f"Embedding smoke test failed: {report['error']}")
        print(
            "Confirm Ollama is running and `ollama list` shows the model. "
            "This script uses the local Ollama /api/embeddings endpoint."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
