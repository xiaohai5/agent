from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 项目根目录必须在 sys.path 上，`backend` 才是可导入的包（包名 = 根目录下文件夹名）
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.graphs.chat_graph import run_chat_graph


DEFAULT_DATASET_PATH = Path(__file__).with_name("travel_queries_100.jsonl")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")


@dataclass(slots=True)
class EvalSample:
    id: int
    question: str
    expected_route: str
    tags: list[str]


def load_samples(dataset_path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(
                EvalSample(
                    id=int(payload["id"]),
                    question=str(payload["question"]).strip(),
                    expected_route=str(payload["expected_route"]).strip() or "other",
                    tags=[str(item).strip() for item in payload.get("tags", []) if str(item).strip()],
                )
            )
    if not samples:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return samples



def sample_dataset(samples: list[EvalSample], *, sample_size: int, seed: int | None) -> list[EvalSample]:
    if sample_size <= 0 or sample_size >= len(samples):
        return samples
    rng = random.Random(seed)
    return rng.sample(samples, sample_size)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    weight = rank - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def _safe_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


async def run_one_sample(sample: EvalSample, *, top_k: int, user_id: int) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        result = await run_chat_graph(
            question=sample.question,
            top_k=top_k,
            user_id=user_id,
            history=[],
        )
        latency_sec = time.perf_counter() - started_at
        final_summary = _safe_dict(result.get("final_summary"))
        verification = _safe_dict(final_summary.get("verification"))
        actual_route = _safe_text(final_summary.get("route")) or "unknown"
        answer = _safe_text(result.get("answer"))
        covered_needs = _safe_list(verification.get("covered_needs"))
        missing_needs = _safe_list(verification.get("missing_needs"))
        unsupported_needs = _safe_list(verification.get("unsupported_needs"))
        return {
            "id": sample.id,
            "question": sample.question,
            "expected_route": sample.expected_route,
            "actual_route": actual_route,
            "route_match": actual_route == sample.expected_route,
            "tags": sample.tags,
            "ok": True,
            "latency_sec": round(latency_sec, 4),
            "status": _safe_text(result.get("status")) or "unknown",
            "answer_source": _safe_text(final_summary.get("answer_source")) or "unknown",
            "verification_is_complete": verification.get("is_complete"),
            "covered_needs_count": len(covered_needs),
            "missing_needs_count": len(missing_needs),
            "unsupported_needs_count": len(unsupported_needs),
            "answer_length": len(answer),
            "has_answer": bool(answer),
            "pending_confirmation": bool(result.get("pending_confirmation")),
            "error": "",
        }
    except Exception as exc:
        latency_sec = time.perf_counter() - started_at
        return {
            "id": sample.id,
            "question": sample.question,
            "expected_route": sample.expected_route,
            "actual_route": "error",
            "route_match": False,
            "tags": sample.tags,
            "ok": False,
            "latency_sec": round(latency_sec, 4),
            "status": "error",
            "answer_source": "",
            "verification_is_complete": None,
            "covered_needs_count": 0,
            "missing_needs_count": 0,
            "unsupported_needs_count": 0,
            "answer_length": 0,
            "has_answer": False,
            "pending_confirmation": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_benchmark(samples: list[EvalSample], *, concurrency: int, top_k: int, user_id: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    results: list[dict[str, Any] | None] = [None] * len(samples)

    async def worker(index: int, sample: EvalSample) -> None:
        async with semaphore:
            results[index] = await run_one_sample(sample, top_k=top_k, user_id=user_id)

    await asyncio.gather(*(worker(index, sample) for index, sample in enumerate(samples)))
    return [item for item in results if item is not None]


def aggregate_metrics(
    records: list[dict[str, Any]],
    total_wall_time_sec: float,
    *,
    total_dataset_size: int,
    sample_ids: list[int],
    sample_seed: int | None,
) -> dict[str, Any]:
    latencies = [float(item["latency_sec"]) for item in records if isinstance(item.get("latency_sec"), (int, float))]
    success_records = [item for item in records if item.get("ok")]
    route_matches = [item for item in success_records if item.get("route_match")]
    completed_records = [item for item in success_records if item.get("status") == "completed"]
    confirmation_records = [item for item in success_records if item.get("status") == "needs_confirmation"]
    verification_complete_records = [item for item in success_records if item.get("verification_is_complete") is True]

    route_counter = Counter(item.get("actual_route", "unknown") or "unknown" for item in success_records)
    source_counter = Counter(item.get("answer_source", "unknown") or "unknown" for item in success_records)
    status_counter = Counter(item.get("status", "unknown") or "unknown" for item in success_records)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item.get("expected_route", "unknown")].append(item)

    by_expected_route: dict[str, dict[str, Any]] = {}
    for route_name, items in grouped.items():
        ok_items = [item for item in items if item.get("ok")]
        by_expected_route[route_name] = {
            "count": len(items),
            "success_rate": round(len(ok_items) / len(items), 4) if items else 0.0,
            "route_accuracy": round(sum(1 for item in ok_items if item.get("route_match")) / len(ok_items), 4) if ok_items else 0.0,
            "avg_latency_sec": round(statistics.fmean(float(item["latency_sec"]) for item in ok_items), 4) if ok_items else 0.0,
            "verification_complete_rate": round(
                sum(1 for item in ok_items if item.get("verification_is_complete") is True) / len(ok_items), 4
            ) if ok_items else 0.0,
        }

    return {
        "dataset_size": len(records),
        "total_dataset_size": total_dataset_size,
        "sample_ids": sample_ids,
        "sample_seed": sample_seed,
        "success_count": len(success_records),
        "failure_count": len(records) - len(success_records),
        "success_rate": round(len(success_records) / len(records), 4) if records else 0.0,
        "route_accuracy": round(len(route_matches) / len(success_records), 4) if success_records else 0.0,
        "completed_rate": round(len(completed_records) / len(success_records), 4) if success_records else 0.0,
        "needs_confirmation_rate": round(len(confirmation_records) / len(success_records), 4) if success_records else 0.0,
        "verification_complete_rate": round(len(verification_complete_records) / len(success_records), 4) if success_records else 0.0,
        "avg_latency_sec": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "p50_latency_sec": round(percentile(latencies, 0.50), 4) if latencies else 0.0,
        "p90_latency_sec": round(percentile(latencies, 0.90), 4) if latencies else 0.0,
        "p95_latency_sec": round(percentile(latencies, 0.95), 4) if latencies else 0.0,
        "max_latency_sec": round(max(latencies), 4) if latencies else 0.0,
        "throughput_qps": round(len(records) / total_wall_time_sec, 4) if total_wall_time_sec > 0 else 0.0,
        "total_wall_time_sec": round(total_wall_time_sec, 4),
        "route_distribution": dict(route_counter),
        "answer_source_distribution": dict(source_counter),
        "status_distribution": dict(status_counter),
        "by_expected_route": by_expected_route,
    }


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"travel_eval_detail_{timestamp}.json"
    summary_path = output_dir / f"travel_eval_summary_{timestamp}.json"
    detail_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return detail_path, summary_path


def print_console_summary(summary: dict[str, Any], detail_path: Path, summary_path: Path) -> None:
    print("=== Travel Graph Evaluation Summary ===")
    print(f"dataset_size: {summary['dataset_size']}")
    print(f"total_dataset_size: {summary['total_dataset_size']}")
    print(f"sample_seed: {summary['sample_seed']}")
    print(f"sample_ids: {json.dumps(summary['sample_ids'], ensure_ascii=False)}")
    print(f"success_rate: {summary['success_rate']}")
    print(f"route_accuracy: {summary['route_accuracy']}")
    print(f"verification_complete_rate: {summary['verification_complete_rate']}")
    print(f"needs_confirmation_rate: {summary['needs_confirmation_rate']}")
    print(f"avg_latency_sec: {summary['avg_latency_sec']}")
    print(f"p50_latency_sec: {summary['p50_latency_sec']}")
    print(f"p90_latency_sec: {summary['p90_latency_sec']}")
    print(f"p95_latency_sec: {summary['p95_latency_sec']}")
    print(f"max_latency_sec: {summary['max_latency_sec']}")
    print(f"throughput_qps: {summary['throughput_qps']}")
    print(f"route_distribution: {json.dumps(summary['route_distribution'], ensure_ascii=False)}")
    print(f"answer_source_distribution: {json.dumps(summary['answer_source_distribution'], ensure_ascii=False)}")
    print(f"status_distribution: {json.dumps(summary['status_distribution'], ensure_ascii=False)}")
    print(f"summary_file: {summary_path}")
    print(f"detail_file: {detail_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation benchmark for the current chat graph.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="JSONL dataset path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for result files.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests.")
    parser.add_argument("--sample-size", type=int, default=10, help="Random sample size per run; use <=0 to evaluate the full dataset.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Optional random seed for reproducible sampling.")
    parser.add_argument("--top-k", type=int, default=3, help="Graph top_k value.")
    parser.add_argument("--user-id", type=int, default=1, help="Graph user id.")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    all_samples = load_samples(args.dataset)
    samples = sample_dataset(all_samples, sample_size=args.sample_size, seed=args.sample_seed)
    started_at = time.perf_counter()
    records = await run_benchmark(samples, concurrency=args.concurrency, top_k=args.top_k, user_id=args.user_id)
    total_wall_time_sec = time.perf_counter() - started_at
    summary = aggregate_metrics(
        records,
        total_wall_time_sec,
        total_dataset_size=len(all_samples),
        sample_ids=[sample.id for sample in samples],
        sample_seed=args.sample_seed,
    )
    detail_path, summary_path = write_outputs(args.output_dir, records, summary)
    print_console_summary(summary, detail_path, summary_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
