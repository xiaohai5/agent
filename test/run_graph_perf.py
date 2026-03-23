from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from collections import Counter
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
class PerfSample:
    id: int
    question: str
    tags: list[str]


def load_samples(dataset_path: Path) -> list[PerfSample]:
    samples: list[PerfSample] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(
                PerfSample(
                    id=int(payload["id"]),
                    question=str(payload["question"]).strip(),
                    tags=[str(item).strip() for item in payload.get("tags", []) if str(item).strip()],
                )
            )
    if not samples:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return samples


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


async def run_one_sample(sample: PerfSample, *, top_k: int, user_id: int) -> dict[str, Any]:
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
        return {
            "id": sample.id,
            "question": sample.question,
            "tags": sample.tags,
            "ok": True,
            "latency_sec": round(latency_sec, 4),
            "status": _safe_text(result.get("status")) or "unknown",
            "actual_route": _safe_text(final_summary.get("route")) or "unknown",
            "answer_source": _safe_text(final_summary.get("answer_source")) or "unknown",
            "answer_length": len(_safe_text(result.get("answer"))),
            "pending_confirmation": bool(result.get("pending_confirmation")),
            "error": "",
        }
    except Exception as exc:
        latency_sec = time.perf_counter() - started_at
        return {
            "id": sample.id,
            "question": sample.question,
            "tags": sample.tags,
            "ok": False,
            "latency_sec": round(latency_sec, 4),
            "status": "error",
            "actual_route": "error",
            "answer_source": "",
            "answer_length": 0,
            "pending_confirmation": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_benchmark(samples: list[PerfSample], *, concurrency: int, top_k: int, user_id: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(concurrency, 1))
    results: list[dict[str, Any] | None] = [None] * len(samples)

    async def worker(index: int, sample: PerfSample) -> None:
        async with semaphore:
            results[index] = await run_one_sample(sample, top_k=top_k, user_id=user_id)

    await asyncio.gather(*(worker(index, sample) for index, sample in enumerate(samples)))
    return [item for item in results if item is not None]


def aggregate_metrics(records: list[dict[str, Any]], total_wall_time_sec: float) -> dict[str, Any]:
    latencies = [float(item["latency_sec"]) for item in records]
    success_records = [item for item in records if item.get("ok")]
    success_latencies = [float(item["latency_sec"]) for item in success_records]
    status_distribution = Counter(item.get("status", "unknown") or "unknown" for item in success_records)
    route_distribution = Counter(item.get("actual_route", "unknown") or "unknown" for item in success_records)
    answer_source_distribution = Counter(item.get("answer_source", "unknown") or "unknown" for item in success_records)

    return {
        "dataset_size": len(records),
        "success_count": len(success_records),
        "failure_count": len(records) - len(success_records),
        "success_rate": round(len(success_records) / len(records), 4) if records else 0.0,
        "total_wall_time_sec": round(total_wall_time_sec, 4),
        "throughput_qps": round(len(records) / total_wall_time_sec, 4) if total_wall_time_sec > 0 else 0.0,
        "avg_latency_sec": round(statistics.fmean(latencies), 4) if latencies else 0.0,
        "p50_latency_sec": round(percentile(latencies, 0.50), 4) if latencies else 0.0,
        "p90_latency_sec": round(percentile(latencies, 0.90), 4) if latencies else 0.0,
        "p95_latency_sec": round(percentile(latencies, 0.95), 4) if latencies else 0.0,
        "max_latency_sec": round(max(latencies), 4) if latencies else 0.0,
        "success_avg_latency_sec": round(statistics.fmean(success_latencies), 4) if success_latencies else 0.0,
        "status_distribution": dict(status_distribution),
        "route_distribution": dict(route_distribution),
        "answer_source_distribution": dict(answer_source_distribution),
    }


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detail_path = output_dir / f"travel_perf_detail_{timestamp}.json"
    summary_path = output_dir / f"travel_perf_summary_{timestamp}.json"
    detail_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return detail_path, summary_path


def print_console_summary(summary: dict[str, Any], detail_path: Path, summary_path: Path) -> None:
    print("=== Travel Graph Performance Summary ===")
    print(f"dataset_size: {summary['dataset_size']}")
    print(f"success_rate: {summary['success_rate']}")
    print(f"avg_latency_sec: {summary['avg_latency_sec']}")
    print(f"p50_latency_sec: {summary['p50_latency_sec']}")
    print(f"p90_latency_sec: {summary['p90_latency_sec']}")
    print(f"p95_latency_sec: {summary['p95_latency_sec']}")
    print(f"max_latency_sec: {summary['max_latency_sec']}")
    print(f"throughput_qps: {summary['throughput_qps']}")
    print(f"status_distribution: {json.dumps(summary['status_distribution'], ensure_ascii=False)}")
    print(f"route_distribution: {json.dumps(summary['route_distribution'], ensure_ascii=False)}")
    print(f"answer_source_distribution: {json.dumps(summary['answer_source_distribution'], ensure_ascii=False)}")
    print(f"summary_file: {summary_path}")
    print(f"detail_file: {detail_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run performance benchmark for the current chat graph.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="JSONL dataset path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for result files.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests.")
    parser.add_argument("--top-k", type=int, default=3, help="Graph top_k value.")
    parser.add_argument("--user-id", type=int, default=1, help="Graph user id.")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    samples = load_samples(args.dataset)
    started_at = time.perf_counter()
    records = await run_benchmark(samples, concurrency=args.concurrency, top_k=args.top_k, user_id=args.user_id)
    total_wall_time_sec = time.perf_counter() - started_at
    summary = aggregate_metrics(records, total_wall_time_sec)
    detail_path, summary_path = write_outputs(args.output_dir, records, summary)
    print_console_summary(summary, detail_path, summary_path)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
