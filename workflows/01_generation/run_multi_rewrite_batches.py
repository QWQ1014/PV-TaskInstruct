#!/usr/bin/env python3
"""Run the multi-COT rewrite pipeline in fixed-size index batches."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "multi_cot_enriched" / "expert_cot_dataset.multi_enriched.jsonl"
DEFAULT_TARGET = ROOT / "multi_cot_enriched" / "expert_cot_dataset.multi_rewrite_full_batch30.jsonl"
DEFAULT_LOG = ROOT / "multi_cot_enriched" / "expert_cot_dataset.multi_rewrite_full_batch30.log"
DEFAULT_SUMMARY = ROOT / "multi_cot_enriched" / "expert_cot_dataset.multi_rewrite_full_batch30.summary.json"


SAVED_RE = re.compile(r"saved=.* done=(?P<done>\d+) skipped=(?P<skipped>\d+) failed=(?P<failed>\d+)")


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def parse_saved(stdout: str) -> tuple[int, int, int] | None:
    matches = list(SAVED_RE.finditer(stdout))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group("done")), int(match.group("skipped")), int(match.group("failed"))


def run_batch(args: argparse.Namespace, start: int, end: int) -> tuple[int, int, int]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("build_pv_attribution_dataset.py")),
        "--source",
        str(args.source),
        "--target",
        str(args.target),
        "--indices",
        f"{start}-{end}",
        "--api",
        args.api,
        "--concurrency",
        str(args.concurrency),
        "--resume",
    ]
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    append_log(args.log, f"\n=== batch {start}-{end} started {started} ===\n")
    append_log(args.log, " ".join(cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if proc.stdout:
        append_log(args.log, proc.stdout)
    if proc.stderr:
        append_log(args.log, "\n[stderr]\n" + proc.stderr)
    append_log(args.log, f"=== batch {start}-{end} exit={proc.returncode} finished {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    if proc.returncode != 0:
        raise RuntimeError(f"batch {start}-{end} exited with code {proc.returncode}")
    parsed = parse_saved(proc.stdout)
    if parsed is None:
        raise RuntimeError(f"batch {start}-{end} did not print a saved= summary")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PV multi-COT rewrite in fixed-size batches.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--api", default="silicon_deepseek_r1_0528")
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    total_rows = count_jsonl_rows(args.source)
    end_index = args.end_index or total_rows
    if args.start_index < 1 or end_index < args.start_index:
        raise SystemExit("Invalid start/end index range.")
    end_index = min(end_index, total_rows)

    summary = {
        "source": str(args.source),
        "target": str(args.target),
        "log": str(args.log),
        "api": args.api,
        "chunk_size": args.chunk_size,
        "concurrency": args.concurrency,
        "total_source_rows": total_rows,
        "start_index": args.start_index,
        "end_index": end_index,
        "batches": [],
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    batch_count = 0
    total_done = 0
    total_skipped = 0
    total_failed = 0
    try:
        start = args.start_index
        while start <= end_index:
            if args.max_batches is not None and batch_count >= args.max_batches:
                summary["status"] = "stopped_at_max_batches"
                break
            end = min(start + args.chunk_size - 1, end_index)
            done, skipped, failed = run_batch(args, start, end)
            batch_count += 1
            total_done += done
            total_skipped += skipped
            total_failed += failed
            batch_info = {
                "range": f"{start}-{end}",
                "done": done,
                "skipped": skipped,
                "failed": failed,
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            summary["batches"].append(batch_info)
            summary.update(
                {
                    "last_batch": batch_info,
                    "batch_count": batch_count,
                    "total_done_this_run": total_done,
                    "total_skipped_this_run": total_skipped,
                    "total_failed_this_run": total_failed,
                    "target_rows_now": count_jsonl_rows(args.target) if args.target.exists() else 0,
                }
            )
            args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if done == 0 and failed > 0:
                summary["status"] = "stopped_after_zero_success_batch"
                raise RuntimeError(f"batch {start}-{end} had zero successful rows and {failed} failures")
            start = end + 1
        else:
            summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        append_log(args.log, f"\n[FATAL] {exc}\n")
        raise
    finally:
        summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
