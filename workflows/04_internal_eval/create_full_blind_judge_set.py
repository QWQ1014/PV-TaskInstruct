import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args()

    rows = load_jsonl(args.aligned)
    rows.sort(key=lambda row: row["id"])
    rng = random.Random(args.seed)

    blind_rows = []
    map_rows = []
    for row in rows:
        if rng.random() < 0.5:
            answer_a = row["base_response"]
            answer_b = row["tuned_response"]
            mapping = {"A": "base", "B": "tuned"}
        else:
            answer_a = row["tuned_response"]
            answer_b = row["base_response"]
            mapping = {"A": "tuned", "B": "base"}

        blind_rows.append(
            {
                "id": row["id"],
                "cell_type": row["cell_type"],
                "task_type": row["task_type"],
                "question": row["question"],
                "reference_answer": row["reference_answer"],
                "answer_a": answer_a,
                "answer_b": answer_b,
            }
        )
        map_rows.append({"id": row["id"], **mapping})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.out_dir / "codex_blind_judge_sample_200.jsonl"
    map_path = args.out_dir / "codex_blind_judge_sample_200_map.hidden.jsonl"
    report_path = args.out_dir / "codex_blind_judge_sample_200_report.json"

    write_jsonl(sample_path, blind_rows)
    write_jsonl(map_path, map_rows)

    task_counts = {}
    cell_counts = {}
    for row in blind_rows:
        task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
        cell_counts[row["cell_type"]] = cell_counts.get(row["cell_type"], 0) + 1

    report = {
        "sample_size": len(blind_rows),
        "seed": args.seed,
        "task_counts": dict(sorted(task_counts.items())),
        "cell_counts": dict(sorted(cell_counts.items())),
        "blind_sample": str(sample_path),
        "hidden_map": str(map_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
