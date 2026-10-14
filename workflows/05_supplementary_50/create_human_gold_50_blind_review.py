import argparse
import csv
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--out_review", default="human_gold_50_blind_review.csv")
    parser.add_argument("--out_key", default="human_gold_50_blind_review_key.csv")
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    gold_rows = read_jsonl(Path(args.gold))
    base_rows = {row["id"]: row for row in read_jsonl(Path(args.base))}
    tuned_rows = {row["id"]: row for row in read_jsonl(Path(args.tuned))}
    rng = random.Random(args.seed)

    review_rows = []
    key_rows = []
    for gold in gold_rows:
        sample_id = gold["id"]
        pair = [
            ("base", base_rows[sample_id].get("response", "")),
            ("tuned", tuned_rows[sample_id].get("response", "")),
        ]
        rng.shuffle(pair)
        review_rows.append(
            {
                "id": sample_id,
                "source_file": gold.get("source_file", ""),
                "cell_type": gold.get("cell_type", ""),
                "task_type": gold.get("task_type", ""),
                "question": gold.get("question", ""),
                "evidence": gold.get("evidence", ""),
                "reference_answer": gold.get("reference_answer", ""),
                "response_A": pair[0][1],
                "response_B": pair[1][1],
                "which_is_better": "",
                "judge_notes": "",
            }
        )
        key_rows.append(
            {
                "id": sample_id,
                "A_model": pair[0][0],
                "B_model": pair[1][0],
            }
        )

    review_path = Path(args.out_review)
    key_path = Path(args.out_key)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    with review_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    with key_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)

    print(f"review={review_path}")
    print(f"key={key_path}")
    print(f"rows={len(review_rows)}")


if __name__ == "__main__":
    main()
