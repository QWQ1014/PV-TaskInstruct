import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    return "".join(str(text or "").split())


def lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    # Character-level LCS works reasonably for Chinese mechanism answers.
    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    pred_n = normalize_text(pred)
    ref_n = normalize_text(ref)
    if not pred_n or not ref_n:
        return 0.0
    lcs = lcs_len(pred_n, ref_n)
    precision = lcs / len(pred_n)
    recall = lcs / len(ref_n)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="Human gold JSONL with reference_answer field.")
    parser.add_argument("--base", required=True, help="Base model outputs JSONL.")
    parser.add_argument("--tuned", required=True, help="Tuned model outputs JSONL.")
    parser.add_argument("--out_prefix", default="human_gold_50_eval")
    args = parser.parse_args()

    gold_rows = read_jsonl(Path(args.gold))
    base_rows = {row["id"]: row for row in read_jsonl(Path(args.base))}
    tuned_rows = {row["id"]: row for row in read_jsonl(Path(args.tuned))}

    detail_rows = []
    for gold in gold_rows:
        sample_id = gold["id"]
        ref = gold.get("reference_answer", "")
        base_resp = base_rows.get(sample_id, {}).get("response", "")
        tuned_resp = tuned_rows.get(sample_id, {}).get("response", "")
        base_score = rouge_l_f1(base_resp, ref)
        tuned_score = rouge_l_f1(tuned_resp, ref)
        if tuned_score > base_score:
            winner = "tuned"
        elif base_score > tuned_score:
            winner = "base"
        else:
            winner = "tie"
        detail_rows.append(
            {
                "id": sample_id,
                "source_file": gold.get("source_file", ""),
                "cell_type": gold.get("cell_type", ""),
                "task_type": gold.get("task_type", ""),
                "base_rouge_l": round(base_score, 6),
                "tuned_rouge_l": round(tuned_score, 6),
                "delta": round(tuned_score - base_score, 6),
                "winner": winner,
                "base_len": len(base_resp),
                "tuned_len": len(tuned_resp),
                "base_error": str(base_resp).startswith("ERROR:"),
                "tuned_error": str(tuned_resp).startswith("ERROR:"),
                "question": gold.get("question", ""),
                "reference_answer": ref,
                "base_response": base_resp,
                "tuned_response": tuned_resp,
            }
        )

    n = len(detail_rows)
    base_mean = sum(row["base_rouge_l"] for row in detail_rows) / n
    tuned_mean = sum(row["tuned_rouge_l"] for row in detail_rows) / n
    winners = {
        "tuned": sum(row["winner"] == "tuned" for row in detail_rows),
        "base": sum(row["winner"] == "base" for row in detail_rows),
        "tie": sum(row["winner"] == "tie" for row in detail_rows),
    }
    summary = {
        "n": n,
        "base_rouge_l_mean": round(base_mean, 6),
        "tuned_rouge_l_mean": round(tuned_mean, 6),
        "delta_mean": round(tuned_mean - base_mean, 6),
        "winners": winners,
        "base_error_count": sum(row["base_error"] for row in detail_rows),
        "tuned_error_count": sum(row["tuned_error"] for row in detail_rows),
        "base_avg_len": round(sum(row["base_len"] for row in detail_rows) / n, 2),
        "tuned_avg_len": round(sum(row["tuned_len"] for row in detail_rows) / n, 2),
    }

    out_prefix = Path(args.out_prefix)
    detail_csv = out_prefix.with_suffix(".details.csv")
    summary_json = out_prefix.with_suffix(".summary.json")
    summary_md = out_prefix.with_suffix(".summary.md")

    with detail_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(
        "\n".join(
            [
                "# Human Gold 50 Evaluation Summary",
                "",
                f"- n: {summary['n']}",
                f"- Base ROUGE-L mean: {summary['base_rouge_l_mean']}",
                f"- Tuned ROUGE-L mean: {summary['tuned_rouge_l_mean']}",
                f"- Delta: {summary['delta_mean']}",
                f"- Winners: tuned={winners['tuned']}, base={winners['base']}, tie={winners['tie']}",
                f"- Error count: base={summary['base_error_count']}, tuned={summary['tuned_error_count']}",
                f"- Average length: base={summary['base_avg_len']}, tuned={summary['tuned_avg_len']}",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"detail_csv={detail_csv}")
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()
