import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EVAL = Path("data/holdout_eval.jsonl")
DEFAULT_OUT_DIR = Path("outputs/internal_eval")


RESPONSE_KEYS = [
    "response",
    "generated_text",
    "prediction",
    "predict",
    "output",
    "answer",
    "text",
]
MAX_ROUGE_TOKENS = 300


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Row is not an object at {path}:{line_no}")
            rows.append(obj)
    return rows


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.replace("\u00a0", " ").split())


def get_response(row: dict[str, Any]) -> str:
    for key in RESPONSE_KEYS:
        if key in row:
            return str(row[key] or "").strip()
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict) and normalize_text(last.get("content")):
            return str(last["content"]).strip()
    raise ValueError(f"Cannot find response text in row with keys: {list(row.keys())}")


def index_outputs(rows: list[dict[str, Any]], label: str) -> dict[str, str]:
    indexed = {}
    duplicate_ids = []
    missing_id_count = 0
    for idx, row in enumerate(rows, start=1):
        row_id = row.get("id")
        if not row_id:
            missing_id_count += 1
            row_id = f"pv_ecot_test_{idx:03d}"
        if row_id in indexed:
            duplicate_ids.append(row_id)
        indexed[str(row_id)] = get_response(row)
    if duplicate_ids:
        raise ValueError(f"{label} has duplicate ids: {duplicate_ids[:10]}")
    if missing_id_count:
        print(f"[warn] {label}: {missing_id_count} rows had no id; assigned by row order.")
    return indexed


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def tokenize_for_rouge(text: str) -> list[str]:
    text = normalize_text(text)
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in "_+-./%"):
            buf.append(ch.lower())
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
            if not ch.isspace():
                tokens.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = tokenize_for_rouge(prediction)[:MAX_ROUGE_TOKENS]
    ref_tokens = tokenize_for_rouge(reference)[:MAX_ROUGE_TOKENS]
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Align base/tuned outputs and compute ROUGE-L for PV-ECoT experiment 1.")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--base", type=Path, required=True, help="Base model output JSONL.")
    parser.add_argument("--tuned", type=Path, required=True, help="PV-ECoT tuned model output JSONL.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = load_jsonl(args.eval)
    base_outputs = index_outputs(load_jsonl(args.base), "base")
    tuned_outputs = index_outputs(load_jsonl(args.tuned), "tuned")

    eval_ids = [str(row["id"]) for row in eval_rows]
    missing_base = [row_id for row_id in eval_ids if row_id not in base_outputs]
    missing_tuned = [row_id for row_id in eval_ids if row_id not in tuned_outputs]
    extra_base = sorted(set(base_outputs) - set(eval_ids))
    extra_tuned = sorted(set(tuned_outputs) - set(eval_ids))
    if missing_base or missing_tuned:
        raise ValueError(
            "Missing outputs: "
            f"base={missing_base[:10]} ({len(missing_base)}), "
            f"tuned={missing_tuned[:10]} ({len(missing_tuned)})"
        )

    aligned = []
    base_scores = []
    tuned_scores = []
    by_cell = defaultdict(lambda: {"base": [], "tuned": []})
    by_task = defaultdict(lambda: {"base": [], "tuned": []})
    wins = Counter()

    for row in eval_rows:
        row_id = str(row["id"])
        reference = row["reference_answer"]
        base_response = base_outputs[row_id]
        tuned_response = tuned_outputs[row_id]
        base_rouge = rouge_l_f1(base_response, reference)
        tuned_rouge = rouge_l_f1(tuned_response, reference)
        base_scores.append(base_rouge)
        tuned_scores.append(tuned_rouge)
        by_cell[row["cell_type"]]["base"].append(base_rouge)
        by_cell[row["cell_type"]]["tuned"].append(tuned_rouge)
        by_task[row["task_type"]]["base"].append(base_rouge)
        by_task[row["task_type"]]["tuned"].append(tuned_rouge)
        if tuned_rouge > base_rouge:
            wins["tuned"] += 1
        elif tuned_rouge < base_rouge:
            wins["base"] += 1
        else:
            wins["tie"] += 1
        aligned.append(
            {
                **row,
                "base_response": base_response,
                "tuned_response": tuned_response,
                "base_rouge_l": round(base_rouge, 6),
                "tuned_rouge_l": round(tuned_rouge, 6),
            }
        )

    aligned_path = args.out_dir / "experiment1_aligned_outputs.jsonl"
    aligned_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned),
        encoding="utf-8",
    )

    report = {
        "eval_rows": len(eval_rows),
        "base_output_rows": len(base_outputs),
        "tuned_output_rows": len(tuned_outputs),
        "extra_base_ids": len(extra_base),
        "extra_tuned_ids": len(extra_tuned),
        "base_rouge_l_mean": mean(base_scores),
        "tuned_rouge_l_mean": mean(tuned_scores),
        "rouge_l_delta": mean(tuned_scores) - mean(base_scores),
        "rouge_l_win_counts": dict(wins),
        "by_cell_type": {
            key: {
                "n": len(value["base"]),
                "base_rouge_l": mean(value["base"]),
                "tuned_rouge_l": mean(value["tuned"]),
                "delta": mean(value["tuned"]) - mean(value["base"]),
            }
            for key, value in sorted(by_cell.items())
        },
        "by_task_type": {
            key: {
                "n": len(value["base"]),
                "base_rouge_l": mean(value["base"]),
                "tuned_rouge_l": mean(value["tuned"]),
                "delta": mean(value["tuned"]) - mean(value["base"]),
            }
            for key, value in sorted(by_task.items())
        },
    }
    (args.out_dir / "experiment1_rouge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def table_group(title: str, data: dict[str, dict[str, float]]) -> list[str]:
        lines = [f"## {title}", "", "| Group | N | Base ROUGE-L | Tuned ROUGE-L | Delta |", "|---|---:|---:|---:|---:|"]
        for key, value in data.items():
            lines.append(
                f"| {key} | {value['n']} | {value['base_rouge_l']:.4f} | "
                f"{value['tuned_rouge_l']:.4f} | {value['delta']:.4f} |"
            )
        lines.append("")
        return lines

    md = [
        "# PV-ECoT Experiment 1 ROUGE-L Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Evaluation rows | {len(eval_rows)} |",
        f"| Base mean ROUGE-L | {report['base_rouge_l_mean']:.4f} |",
        f"| Tuned mean ROUGE-L | {report['tuned_rouge_l_mean']:.4f} |",
        f"| Delta | {report['rouge_l_delta']:.4f} |",
        f"| Tuned wins | {wins['tuned']} |",
        f"| Base wins | {wins['base']} |",
        f"| Ties | {wins['tie']} |",
        "",
        *table_group("By Task Type", report["by_task_type"]),
        *table_group("By Cell Type", report["by_cell_type"]),
        f"Aligned outputs: `{aligned_path}`",
    ]
    (args.out_dir / "experiment1_rouge_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
