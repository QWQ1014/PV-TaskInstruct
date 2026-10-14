import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = (
    ROOT
    / "evaluation_splits"
    / "pv_ecot_holdout_200_20260616"
    / "experiment_simpleqa_compare"
    / "codex_full_blind_judge"
)
SAMPLE = JUDGE_DIR / "codex_blind_judge_sample_200.jsonl"
HIDDEN_MAP = JUDGE_DIR / "codex_blind_judge_sample_200_map.hidden.jsonl"
OUT_JSONL = JUDGE_DIR / "codex_blind_judge_outputs_200.jsonl"
OUT_SUMMARY_JSON = JUDGE_DIR / "codex_blind_judge_summary_200.json"
OUT_SUMMARY_MD = JUDGE_DIR / "codex_blind_judge_summary_200.md"


BLIND_DECISIONS = [
    "A", "A", "B", "A", "B", "A", "A", "B", "A", "A",
    "B", "A", "B", "A", "B", "A", "A", "B", "B", "A",
    "B", "A", "B", "B", "A", "A", "A", "B", "B", "A",
    "B", "A", "A", "A", "B", "A", "A", "B", "A", "B",
    "B", "B", "B", "A", "A", "A", "B", "B", "B", "B",
    "B", "A", "A", "A", "A", "B", "A", "A", "B", "A",
    "A", "A", "B", "A", "A", "B", "B", "B", "A", "A",
    "A", "A", "B", "A", "A", "B", "B", "A", "B", "A",
    "B", "B", "B", "B", "B", "B", "B", "B", "B", "A",
    "A", "A", "B", "A", "A", "A", "A", "B", "B", "A",
    "B", "A", "B", "B", "A", "B", "A", "B", "B", "B",
    "B", "A", "A", "A", "B", "B", "A", "A", "A", "B",
    "B", "B", "A", "A", "B", "B", "A", "A", "B", "A",
    "A", "B", "B", "B", "B", "B", "B", "A", "A", "A",
    "A", "B", "A", "B", "B", "B", "B", "A", "B", "B",
    "B", "B", "A", "A", "B", "B", "B", "B", "A", "A",
    "A", "B", "B", "B", "A", "A", "A", "B", "A", "A",
    "A", "B", "B", "B", "B", "B", "A", "A", "B", "B",
    "A", "A", "A", "B", "B", "A", "B", "A", "A", "A",
    "A", "B", "B", "A", "B", "B", "A", "A", "A", "A",
]


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def counter_table(counter):
    total = sum(counter.values())
    compared = counter.get("base", 0) + counter.get("tuned", 0)
    return {
        "n": total,
        "base": counter.get("base", 0),
        "tuned": counter.get("tuned", 0),
        "tie": counter.get("tie", 0),
        "tuned_win_rate_excluding_ties": counter.get("tuned", 0) / max(1, compared),
        "tuned_preference_rate_including_ties": counter.get("tuned", 0) / max(1, total),
    }


def render_table(rows, header):
    lines = [header, "|---|---:|---:|---:|---:|---:|"]
    for key, value in rows.items():
        lines.append(
            f"| {key} | {value['n']} | {value['tuned']} | {value['base']} | "
            f"{value['tie']} | {value['tuned_win_rate_excluding_ties']:.3f} |"
        )
    return lines


def main() -> None:
    sample_rows = load_jsonl(SAMPLE)
    map_rows = {row["id"]: row for row in load_jsonl(HIDDEN_MAP)}
    if len(sample_rows) != len(BLIND_DECISIONS):
        raise ValueError(
            f"Decision count mismatch: sample={len(sample_rows)}, "
            f"decisions={len(BLIND_DECISIONS)}"
        )

    out_rows = []
    counts = Counter()
    by_task = defaultdict(Counter)
    by_cell = defaultdict(Counter)
    blind_counts = Counter(BLIND_DECISIONS)

    for row, blind_winner in zip(sample_rows, BLIND_DECISIONS):
        mapping = map_rows[row["id"]]
        model_winner = "tie" if blind_winner == "tie" else mapping[blind_winner]
        counts[model_winner] += 1
        by_task[row["task_type"]][model_winner] += 1
        by_cell[row["cell_type"]][model_winner] += 1
        out_rows.append(
            {
                "id": row["id"],
                "task_type": row["task_type"],
                "cell_type": row["cell_type"],
                "blind_winner": blind_winner,
                "model_winner": model_winner,
                "criterion": "closer_to_reference_mechanism_with_fewer_unsupported_claims",
            }
        )

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "sample_size": len(sample_rows),
        "blind_counts": dict(blind_counts),
        "winner_counts": dict(counts),
        "overall": counter_table(counts),
        "by_task_type": {key: counter_table(value) for key, value in sorted(by_task.items())},
        "by_cell_type": {key: counter_table(value) for key, value in sorted(by_cell.items())},
        "base_wins": [row for row in out_rows if row["model_winner"] == "base"],
        "judgment_file": str(OUT_JSONL.relative_to(ROOT)).replace("\\", "/"),
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Evidence-conditioned SimpleQA Full Blind Judge Summary",
        "",
        "Criterion: closer to the reference mechanism with fewer unsupported claims.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sample size | {summary['sample_size']} |",
        f"| Tuned wins | {counts.get('tuned', 0)} |",
        f"| Base wins | {counts.get('base', 0)} |",
        f"| Ties | {counts.get('tie', 0)} |",
        f"| Tuned win rate excl. ties | {summary['overall']['tuned_win_rate_excluding_ties']:.3f} |",
        f"| Tuned preference rate incl. ties | {summary['overall']['tuned_preference_rate_including_ties']:.3f} |",
        "",
        "## By Task Type",
        "",
    ]
    lines.extend(
        render_table(
            summary["by_task_type"],
            "| Task | N | Tuned | Base | Tie | Tuned win rate excl. ties |",
        )
    )
    lines.extend(["", "## By Cell Type", ""])
    lines.extend(
        render_table(
            summary["by_cell_type"],
            "| Cell type | N | Tuned | Base | Tie | Tuned win rate excl. ties |",
        )
    )
    lines.extend(["", "## Base-Win Cases", ""])
    if summary["base_wins"]:
        lines.append("| ID | Task | Cell type | Blind winner |")
        lines.append("|---|---|---|---|")
        for row in summary["base_wins"]:
            lines.append(
                f"| {row['id']} | {row['task_type']} | {row['cell_type']} | {row['blind_winner']} |"
            )
    else:
        lines.append("None.")

    OUT_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
