import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FINAL_JSONL = Path("data/base_final.jsonl")
ENRICHED_JSONL = Path("data/base_enriched.jsonl")
OUT_DIR = Path("outputs/holdout")

SEED = 20260616
TEST_SIZE = 200
TEST_INPUT_MODE = "literature_input"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(obj)
    return rows


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def get_enriched_for_final_row(final_row: dict[str, Any], enriched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    meta = final_row.get("meta_info") or {}
    try:
        source_index = int(meta["source_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid meta_info.source_index: {meta!r}") from exc
    if source_index < 1 or source_index > len(enriched_rows):
        raise ValueError(f"source_index {source_index} out of range 1..{len(enriched_rows)}")
    return enriched_rows[source_index - 1]


def enrich_final_row(final_row: dict[str, Any], enriched_row: dict[str, Any], split: str) -> dict[str, Any]:
    row = dict(final_row)
    meta = dict(row.get("meta_info") or {})
    enriched_meta = enriched_row.get("meta_info") or {}
    provenance = enriched_row.get("provenance") or {}
    source_file = enriched_row.get("source_file") or f"source_index_{meta.get('source_index')}"

    meta.update(
        {
            "split": split,
            "source_file": source_file,
            "source_file_hash": stable_id(source_file),
            "source_row_id": enriched_row.get("row_id"),
            "sample_uid": enriched_row.get("sample_uid"),
            "cell_type": enriched_meta.get("cell_type") or provenance.get("final_cell_type") or "unknown",
            "chain_type": enriched_meta.get("chain_type") or "unknown",
            "sample_type": enriched_meta.get("sample_type") or provenance.get("sample_type_auto") or "unknown",
            "chain_quality": enriched_meta.get("chain_quality") or provenance.get("chain_grade_auto") or "unknown",
            "grounded_quality": enriched_meta.get("grounded_quality") or provenance.get("grounded_grade_auto") or "unknown",
        }
    )
    row["meta_info"] = meta
    return row


def largest_remainder_targets(counts: Counter, total: int) -> dict[str, int]:
    grand = sum(counts.values())
    raw = {key: counts[key] * total / grand for key in counts}
    targets = {key: int(raw[key]) for key in counts}
    remainder = total - sum(targets.values())
    for key, _frac in sorted(((key, raw[key] - targets[key]) for key in counts), key=lambda x: (-x[1], x[0]))[:remainder]:
        targets[key] += 1
    return targets


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def counter_table(counter: Counter, denominator: int) -> str:
    lines = ["| Item | Count | Percent |", "|---|---:|---:|"]
    for key, value in counter.most_common():
        pct = value / denominator * 100 if denominator else 0
        lines.append(f"| {key} | {value} | {pct:.2f}% |")
    return "\n".join(lines)


def exact_duplicate_count(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], key: str) -> int:
    train_values = {normalize_text(row.get(key)) for row in train_rows}
    return sum(1 for row in test_rows if normalize_text(row.get(key)) in train_values)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def near_duplicate_questions(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], threshold: float = 0.90) -> tuple[int, float]:
    train_sets = [char_ngrams(row.get("instruction")) for row in train_rows]
    count = 0
    max_score = 0.0
    for test_row in test_rows:
        test_set = char_ngrams(test_row.get("instruction"))
        if not test_set:
            continue
        row_max = 0.0
        for train_set in train_sets:
            if not train_set:
                continue
            score = len(test_set & train_set) / len(test_set | train_set)
            if score > row_max:
                row_max = score
        max_score = max(max_score, row_max)
        if row_max >= threshold:
            count += 1
    return count, max_score


def main() -> None:
    global FINAL_JSONL, ENRICHED_JSONL, OUT_DIR, SEED, TEST_SIZE
    parser = argparse.ArgumentParser(description="Create a source-file-isolated holdout from base instruction records")
    parser.add_argument("--final", type=Path, required=True, help="Base instruction JSONL")
    parser.add_argument("--enriched", type=Path, required=True, help="Row-aligned provenance JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    args = parser.parse_args()
    FINAL_JSONL, ENRICHED_JSONL, OUT_DIR = args.final, args.enriched, args.out
    SEED, TEST_SIZE = args.seed, args.test_size
    if TEST_SIZE < 1:
        parser.error("--test-size must be positive")
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    final_rows_raw = load_jsonl(FINAL_JSONL)
    enriched_rows = load_jsonl(ENRICHED_JSONL)

    enriched_final: list[dict[str, Any]] = []
    for final_row in final_rows_raw:
        enriched_row = get_enriched_for_final_row(final_row, enriched_rows)
        enriched_final.append(enrich_final_row(final_row, enriched_row, split="unassigned"))

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in enriched_final:
        by_source[row["meta_info"]["source_file"]].append(row)

    # The main evaluation is evidence-grounded QA, so choose held-out samples
    # only from rows with literature evidence in the input field. To keep the
    # evaluation set exactly 200 rows and guarantee paper-level isolation, use
    # only sources that contribute one final sample.
    eligible = [
        row
        for row in enriched_final
        if row["meta_info"].get("input_mode") == TEST_INPUT_MODE
        and normalize_text(row.get("input"))
        and len(by_source[row["meta_info"]["source_file"]]) == 1
    ]
    cell_counts = Counter(row["meta_info"]["cell_type"] for row in eligible)
    targets = largest_remainder_targets(cell_counts, TEST_SIZE)

    selected: list[dict[str, Any]] = []
    for cell_type, target in sorted(targets.items()):
        pool = [row for row in eligible if row["meta_info"]["cell_type"] == cell_type]
        rng.shuffle(pool)
        if len(pool) < target:
            raise ValueError(f"Not enough eligible rows for {cell_type}: need {target}, have {len(pool)}")
        selected.extend(pool[:target])

    rng.shuffle(selected)
    selected_source_files = {row["meta_info"]["source_file"] for row in selected}

    test_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    for row in enriched_final:
        source_file = row["meta_info"]["source_file"]
        if source_file in selected_source_files:
            test_rows.append(enrich_final_row(row, get_enriched_for_final_row(row, enriched_rows), split="heldout_test"))
        else:
            train_rows.append(enrich_final_row(row, get_enriched_for_final_row(row, enriched_rows), split="train"))

    if len(test_rows) != TEST_SIZE:
        raise AssertionError(f"Expected {TEST_SIZE} test rows, got {len(test_rows)}")

    train_sources = {row["meta_info"]["source_file"] for row in train_rows}
    test_sources = {row["meta_info"]["source_file"] for row in test_rows}
    source_overlap = sorted(train_sources & test_sources)
    if source_overlap:
        raise AssertionError(f"Train/test source overlap detected: {source_overlap[:5]}")

    eval_rows = []
    for i, row in enumerate(test_rows, start=1):
        meta = row["meta_info"]
        eval_rows.append(
            {
                "id": f"pv_ecot_test_{i:03d}",
                "source_id": meta["source_file"],
                "source_file_hash": meta["source_file_hash"],
                "cell_type": meta["cell_type"],
                "task_type": meta["chain_type"],
                "answer_style": meta.get("answer_style"),
                "question": row.get("instruction", ""),
                "evidence": row.get("input", ""),
                "reference_answer": row.get("output", ""),
            }
        )

    train_path = OUT_DIR / "pv_ecot_train_without_holdout_sources.jsonl"
    test_path = OUT_DIR / "pv_ecot_holdout_200.jsonl"
    eval_path = OUT_DIR / "pv_ecot_holdout_200_eval_format.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)
    write_jsonl(eval_path, eval_rows)
    write_lines(OUT_DIR / "train_source_files.txt", sorted(train_sources))
    write_lines(OUT_DIR / "test_source_files.txt", sorted(test_sources))

    exact_instruction = exact_duplicate_count(train_rows, test_rows, "instruction")
    exact_input = exact_duplicate_count(train_rows, test_rows, "input")
    exact_output = exact_duplicate_count(train_rows, test_rows, "output")
    near_q_count, max_q_jaccard = near_duplicate_questions(train_rows, test_rows)

    report = {
        "input_file": str(FINAL_JSONL),
        "enriched_source_file": str(ENRICHED_JSONL),
        "seed": SEED,
        "strategy": "200-row held-out split from evidence-grounded literature_input rows; stratified by cell_type; train/test isolated by source_file.",
        "full_rows": len(enriched_final),
        "full_unique_source_files": len(by_source),
        "eligible_literature_input_single_source_rows": len(eligible),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_unique_source_files": len(train_sources),
        "test_unique_source_files": len(test_sources),
        "source_overlap_count": len(source_overlap),
        "target_test_cell_counts": targets,
        "actual_test_cell_counts": dict(Counter(row["meta_info"]["cell_type"] for row in test_rows)),
        "test_chain_type_counts": dict(Counter(row["meta_info"]["chain_type"] for row in test_rows)),
        "test_answer_style_counts": dict(Counter(row["meta_info"].get("answer_style") for row in test_rows)),
        "test_input_mode_counts": dict(Counter(row["meta_info"].get("input_mode") for row in test_rows)),
        "train_input_mode_counts": dict(Counter(row["meta_info"].get("input_mode") for row in train_rows)),
        "exact_duplicate_instruction_count": exact_instruction,
        "exact_duplicate_input_count": exact_input,
        "exact_duplicate_output_count": exact_output,
        "near_duplicate_question_count_jaccard_ge_0_90": near_q_count,
        "max_question_char_5gram_jaccard": round(max_q_jaccard, 4),
    }
    (OUT_DIR / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# PV-ECoT Held-Out Split Report",
        "",
        f"- Input file: `{FINAL_JSONL}`",
        f"- Enriched source file: `{ENRICHED_JSONL}`",
        f"- Random seed: `{SEED}`",
        f"- Test size: `{TEST_SIZE}`",
        f"- Test selection: evidence-grounded `{TEST_INPUT_MODE}` rows only",
        f"- Split key: `meta_info.source_file` recovered from the enriched source file",
        f"- Strategy: stratified by `cell_type`; selected source files are fully excluded from training",
        "",
        "## Counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Full rows | {len(enriched_final)} |",
        f"| Full unique source files | {len(by_source)} |",
        f"| Eligible evidence-grounded singleton rows | {len(eligible)} |",
        f"| Train rows | {len(train_rows)} |",
        f"| Held-out test rows | {len(test_rows)} |",
        f"| Train unique source files | {len(train_sources)} |",
        f"| Test unique source files | {len(test_sources)} |",
        f"| Train/test source overlap | {len(source_overlap)} |",
        "",
        "## Held-Out Cell Type Distribution",
        "",
        counter_table(Counter(row["meta_info"]["cell_type"] for row in test_rows), len(test_rows)),
        "",
        "## Held-Out Task Type Distribution",
        "",
        counter_table(Counter(row["meta_info"]["chain_type"] for row in test_rows), len(test_rows)),
        "",
        "## Held-Out Answer Style Distribution",
        "",
        counter_table(Counter(str(row["meta_info"].get("answer_style")) for row in test_rows), len(test_rows)),
        "",
        "## Leakage Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Train/test source_file intersection | {len(source_overlap)} |",
        f"| Exact duplicate instructions in train/test | {exact_instruction} |",
        f"| Exact duplicate inputs in train/test | {exact_input} |",
        f"| Exact duplicate outputs in train/test | {exact_output} |",
        f"| Near-duplicate questions, char 5-gram Jaccard >= 0.90 | {near_q_count} |",
        f"| Max question char 5-gram Jaccard | {max_q_jaccard:.4f} |",
        "",
        "## Output Files",
        "",
        f"- Training set: `{train_path}`",
        f"- Held-out test set: `{test_path}`",
        f"- Evaluation format: `{eval_path}`",
        f"- Train source list: `{OUT_DIR / 'train_source_files.txt'}`",
        f"- Test source list: `{OUT_DIR / 'test_source_files.txt'}`",
    ]
    (OUT_DIR / "split_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
