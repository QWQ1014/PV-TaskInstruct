import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


FINAL_PATH = Path("data/base_final.jsonl")
ENRICHED_PATH = Path("data/base_enriched.jsonl")
OUT_DIR = Path("outputs/quality_audit")
SAMPLE_SIZE = 200
SEED = 20260617


QUESTION_TERMS = ("为什么", "为何", "如何", "什么", "哪些", "是否", "能否", "影响", "原因", "?")
CAUSAL_TERMS = (
    "因为",
    "由于",
    "通过",
    "导致",
    "从而",
    "因此",
    "进而",
    "归因",
    "核心在于",
    "关键在于",
    "抑制",
    "促进",
    "增强",
    "降低",
    "提升",
    "改善",
    "减少",
)
UNUSABLE_TERMS = ("无法判断", "无法确定", "没有提供", "不能确定", "不清楚", "不知道")


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def text_len(text):
    return len((text or "").strip())


def count_layers(fact_skeleton):
    return len(set(re.findall(r"\[(L[1-5]_[^\]]+)\]", fact_skeleton or "")))


def has_question_shape(question):
    q = (question or "").strip()
    return 10 <= len(q) <= 180 and any(term in q for term in QUESTION_TERMS)


def has_causal_language(answer):
    answer = answer or ""
    return sum(1 for term in CAUSAL_TERMS if term in answer) >= 2


def is_training_usable(question, answer):
    if not question or not answer:
        return False
    if any(term in answer for term in UNUSABLE_TERMS):
        return False
    # Avoid answers that are too short to supervise mechanism attribution or too long/noisy.
    return 40 <= len(answer.strip()) <= 1200


def context_text(final_obj, enriched_obj):
    provenance = enriched_obj.get("provenance", {}) or {}
    parts = [
        final_obj.get("input") or "",
        provenance.get("material_system") or "",
        provenance.get("target_problem") or "",
        provenance.get("research_focus") or "",
        provenance.get("fact_skeleton") or "",
    ]
    return "\n".join(p for p in parts if p)


def audit_one(final_obj, enriched_obj):
    meta = enriched_obj.get("meta_info", {}) or {}
    provenance = enriched_obj.get("provenance", {}) or {}
    question = final_obj.get("instruction") or enriched_obj.get("instruction") or ""
    answer = final_obj.get("output") or enriched_obj.get("output") or ""
    ctx = context_text(final_obj, enriched_obj)
    fact_skeleton = provenance.get("fact_skeleton") or ""
    layer_count = count_layers(fact_skeleton)

    question_clear = has_question_shape(question)
    context_traceable = text_len(ctx) >= 120 and "->" in fact_skeleton
    mechanism_complete = meta.get("chain_quality") in {"A", "B"} and layer_count >= 3 and has_causal_language(answer)
    context_consistent = meta.get("grounded_quality") in {"A", "B"} and context_traceable
    causal_direction = "->" in fact_skeleton and has_causal_language(answer)
    training_usable = is_training_usable(question, answer)

    checks = {
        "question_clarity": question_clear,
        "context_traceability": context_traceable,
        "context_consistency": context_consistent,
        "mechanism_completeness": mechanism_complete,
        "causal_direction": causal_direction,
        "training_usability": training_usable,
    }
    fail_reasons = [k for k, v in checks.items() if not v]
    return checks, {
        "overall_pass": all(checks.values()),
        "fail_reasons": ";".join(fail_reasons),
        "layer_count": layer_count,
        "chain_quality": meta.get("chain_quality", ""),
        "grounded_quality": meta.get("grounded_quality", ""),
        "mechanism_high_confidence": meta.get("chain_quality") == "A",
        "context_high_confidence": meta.get("grounded_quality") == "A",
    }


def allocate_stratified(rows, n, seed):
    strata = defaultdict(list)
    for row in rows:
        key = (row["cell_type"], row["task_type"])
        strata[key].append(row)

    total = len(rows)
    allocations = {}
    remainders = []
    for key, items in strata.items():
        exact = len(items) / total * n
        base = math.floor(exact)
        allocations[key] = base
        remainders.append((exact - base, key))

    remaining = n - sum(allocations.values())
    for _, key in sorted(remainders, reverse=True)[:remaining]:
        allocations[key] += 1

    # Ensure rare non-empty strata are represented, then compensate from largest strata.
    for key, items in strata.items():
        if allocations[key] == 0 and items:
            allocations[key] = 1
    while sum(allocations.values()) > n:
        key = max(allocations, key=lambda k: allocations[k])
        allocations[key] -= 1

    rng = random.Random(seed)
    sample = []
    for key, items in sorted(strata.items()):
        take = min(allocations[key], len(items))
        sample.extend(rng.sample(items, take))
    rng.shuffle(sample)
    return sample


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def main():
    global FINAL_PATH, ENRICHED_PATH, OUT_DIR, SAMPLE_SIZE, SEED
    parser = argparse.ArgumentParser(description="Sample and audit instruction quality")
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    FINAL_PATH, ENRICHED_PATH, OUT_DIR = args.final, args.enriched, args.out
    SAMPLE_SIZE, SEED = args.sample_size, args.seed
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = read_jsonl(FINAL_PATH)
    enriched_rows = read_jsonl(ENRICHED_PATH)

    enriched_by_position = {idx: row for idx, row in enumerate(enriched_rows, start=1)}
    audit_pool = []
    for idx, final_obj in enumerate(final_rows, start=1):
        source_index = final_obj.get("meta_info", {}).get("source_index")
        enriched_obj = enriched_by_position.get(source_index)
        if not enriched_obj:
            continue
        meta = enriched_obj.get("meta_info", {})
        audit_pool.append(
            {
                "final_index": idx,
                "source_index": source_index,
                "final": final_obj,
                "enriched": enriched_obj,
                "cell_type": meta.get("cell_type", "unknown"),
                "task_type": meta.get("chain_type", "unknown"),
            }
        )

    sample = allocate_stratified(audit_pool, SAMPLE_SIZE, SEED)
    audited = []
    for i, row in enumerate(sample, start=1):
        checks, extra = audit_one(row["final"], row["enriched"])
        meta = row["enriched"].get("meta_info", {})
        provenance = row["enriched"].get("provenance", {})
        final_obj = row["final"]
        enriched_obj = row["enriched"]
        audited.append(
            {
                "audit_id": f"pvmi_audit_{i:03d}",
                "final_index": row["final_index"],
                "source_index": row["source_index"],
                "sample_uid": enriched_obj.get("sample_uid", ""),
                "source_file": enriched_obj.get("source_file", ""),
                "cell_type": meta.get("cell_type", ""),
                "task_type": meta.get("chain_type", ""),
                "answer_style": final_obj.get("meta_info", {}).get("answer_style", ""),
                "question": final_obj.get("instruction", ""),
                "context_excerpt": (context_text(final_obj, enriched_obj) or "")[:1200],
                "answer": final_obj.get("output", ""),
                "fact_skeleton_excerpt": (provenance.get("fact_skeleton") or "")[:1200],
                **checks,
                **extra,
            }
        )

    jsonl_path = OUT_DIR / "pv_mechinstruct_quality_audit_200.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in audited:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = OUT_DIR / "pv_mechinstruct_quality_audit_200.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audited[0].keys()))
        writer.writeheader()
        writer.writerows(audited)

    check_names = [
        "question_clarity",
        "context_traceability",
        "context_consistency",
        "mechanism_completeness",
        "causal_direction",
        "training_usability",
        "overall_pass",
        "mechanism_high_confidence",
        "context_high_confidence",
    ]
    summary = {
        "audit_pool_size": len(audit_pool),
        "sample_size": len(audited),
        "sample_seed": SEED,
        "stratification": "cell_type x task_type",
        "dimension_pass_counts": {name: sum(1 for row in audited if row[name]) for name in check_names},
        "dimension_pass_rates": {name: pct(sum(1 for row in audited if row[name]), len(audited)) for name in check_names},
        "cell_type_counts": dict(Counter(row["cell_type"] for row in audited)),
        "task_type_counts": dict(Counter(row["task_type"] for row in audited)),
        "chain_quality_counts": dict(Counter(row["chain_quality"] for row in audited)),
        "grounded_quality_counts": dict(Counter(row["grounded_quality"] for row in audited)),
        "fail_reason_counts": dict(Counter(reason for row in audited for reason in row["fail_reasons"].split(";") if reason)),
    }

    summary_path = OUT_DIR / "pv_mechinstruct_quality_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = OUT_DIR / "pv_mechinstruct_quality_audit_report.md"
    lines = [
        "# PV-MechInstruct Sample-Level Quality Audit",
        "",
        f"- Audit pool: {summary['audit_pool_size']} final samples mapped to enriched provenance.",
        f"- Sample size: {summary['sample_size']} stratified by `cell_type x task_type`.",
        f"- Random seed: {SEED}.",
        "",
        "## Dimension Pass Rates",
        "",
        "| Dimension | Pass count | Pass rate |",
        "|---|---:|---:|",
    ]
    labels = {
        "question_clarity": "Question clarity",
        "context_traceability": "Context traceability",
        "context_consistency": "Context consistency",
        "mechanism_completeness": "Mechanism completeness",
        "causal_direction": "Causal direction",
        "training_usability": "Training usability",
        "overall_pass": "Overall pass",
        "mechanism_high_confidence": "Mechanism high-confidence (A-level)",
        "context_high_confidence": "Context high-confidence (A-level)",
    }
    for name in check_names:
        lines.append(
            f"| {labels[name]} | {summary['dimension_pass_counts'][name]} / {len(audited)} | {summary['dimension_pass_rates'][name]:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Failure Reasons",
            "",
            "| Failure reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(summary["fail_reason_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Sample Distribution",
            "",
            "### By cell type",
            "",
            "| Cell type | Count |",
            "|---|---:|",
        ]
    )
    for key, count in sorted(summary["cell_type_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "### By task type", "", "| Task type | Count |", "|---|---:|"])
    for key, count in sorted(summary["task_type_counts"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"| {key} | {count} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote:\n- {jsonl_path}\n- {csv_path}\n- {summary_path}\n- {md_path}")


if __name__ == "__main__":
    main()
