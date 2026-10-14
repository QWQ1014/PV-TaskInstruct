# -*- coding: utf-8 -*-
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DIMENSIONS = [
    "question_clear",
    "context_supported",
    "mechanism_complete",
    "relation_direction_correct",
    "training_usable",
    "accept_for_dataset",
]


def norm(value):
    return (value or "").strip()


def load_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pass_rate(rows, field):
    vals = [norm(row.get(field)) for row in rows if norm(row.get(field))]
    if not vals:
        return {"n": 0, "yes": 0, "rate": None}
    yes = sum(1 for v in vals if v.upper() == "Y")
    return {"n": len(vals), "yes": yes, "rate": round(yes / len(vals) * 100, 2)}


def exact_agreement(rows_a, rows_b, field):
    pairs = []
    for a, b in zip(rows_a, rows_b):
        va, vb = norm(a.get(field)), norm(b.get(field))
        if va and vb:
            pairs.append((va, vb))
    if not pairs:
        return {"n": 0, "agreement": None}
    agree = sum(1 for a, b in pairs if a == b)
    return {"n": len(pairs), "agreement": round(agree / len(pairs) * 100, 2)}


def cohen_kappa(rows_a, rows_b, field):
    pairs = []
    labels = set()
    for a, b in zip(rows_a, rows_b):
        va, vb = norm(a.get(field)), norm(b.get(field))
        if va and vb:
            pairs.append((va, vb))
            labels.add(va)
            labels.add(vb)
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if pe == 1:
        return 1.0
    return round((po - pe) / (1 - pe), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer-a", default="human_review/pv_mechinstruct_human_review_reviewer_A.csv")
    parser.add_argument("--reviewer-b", default="human_review/pv_mechinstruct_human_review_reviewer_B.csv")
    parser.add_argument("--out-dir", default="human_review")
    args = parser.parse_args()

    rows_a = load_csv(args.reviewer_a)
    rows_b = load_csv(args.reviewer_b)
    if len(rows_a) != len(rows_b):
        raise ValueError(f"row count mismatch: A={len(rows_a)} B={len(rows_b)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    summary = {
        "n_rows": len(rows_a),
        "dimension_pass_rates_reviewer_a": {field: pass_rate(rows_a, field) for field in DIMENSIONS},
        "dimension_pass_rates_reviewer_b": {field: pass_rate(rows_b, field) for field in DIMENSIONS},
        "overall_quality_counts_reviewer_a": dict(Counter(norm(row.get("overall_quality")) for row in rows_a if norm(row.get("overall_quality")))),
        "overall_quality_counts_reviewer_b": dict(Counter(norm(row.get("overall_quality")) for row in rows_b if norm(row.get("overall_quality")))),
        "main_error_type_counts_reviewer_a": dict(Counter(norm(row.get("main_error_type")) for row in rows_a if norm(row.get("main_error_type")))),
        "main_error_type_counts_reviewer_b": dict(Counter(norm(row.get("main_error_type")) for row in rows_b if norm(row.get("main_error_type")))),
        "exact_agreement": {field: exact_agreement(rows_a, rows_b, field) for field in DIMENSIONS + ["overall_quality"]},
        "cohen_kappa": {field: cohen_kappa(rows_a, rows_b, field) for field in DIMENSIONS + ["overall_quality"]},
    }

    json_path = out_dir / "pv_mechinstruct_human_review_summary.json"
    md_path = out_dir / "pv_mechinstruct_human_review_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# PV-MechInstruct 人工复核结果汇总",
        "",
        f"- 样本数：{summary['n_rows']}",
        "",
        "## Reviewer A 维度通过率",
    ]
    for field, item in summary["dimension_pass_rates_reviewer_a"].items():
        lines.append(f"- {field}: {item['yes']}/{item['n']} ({item['rate']}%)")
    lines += ["", "## Reviewer B 维度通过率"]
    for field, item in summary["dimension_pass_rates_reviewer_b"].items():
        lines.append(f"- {field}: {item['yes']}/{item['n']} ({item['rate']}%)")
    lines += ["", "## 一致性"]
    for field in DIMENSIONS + ["overall_quality"]:
        agree = summary["exact_agreement"][field]
        kappa = summary["cohen_kappa"][field]
        lines.append(f"- {field}: exact agreement {agree['agreement']}% (n={agree['n']}), Cohen's kappa={kappa}")
    lines += [
        "",
        "## A/B/C 分布",
        f"- Reviewer A: {summary['overall_quality_counts_reviewer_a']}",
        f"- Reviewer B: {summary['overall_quality_counts_reviewer_b']}",
        "",
        "## 错误类型分布",
        f"- Reviewer A: {summary['main_error_type_counts_reviewer_a']}",
        f"- Reviewer B: {summary['main_error_type_counts_reviewer_b']}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"json={json_path}")
    print(f"markdown={md_path}")


if __name__ == "__main__":
    main()
