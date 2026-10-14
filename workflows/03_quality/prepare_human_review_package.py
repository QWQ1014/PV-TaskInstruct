# -*- coding: utf-8 -*-
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "quality_audit" / "pv_mechinstruct_quality_audit_200.csv"
OUT_DIR = ROOT / "outputs" / "human_review"
ROWS_JSON = OUT_DIR / "pv_mechinstruct_human_review_rows.json"
PROTOCOL = OUT_DIR / "pv_mechinstruct_human_review_protocol.md"
REVIEWER_A_CSV = OUT_DIR / "pv_mechinstruct_human_review_reviewer_A.csv"
REVIEWER_B_CSV = OUT_DIR / "pv_mechinstruct_human_review_reviewer_B.csv"


REVIEW_COLUMNS = [
    "review_id",
    "audit_id",
    "source_file",
    "cell_type",
    "task_type",
    "question",
    "context_excerpt",
    "answer",
    "question_clear",
    "context_supported",
    "mechanism_complete",
    "relation_direction_correct",
    "training_usable",
    "overall_quality",
    "accept_for_dataset",
    "main_error_type",
    "reviewer_comment",
]


def clean_text(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def main():
    global SRC, OUT_DIR, ROWS_JSON, PROTOCOL, REVIEWER_A_CSV, REVIEWER_B_CSV
    parser = argparse.ArgumentParser(description="Prepare two reviewer sheets from the quality audit CSV")
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    SRC, OUT_DIR = args.audit_csv, args.out
    ROWS_JSON = OUT_DIR / "pv_mechinstruct_human_review_rows.json"
    PROTOCOL = OUT_DIR / "pv_mechinstruct_human_review_protocol.md"
    REVIEWER_A_CSV = OUT_DIR / "pv_mechinstruct_human_review_reviewer_A.csv"
    REVIEWER_B_CSV = OUT_DIR / "pv_mechinstruct_human_review_reviewer_B.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SRC.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    review_rows = []
    for i, row in enumerate(rows, 1):
        review_rows.append(
            {
                "review_id": f"PVMI-HR-{i:03d}",
                "audit_id": row["audit_id"],
                "sample_uid": row["sample_uid"],
                "source_file": row["source_file"],
                "cell_type": row["cell_type"],
                "task_type": row["task_type"],
                "question": clean_text(row["question"]),
                "context_excerpt": clean_text(row["context_excerpt"]),
                "answer": clean_text(row["answer"]),
                "fact_skeleton_excerpt": clean_text(row.get("fact_skeleton_excerpt", "")),
            }
        )

    ROWS_JSON.write_text(json.dumps(review_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for path, reviewer in [(REVIEWER_A_CSV, "A"), (REVIEWER_B_CSV, "B")]:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
            writer.writeheader()
            for row in review_rows:
                out = {col: "" for col in REVIEW_COLUMNS}
                for col in [
                    "review_id",
                    "audit_id",
                    "source_file",
                    "cell_type",
                    "task_type",
                    "question",
                    "context_excerpt",
                    "answer",
                ]:
                    out[col] = row[col]
                writer.writerow(out)

    PROTOCOL.write_text(
        """# PV-MechInstruct 人工质量复核说明

## 复核目的

本次人工复核用于评价 PV-MechInstruct 自动生成样本本身是否可靠，而不是比较基础模型和微调模型输出。评审者需要根据给定论文上下文，判断问题和参考答案是否清楚、可信、完整，并能否作为指令微调样本使用。

## 样本来源

复核样本为从最终 3481 条 PV-MechInstruct 样本中按 `cell_type × task_type` 分层抽取的 200 条样本。该 200 条用于数据质量评估；论文中的 200 条 source-level held-out 测试集用于下游模型效用验证，二者服务于不同目的。

## 推荐评审流程

1. 阅读 `question`，判断问题是否清楚、是否能够从上下文中回答。
2. 阅读 `context_excerpt`，确认其中是否提供了支持答案的关键信息。
3. 阅读 `answer`，判断答案是否贴合上下文、是否包含必要机制环节。
4. 填写 5 个维度评分、整体质量等级、是否建议纳入数据集和错误类型。
5. 对存在问题的样本，在 `reviewer_comment` 中用一句话说明原因。

## 评分字段

- `question_clear`：问题是否清楚。填 `Y` 或 `N`。
- `context_supported`：答案是否能被给定论文上下文支持。填 `Y` 或 `N`。
- `mechanism_complete`：答案是否包含机制解释所需的关键中间环节。填 `Y` 或 `N`。
- `relation_direction_correct`：干预、机制、参数、性能之间的方向是否合理。填 `Y` 或 `N`。
- `training_usable`：字段完整、答案形式正常，是否适合作为监督微调样本。填 `Y` 或 `N`。
- `overall_quality`：整体等级。填 `A`、`B` 或 `C`。
  - `A`：可直接使用，机制清楚且上下文支持充分。
  - `B`：基本可用，但存在表达不够清楚、机制略简或边界条件不足。
  - `C`：不建议使用，存在明显上下文不支持、机制错误或问题不可回答。
- `accept_for_dataset`：是否建议保留。填 `Y` 或 `N`。
- `main_error_type`：主要问题类型，可填 `None`、`Question unclear`、`Context unsupported`、`Mechanism incomplete`、`Wrong direction`、`Over-generalization`、`Parameter misread`、`Other`。

## 论文中建议报告的统计量

- 两名评审者的整体可用率。
- 各维度通过率。
- A/B/C 等级分布。
- 两名评审者一致率，必要时补充 Cohen's kappa。
- 主要错误类型分布。
""",
        encoding="utf-8",
    )

    print(f"rows={len(review_rows)}")
    print(f"json={ROWS_JSON}")
    print(f"protocol={PROTOCOL}")
    print(f"csv_a={REVIEWER_A_CSV}")
    print(f"csv_b={REVIEWER_B_CSV}")


if __name__ == "__main__":
    main()
