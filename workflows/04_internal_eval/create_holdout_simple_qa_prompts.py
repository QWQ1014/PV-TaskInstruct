import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = "你是光伏材料与器件领域的科研助手。请只根据给定证据回答问题，不要编造。"


def build_user_prompt(row: dict) -> str:
    question = str(row.get("question") or "").strip()
    evidence = str(row.get("evidence") or "").strip()
    return (
        "/no_think\n\n"
        f"问题：{question}\n\n"
        f"证据：\n{evidence}\n\n"
        "请基于上述证据回答问题。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create evidence-conditioned prompts for the internal evaluation")
    parser.add_argument("--eval", type=Path, required=True, help="Holdout evaluation JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Prompt JSONL")
    args = parser.parse_args()
    rows = []
    with args.eval.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            user_prompt = build_user_prompt(row)
            rows.append(
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "source_file_hash": row["source_file_hash"],
                    "cell_type": row["cell_type"],
                    "task_type": row["task_type"],
                    "answer_style": row["answer_style"],
                    "question": row["question"],
                    "evidence": row["evidence"],
                    "system": SYSTEM_PROMPT,
                    "prompt": user_prompt,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"prompts": str(args.out), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
