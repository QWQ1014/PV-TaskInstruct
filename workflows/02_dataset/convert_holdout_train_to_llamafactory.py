import argparse
import json
from pathlib import Path


def build_output(item: dict) -> str:
    reasoning_text = str(item.get("reasoning_text") or "").strip()
    answer = str(item.get("output") or "").strip()
    if reasoning_text:
        return f"<think>\n{reasoning_text}\n</think>\n\n{answer}"
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert training JSONL to LLaMA-Factory Alpaca JSON")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    data = []
    skipped = 0
    with args.source.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            instruction = str(item.get("instruction") or "").strip()
            output = build_output(item)
            if not instruction or not output:
                skipped += 1
                continue
            data.append(
                {
                    "instruction": instruction,
                    "input": str(item.get("input") or "").strip(),
                    "output": output,
                }
            )

    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source": str(args.source), "target": str(args.target), "rows": len(data), "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
