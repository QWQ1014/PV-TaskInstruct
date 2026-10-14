import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def response_text(row: dict) -> str:
    for key in ["response", "generated_text", "prediction", "output", "answer", "text"]:
        if key in row:
            return str(row.get(key) or "").strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace failed/empty output rows with retried rows by id.")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--retry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    original = load_jsonl(args.original)
    retry_rows = {str(row["id"]): row for row in load_jsonl(args.retry)}
    replaced = 0
    merged = []
    for row in original:
        row_id = str(row.get("id"))
        if row_id in retry_rows and (not response_text(row) or row.get("error")):
            merged.append(retry_rows[row_id])
            replaced += 1
        else:
            merged.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8")
    print(json.dumps({"original": len(original), "retry": len(retry_rows), "replaced": replaced, "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
