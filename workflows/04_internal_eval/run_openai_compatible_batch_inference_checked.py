import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SYSTEM_FALLBACK = "你是光伏材料与器件领域的科研助手。请直接回答问题，不要输出分析过程。"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_messages(row: dict):
    messages = row.get("messages")
    if messages:
        return messages
    system = row.get("system") or SYSTEM_FALLBACK
    prompt = row.get("prompt") or row.get("question") or ""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def chat_completion(api_base, model, messages, temperature, max_tokens, timeout):
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"].get("content", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print_first_prompt", action="store_true")
    args = parser.parse_args()

    prompt_path = Path(args.prompts)
    out_path = Path(args.out)
    rows = list(read_jsonl(prompt_path))
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        raise SystemExit("No prompts found.")

    if args.print_first_prompt:
        first_messages = build_messages(rows[0])
        print("========== FIRST REQUEST MESSAGES ==========")
        print(json.dumps(first_messages, ensure_ascii=False, indent=2))
        print("============================================")

    with out_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, 1):
            messages = build_messages(row)
            response = ""
            last_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    response = chat_completion(
                        args.api_base,
                        args.model,
                        messages,
                        args.temperature,
                        args.max_tokens,
                        args.timeout,
                    )
                    last_error = None
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                    last_error = exc
                    wait = min(2 ** attempt, 10)
                    print(f"[warn] {row.get('id')} attempt {attempt}/{args.retries} failed: {exc}; sleep {wait}s")
                    time.sleep(wait)

            if last_error is not None:
                response = f"ERROR: {last_error}"

            out_row = {
                "id": row.get("id"),
                "response": response,
                "cell_type": row.get("cell_type"),
                "task_type": row.get("task_type"),
                "source_id": row.get("source_id"),
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{idx}/{len(rows)}] {row.get('id')}")
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
