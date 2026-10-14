"""Audit source-file overlap between the included data components and test set.

Run from the repository root: python run.py audit-overlap
The expanded records are linked to their source files through the accompanying
provenance file in the same row order. No model call or third-party package is
required.
"""

import json
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main():
    full = list(read_jsonl(DATA / "pv_mechinstruct_10682.jsonl"))
    base_train = list(read_jsonl(DATA / "base_train_3281.jsonl"))
    test = list(read_jsonl(DATA / "holdout_200.jsonl"))
    expanded = list(read_jsonl(DATA / "expanded_source_index_7201.jsonl"))

    assert len(full) == 10682
    assert len(base_train) == 3281
    assert len(test) == 200
    assert len(expanded) == 7201

    test_sources = {row["meta_info"]["source_file"] for row in test}
    base_sources = {row["meta_info"]["source_file"] for row in base_train}
    overlap_rows = [row for row in expanded if row["source_file"] in test_sources]
    expanded_sources = {row["source_file"] for row in expanded}

    # This check binds the provenance file to the expanded portion of the
    # merged data and prevents an incorrect row-order join.
    merged_expanded = [
        row for row in full
        if row["meta_info"]["merge_source"]
        == "expert_cot_dataset.multi_rewrite_full_mixed_concurrency30"
    ]
    assert len(merged_expanded) == len(expanded)
    assert all(
        a["meta_info"]["source_index"] == i
        and digest(a["instruction"]) == b["instruction_sha256"]
        for i, (a, b) in enumerate(zip(merged_expanded, expanded), start=1)
    ), "Expanded provenance does not align with merged records"

    print(json.dumps({
        "full_rows": len(full),
        "base_train_rows": len(base_train),
        "expanded_rows": len(expanded),
        "holdout_rows": len(test),
        "base_train_test_source_overlap": len(base_sources & test_sources),
        "expanded_train_test_source_overlap_rows": len(overlap_rows),
        "expanded_train_test_source_overlap_sources": len(expanded_sources & test_sources),
        "hypothetical_source_isolated_training_rows":
            len(base_train) + len(expanded) - len(overlap_rows),
        "expanded_overlap_by_cell_type": dict(Counter(
            row.get("meta_info", {}).get("cell_type", "unknown")
            for row in overlap_rows
        )),
    }, indent=2))


if __name__ == "__main__":
    main()
