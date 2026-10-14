# PV-MechInstruct dataset

The dataset contains 10,682 photovoltaic instruction records. Each record provides a question or request, optional evidence context, a target answer, and metadata. The collection covers four task types—Forward, Backward, Mechanism, and Trade-off—constructed from evidence-linked mechanistic paths. Full paper texts and PDFs are not included.

## Files

| File | Records | Description |
| --- | ---: | --- |
| `pv_mechinstruct_10682.jsonl` | 10,682 | Complete instruction dataset: 3,481 base and 7,201 expanded records |
| `base_train_3281.jsonl` | 3,281 | Training side of the base-record split |
| `holdout_200.jsonl` | 200 | Test side of the base-record split |
| `base_split_report.json` | — | Split settings, counts, and duplicate checks |
| `expanded_source_index_7201.jsonl` | 7,201 | Source-file labels and content hashes for expanded records |
| `evaluation/holdout_200_eval_format.jsonl` | 200 | Internal evaluation questions, evidence, and reference answers |
| `evaluation/internal_200_aligned_outputs.jsonl` | 200 | Paired model answers and per-item scores |
| `evaluation/internal_200_rouge_report.json` | — | Internal evaluation score summary |
| `evaluation/internal_200_blind_judge_*` | 200 | Model-assisted blind comparison files and summary |
| `evaluation/dataset_200_*` | 200 | Separate dataset quality review files and summaries |
| `evaluation/supplementary_50_gold.jsonl` | 50 | Perovskite questions, evidence, and reference answers |
| `evaluation/supplementary_50_*` | 50 | Paired answers, blinded human preferences, and score summaries |

The main JSONL records have four top-level fields:

| Field | Meaning |
| --- | --- |
| `instruction` | Question or requested task |
| `input` | Supplied literature context; it may be empty for some expanded records |
| `output` | Target answer |
| `meta_info` | Source index, input mode, answer style, and merge origin |

Each JSONL line is one UTF-8 JSON object. The `meta_info.merge_source` value distinguishes the base and expanded portions of the final file. The separate expanded source index links expanded rows to source-file names using row order and content hashes. Source-file names refer to research documents and are not paths to papers included in this repository.

## Split interpretation

The 3,281/200 base-record split has no source-file overlap within the 3,481 base records. The full dataset also contains 7,201 expanded records, of which 411 correspond to 184 source files in the base holdout. Therefore, the base split must not be described as a source-isolated split of all 10,682 records. The `audit-overlap` command checks these counts and the expanded source-index alignment.

The 200-item internal evaluation set and the separate 200-item dataset quality-review sample serve different purposes. The supplementary 50-item set contains perovskite examples. In `supplementary_50_gold.jsonl`, five author email-address occurrences in evidence text are replaced by `[email redacted]`; scientific question and answer fields are unchanged.

## Use

From the repository root, inspect the data with any JSONL reader or use the commands documented in the main `README.md`. Keep generated outputs outside `data/` so the distributed files remain unchanged. `SHA256SUMS.txt` lists file hashes for integrity checks.
