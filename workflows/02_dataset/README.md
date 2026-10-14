# Dataset preparation

- `compute_pv_ecot_stats.py`: descriptive statistics from instruction and relation files; specify input paths with its arguments.
- `create_pv_ecot_holdout_split.py`: base instruction JSONL plus row-aligned enriched provenance JSONL to source-level train/holdout files and an evaluation-format JSONL. Its original default holdout size is 200.
- `convert_holdout_train_to_llamafactory.py`: train JSONL to Alpaca JSON for LLaMA-Factory.
- `audit_source_overlap.py`: checks the relationship between the included base split, expanded-record source index, and final merged dataset.

Run `python run.py split --help`, `python run.py convert --help`, or `python run.py stats --help` from the repository root. No model API is needed. The split does not check an additional expanded training file for source overlap.

The final merged dataset and its provenance index can be checked with `python run.py audit-overlap`. The provided 3,281/200 split is isolated at the base-record level; the audit documents expanded-record source overlap separately.
