# Internal 200-item evaluation

`prompts` converts the holdout evaluation JSONL to model prompts. `infer` calls an OpenAI-compatible endpoint and writes predictions. `retry-merge` combines retried predictions. `score` aligns base and tuned predictions and computes ROUGE-L. `blind-set` produces blinded comparison material. Each command accepts file paths; see `python run.py <stage> --help`.

`summarize_full_codex_blind_judgments.py` summarizes the recorded 200-item blind comparison decisions. It uses the included evaluation layout and is not a general model judge. The inference stage needs API access; prompt creation and score calculation do not.
