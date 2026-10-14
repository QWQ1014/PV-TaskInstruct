# PV-TaskInstruct

PV-TaskInstruct constructs photovoltaic research instruction data from scientific literature. It organizes evidence about material or process interventions, structural changes, physical mechanisms, device metrics, and performance into directed mechanistic paths. Those paths support four question types: Forward, Backward, Mechanism, and Trade-off.

This repository contains the implementation, the **PV-MechInstruct** dataset, and evaluation materials. The dataset has **10,682 instruction records** linked to **3,437 source-file names**. Complete source papers and model checkpoints are not included.

## Repository contents

| Path | Contents |
| --- | --- |
| `data/` | Instruction dataset, provenance index, base-data split, and evaluation files; see `data/README.md` |
| `Extraction_project/` | Literature relation extraction, refinement, and instruction generation |
| `workflows/01_generation/` | Batch rewriting of generated instructions |
| `workflows/02_dataset/` | Dataset statistics, splitting, format conversion, and source-overlap audit |
| `workflows/03_quality/` | Quality audit and human-review preparation |
| `workflows/04_internal_eval/` | Internal evaluation prompts, inference, scoring, and blind comparison |
| `workflows/05_supplementary_50/` | Supplementary 50-item perovskite evaluation |
| `run.py` | Command entry point for the workflow stages |
| `SHA256SUMS.txt` | SHA-256 hashes of packaged files |

## Getting started

Use Python 3.10 or newer. From the repository root:

```powershell
python -m pip install -r requirements.txt
python run.py --help
python run.py doctor
```

Model-backed stages require an OpenAI-compatible service. Copy `Extraction_project/config.example.py` to `Extraction_project/config.py` and set `PV_API_KEY`, `PV_API_BASE_URL`, `PV_MODEL_NAME`, and `PV_SOURCE_DIRECTORY` in your environment. The local `config.py` is ignored by Git. See `Extraction_project/README.md` for the extraction and generation stages.

The included dataset can be used without calling a model API. For example:

```powershell
python run.py convert --source data/base_train_3281.jsonl --target outputs/base_train_alpaca.json
python run.py prompts --eval data/evaluation/holdout_200_eval_format.jsonl --out outputs/eval_prompts.jsonl
python run.py audit-overlap
```

The commands write derived files under `outputs/`, which Git ignores. Run `python run.py <stage> --help` to see a stage's arguments.

## Dataset and evaluation scope

`data/pv_mechinstruct_10682.jsonl` is the complete instruction collection: 3,481 base records and 7,201 expanded records. The provided `base_train_3281.jsonl` and `holdout_200.jsonl` form a source-file-isolated split of the **base records only**. The expanded records are not part of that isolation guarantee. In particular, 411 expanded records come from 184 source files present in the base holdout. Use `python run.py audit-overlap` to inspect the relationship between the files.

The evaluation directory contains materials for a 200-item internal comparison, a separate 200-item dataset quality review, and a 50-item perovskite comparison. Their fields and purposes are described in `data/README.md`. The 200-item internal and 50-item supplementary evaluations use different scoring implementations; compare model outputs within each evaluation set.
