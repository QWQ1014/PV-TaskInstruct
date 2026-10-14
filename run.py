"""Entry point for the PV-TaskInstruct research workflow.

Run ``python run.py`` to list stages, or ``python run.py split --help`` for
the arguments of a data-only stage. Model-backed stages require credentials
and source files; they never run from a bare ``python run.py`` invocation.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGES = {
    "extract": ("Extraction_project/run_batch_extractor.py", "Read paper Markdown and extract relations", True),
    "merge": ("Extraction_project/merge_doc/merge_results.py", "Merge extracted relation files", True),
    "refine": ("Extraction_project/merge_doc/one_shot_refine_multiapi_resume_failed_queue.py", "Refine relation graphs with a model", True),
    "generate-base": ("Extraction_project/merge_doc/expert_cot_generator_one_call_prompt_celltype_v6_expert_reasoning.py", "Generate base instruction records", True),
    "generate-multi": ("Extraction_project/merge_doc/expert_cot_generator_multi_samples_v8.py", "Generate additional instruction records", True),
    "rewrite": ("workflows/01_generation/run_multi_rewrite_batches.py", "Rewrite generated records in batches", False),
    "stats": ("workflows/02_dataset/compute_pv_ecot_stats.py", "Compute dataset statistics", False),
    "split": ("workflows/02_dataset/create_pv_ecot_holdout_split.py", "Create source-isolated base-data holdout", False),
    "audit-overlap": ("workflows/02_dataset/audit_source_overlap.py", "Audit source overlap in the included data", False),
    "convert": ("workflows/02_dataset/convert_holdout_train_to_llamafactory.py", "Convert training JSONL to Alpaca JSON", False),
    "quality": ("workflows/03_quality/run_pv_mechinstruct_quality_audit.py", "Sample and audit instruction quality", False),
    "review-sheets": ("workflows/03_quality/prepare_human_review_package.py", "Prepare human-review sheets", False),
    "review-summary": ("workflows/03_quality/summarize_human_review_results.py", "Summarize completed review sheets", False),
    "prompts": ("workflows/04_internal_eval/create_holdout_simple_qa_prompts.py", "Create internal evaluation prompts", False),
    "infer": ("workflows/04_internal_eval/run_openai_compatible_batch_inference_checked.py", "Run model inference", False),
    "retry-merge": ("workflows/04_internal_eval/merge_retried_outputs.py", "Merge retried inference outputs", False),
    "score": ("workflows/04_internal_eval/score_experiment1_outputs.py", "Score internal outputs with ROUGE-L", False),
    "blind-set": ("workflows/04_internal_eval/create_full_blind_judge_set.py", "Prepare blinded A/B comparisons", False),
    "gold-blind": ("workflows/05_supplementary_50/create_human_gold_50_blind_review.py", "Prepare 50-item supplementary blind review", False),
    "gold-score": ("workflows/05_supplementary_50/evaluate_human_gold_50_outputs.py", "Score 50-item supplementary outputs", False),
}


def doctor() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repository: {ROOT}")
    print(f"Local config: {'present' if (ROOT / 'Extraction_project/config.py').exists() else 'missing (copy config.example.py)'}")
    print(f"PV_API_KEY: {'set' if os.getenv('PV_API_KEY') else 'missing'}")
    print(f"PV_MODEL_NAME: {'set' if os.getenv('PV_MODEL_NAME') else 'missing'}")
    for name in ("aiofiles", "openai", "pydantic"):
        print(f"{name}: {'installed' if importlib.util.find_spec(name) else 'missing'}")
    return 0


def usage() -> None:
    print("PV-TaskInstruct workflow\n")
    print("Usage: python run.py <stage> [stage arguments]\n")
    print("Stages:")
    for name, (_, description, _) in STAGES.items():
        print(f"  {name:<17} {description}")
    print("  doctor            Check the local setup without making an API call")
    print("\nRun 'python run.py split --help' for an argument example.")
    print("The first five original pipeline stages require '--run' after reviewing their input paths and settings.")


def main() -> int:
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        usage()
        return 0
    stage = sys.argv[1]
    if stage == "doctor":
        return doctor()
    if stage not in STAGES:
        print(f"Unknown stage: {stage}", file=sys.stderr)
        usage()
        return 2
    relative, description, needs_run_flag = STAGES[stage]
    args = sys.argv[2:]
    if needs_run_flag:
        if args != ["--run"]:
            print(f"{stage}: {description}")
            print("Review its input paths and model settings, then run this stage with '--run'.")
            return 0 if args in ([], ["--help"]) else 2
        args = []
    script = ROOT / relative
    if not script.is_file():
        print(f"Missing stage script: {script}", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, str(script), *args], cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
