# Literature extraction and instruction generation

This directory contains the Python packages used for literature extraction and instruction generation: `prompt_schema`, `llm_api`, and `api_all`.

Order of operation: `run_batch_extractor.py` reads paper Markdown; `merge_doc/merge_results.py` merges extracted relations; `merge_doc/one_shot_refine_multiapi_resume_failed_queue.py` refines them; the `v6` one-call and `v8` multi-sample generators in `merge_doc/` produce instruction records. See `python run.py` at the repository root for entry points.

Copy `config.example.py` to the ignored `config.py` and set the `PV_*` environment variables. The extractor's source directory defaults to `Extraction_project/data/source_papers`; extracted results default to `Extraction_project/data/extractor_results`. Merge, refine, and generation paths can be changed with the `PV_MERGE_INPUT_DIR`, `PV_MERGED_RESULTS_DIR`, `PV_REFINED_RESULTS_DIR`, `PV_BASE_INSTRUCTIONS_JSONL`, and `PV_MULTI_INSTRUCTIONS_JSONL` variables. Check each script's configuration block before calling a model. These steps need source papers and API access.
