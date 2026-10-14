# Instruction rewrite

`run_multi_rewrite_batches.py` launches `build_pv_attribution_dataset.py` repeatedly. Keep both scripts together. Pass `--source`, `--target`, `--log`, and `--summary` to the batch script to choose files; run `python run.py rewrite --help` at the repository root for all options. This stage uses a configured model API and may incur charges. The final merged instruction dataset is in `data/`.
