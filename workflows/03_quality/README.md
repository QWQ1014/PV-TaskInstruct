# Quality and human review

Run the automated audit with `python run.py quality --final <base.jsonl> --enriched <provenance.jsonl> --out <directory>`. Its sampled CSV can be turned into two reviewer sheets with `python run.py review-sheets --audit-csv <sample.csv> --out <directory>`. After reviewers fill their sheets, run `python run.py review-summary --reviewer-a <A.csv> --reviewer-b <B.csv> --out-dir <directory>`. Use `--help` on each stage for the complete options. The reviewer decisions are human inputs and are not created by this package.
