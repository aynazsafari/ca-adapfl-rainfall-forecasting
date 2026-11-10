# Completed-result audit readiness

The standalone audit is ready for the completed `revision_v2` run. Syntax compilation, CLI help and 12 deterministic synthetic checks have passed. No actual evaluation input, truth, prediction or score artifact has been opened by this task.

After root confirms that `runs/revision_v2/evaluation/complete.json` exists, run from the workspace:

```powershell
work/env/Scripts/python.exe -X utf8 work/real_run_qa/audit_completed_run.py --run outputs/03_benchmark/caadapfl_benchmark/runs/revision_v2 --output work/real_run_qa/completed_revision_v2
```

The output directory must be fresh. A missing completion receipt causes refusal before scientific inputs/results are opened. Completion and frozen-choice hashes authenticate the source/module/runtime identity, prepared files and every result artifact. All unique fit receipts and their files are rehashed sequentially without loading model objects or parameter histories into RAM.

The final audit writes identical bytes to `completed_result_audit.json` and `complete.json`. Its top-level fields include `status`, `run_sha256`, `evaluation_complete_sha256`, `checked_files`, `formulas`, `scope`, `limitations`, explicit actual/expected key-calendar hashes and honest table coverage. Root can copy the unchanged `complete.json` to `runs/revision_v2/independent_qa/complete.json` for the publication-figure gate. A failed audit states only completed sections and marks later checks as planned, never as completed.

The audit independently recalculates every model/seed/client scalar metric, macro and pooled estimands, five-seed summaries, training-only q80 thresholds, actual-heavy confusion/tail metrics, PR/AP/AUC, negative forecast diagnostics and no-head probability policy. It checks exactly96 months,5 clients and5 prescribed seeds with identical keyed observations, checkpoint metadata and identical-fit aliases.

All31 frozen comparisons receive an independent reconstruction of 6,000 paired PCG64 draws (2,000 each for 12/6/24 month blocks), observed/per-seed deltas and percentile intervals. CI midpoint and draw mean are diagnostic values; they are not substituted for the observed paired point or required to equal it. Requested sign tests use all256 common eight-block sign patterns and one31-member Holm family.

Cost validation covers actual60-round updates, admitted299/298 training calendars, distinct Ditto personal updates, central pooling, optimizer steps/examples, per-algorithm messages, fixed/adaptive/disabled proximity metadata, target/input summary sharing and zero extra charge for graph/sensitivity aliases. Classical/raw-pooled/normalized-pooled receipts and router OOF/calibration embargoes remain distinct. Tree/Ridge estimator-operation counts are explicitly inferred from saved pooled/local scope, while neural update counts come from actual telemetry.

Files:

- `audit_completed_run.py`: completion gate, authentication, complete-key/metric/uncertainty/cost orchestration.
- `metric_reference.py`: independent scalar rainfall and heavy-confusion formulas.
- `uncertainty_reference.py`: paired circular block bootstrap and exact block-sign/Holm arithmetic.
- `fit_cost_reference.py`: chronology, checkpoint-selection, message and step arithmetic.

No audit helper imports `caadapfl`, constructs a model, fits, loads a checkpoint, or performs model inference. Existing results remain untouched. A blocking discrepancy produces `completed_result_audit.json` with status `FAIL`; a fully consistent run produces `PASS` and numerical-discrepancy evidence.

The bounded synthetic check imports only the current evaluator's numerical functions and reads scientific source text for schema comparison. Its 96-month, five-client, five-seed fixture agrees on all 6,000 paired bootstrap draws, exact block-sign results and Holm arithmetic. Closed-form rainfall metrics, undefined denominators, strict q80/probability boundaries, tied PR scores, all seven neural cost schemas and the matched-latency export pipeline also pass. It found and repaired two audit-only false positives: adaptive proximity belongs to `[.3mu,mu]`, and equivalent integer/float CSV values must compare numerically. Changed costs, values and keys remain rejected. Evidence is in `synthetic_helper_validation_receipt.json`; none of these checks fits or scores an actual model.
