> Historical preparation note: the readiness checks and workspace commands below record pre-run preparation. Actual primary, supplemental and descriptive audits now pass. Current portable repository-relative instructions are in [QA README](README.md) and [current release](../CURRENT_RELEASE.md). For new work use python tools/qa scripts and fresh output directories. Original preparation bytes are retained under provenance/historical_release_guides.

# Gated descriptive diagnostics

`descriptive_diagnostics.py` is prepared and synthetic-validated. It has not read actual study results or source matrices. It cannot proceed until the primary evaluation completion receipt and `independent_qa/complete.json` exist, the latter has a scoped `PASS`, and both bind the same run identity and exact evaluation-completion bytes. It also authenticates the evaluated prediction file, frozen choices, canonical files, active scientific module hashes, frozen CHIRPS source-quality table and admitted IMD receipt/output hashes before any scientific CSV read.

After completion and independent QA, run from the workspace with a fresh output directory:

```powershell
work/env/Scripts/python.exe -X utf8 work/real_run_qa/descriptive_diagnostics.py --run outputs/03_benchmark/caadapfl_benchmark/runs/revision_v2 --output work/real_run_qa/completed_descriptive_revision_v2 --imd-dir outputs/03_benchmark/caadapfl_benchmark/tools/derived/imd --imd-manifest-sha256 <frozen-admission-manifest-sha256>
```

The IMD manifest SHA is the already frozen admission receipt SHA from the package/source provenance. Supply that original value; recalculating and accepting changed source bytes defeats the external authentication. Paths can relocate; identity uses scientific contents, not the directory name.

ONI bins use exactly `oni_anomaly_3mo` at target month minus two: values `<= -0.5`, between `-0.5` and `+0.5` exclusively, and `>= +0.5`. All 96 evaluation dates receive exactly one bin, and every saved model/seed/client/date prediction key is retained. These are descriptive eligible-value bins. No five-season persistence rule or official NOAA event classification is inferred. No new p-values, fitted models, threshold selection, lag selection, retuning or inference family is added.

Tables preserve per-client/seed/bin metrics, equal-five-client macro metrics and pooled metrics separately, with explicit date/pair/client counts. Five-seed summaries summarize per-seed metrics. Empty/constant/zero-denominator cases remain NaN and have per-metric flags. No undefined client or seed is silently dropped from the macro mean. Raw negative forecasts remain included.

CHIRPS–IMD agreement uses same-month regional monthly totals in mm, exact train/monitor/validation/evaluation target roles, and each source's spatial coverage/quality fields. All raw pair keys are saved. The invalid Client1 October 2023 IMD record is visibly excluded, retains its physical mean in the quality column, and keeps its admitted value missing. The three role-boundary Decembers (2007/2010/2016) are excluded by role definition, as are native 1981–1982 months before the benchmark target cohort.

This source comparison differs from the matched forecasting IMD sensitivity: lagged feature histories exclude targets February–December 2024 there. Those later target exclusions are not applied to direct same-month product agreement. CHIRPS is station blended and IMD gauge gridded; agreement does not establish statistical independence or provide target-free verification. CHIRPS is the numerical reference for descriptive R2/NSE and signed PBIAS, not an unqualified truth claim. These tables are separate from predictive benchmark scores.

Eight CSV outputs:

- `oni_month_bins.csv`: all 96 eligible support months, values and bins.
- `oni_prediction_key_bins.csv`: every evaluated key and its bin, without filtering.
- `oni_per_client_seed_bin.csv`: scalar metrics and undefined flags.
- `oni_aggregate_per_seed_bin.csv`: separate macro/pooled metrics and counts.
- `oni_five_seed_summary.csv`: paired seed means/standard deviations and undefined counts.
- `CHIRPS_IMD_same_month_pairs.csv`: role/date pairs, raw/admitted values, coverage and explicit exclusions.
- `CHIRPS_IMD_agreement_per_client_role.csv`: scalar agreement and raw/valid/missing/excluded counts.
- `CHIRPS_IMD_agreement_macro_pooled_role.csv`: separate equal-client and pooled source agreement.

`complete.json` records all file hashes/row counts, gates, source hashes, prespecified interpretation, code hashes and retained-key hash. Synthetic validation covers threshold edges, target-minus-two support, complete key retention, separate macro/pooled populations, empty-bin NaNs, direct source exclusions, role counts, preservation of invalid-source NA, changed calendar/fill rejection, and completion/QA refusal before CSV reads. The actual scientific run remains unopened by preparation.
