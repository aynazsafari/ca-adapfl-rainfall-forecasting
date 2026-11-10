# Independent checks

These tools read authenticated saved evidence. They do not train, select
features, load models or perform inference. Run from the repository root with
a fresh output directory. Completion checks reject unfinished scientific runs
before result reads.

```powershell
python tools/qa/audit_completed_run.py --run runs/revision_v2 --output work/qa_primary
```

A successful audit writes `complete.json` with status PASS and binds the exact
evaluation completion receipt. Copy these unchanged bytes to
`runs/revision_v2/independent_qa/complete.json` to satisfy the descriptive-diagnostic
gate. The figure consumers require a textual scope. Preserve the original receipt
and create its portable textual view with:

```powershell
python tools/qa/make_publication_gate.py --run runs/revision_v2
```

Pass `--qa-receipt runs/revision_v2/independent_qa/publication_gate.json` to both
figure consumers. This view retains the original structured scope and audit hash,
performs no scientific calculations and refuses an existing output. The accepted
release retains its original workspace-created view; the portable tool supports
fresh completed runs. A failed audit is not a publication gate.

The audit checks complete keys, saved file hashes, all per-client metrics,
separate macro/pooled and five-seed summaries, training thresholds, heavy-event
and PR metrics, all 31 paired contrasts, 186,000 bootstrap draws, exact annual
block sign tests, and actual cost/checkpoint/message telemetry. The independent
formula helpers are adjacent to the script; no `caadapfl` module is imported.
Read the retained scope and limitations in each receipt.

```powershell
python tools/qa/descriptive_diagnostics.py --run runs/revision_v2 --output work/diagnostics --imd-dir tools/derived/imd --imd-manifest-sha256 ORIGINAL_FROZEN_IMD_MANIFEST_SHA256
```

Supply the original frozen admission-manifest hash from source provenance,
not a newly accepted hash of potentially changed files. The diagnostics also
require the primary independent PASS receipt. Eligible ONI-value bins use
target-minus-two-month support; they are descriptive bins, not official NOAA
event classifications. CHIRPS–IMD agreement is same-month product agreement,
with the invalid C1 October 2023 record visibly excluded. These diagnostics
add no fitted model, selection rule or inferential comparison family.

The retained readiness and synthetic receipts describe implementation checks
performed before actual results. They do not substitute for the completed
study audit. Source-only feature redundancy is a separate TRAIN diagnostic in
`docs/feature_diagnostics`; its centralized offline access is distinct from
model communication telemetry.
