# CA-AdapFL revision benchmark

This package preserves the accepted revision_v2 study and supplies a separately authenticated exploratory revision_v3 experiment:100 full-budget fits, validation-selected capacity and common output loss, active-component controls, nonnegative/tail tradeoffs, matched observation latency, raw predictions and independent numerical QA. The2017–2024 evaluation period was already inspected; no untouched2025 test was verified.

Start with [the v3 reproduction guide](docs/REVISION_V3_REPRODUCTION.md). Source and completed-results archives serve different purposes. Source contains immutable preprocessed inputs, original monthly/index CSVs, verified archived prediction input, code and direct dependency pins. The completed-results companion contains all100 configs/checkpoint states/histories, choices, raw forecasts, metric/event/tail tables,49 contrasts and294000 paired draws, evaluation and independent-QA receipts. Exact aliases/projections do not add fits.

```sh
python -m pip install -r requirements-lock.txt
python tools/reproduce_revision_v3.py prepare
python tools/reproduce_revision_v3.py train --execute
python tools/reproduce_revision_v3.py evaluate --execute
```

Use a fresh short checkout path. Preparation performs zero empirical fits; training is a bounded100-fit study. In the delivered evidence checkout, use `python tools/reproduce_revision_v3.py verify`; new verification receipts never replace accepted evidence.

The executed `notebooks/Revision_v3_Inspectable_Analysis.ipynb` inspects evidence without fitting. `publication/revision_v3` binds scientific figures and supplementary tables to their reviewed CSVs. `docs/README_ACCEPTED_V2.md` retains the original accepted-study protocol and provenance.

The newly declared36-comparison annual-sign/Holm family includes32 fresh-output and four archive-based v2 controls. Eleven other archived baseline/router and two feature intervals are separate unadjusted diagnostics. The original31-comparison family is preserved. With eight annual blocks, the36-family minimum first adjusted p is0.28125; non-rejection is not equivalence.

CI runs implementation invariants and fresh preparation/contracts on Windows and Linux; it does not train100 empirical fits. The local Windows preparation path was tested. A GitHub-hosted CI run, public repository URL and DOI have not been represented as completed.

The accepted v2 headline remains historical and is not replaced by a better posthoc score. Five seeds are optimizer repeats; fixed regions overlap. Physical nonnegative outputs alone establish neither flood-warning suitability nor institutional-deployment/privacy benefit. Licensing and public-deposit decisions remain with the authors; no licence is invented.

`ENVIRONMENT.json` and `CURRENT_ENVIRONMENT.json` are preserved historical snapshots. The actual v3 training environment is bound in `runs/revision_v3/run_identity.json`; the separate document/notebook authoring environment is recorded in `provenance/authoring_environment.json`. See `provenance/ENVIRONMENT_SNAPSHOT_SCOPE_V3.json` for their exact scopes. The current source inventory is `PACKAGE_MANIFEST.json`; its accepted v2 predecessor is preserved as `provenance/ACCEPTED_SOURCE_PACKAGE_MANIFEST.json`.

Both v3 archives extract into `caadapfl_benchmark/`. Run the commands above from that folder. Accepted v2 main/supplement raw outputs and their original independent QA remain in the unchanged `CAAdapFL_Inspect_Completed_Results.zip` companion. That historical ZIP starts directly with `runs/`: extract it **inside the already created `caadapfl_benchmark/` folder**, so its outputs become `caadapfl_benchmark/runs/revision_v2` and `runs/supplements_v2`. The v3 results archive retains the archived prediction inputs needed for its own 49 contrasts, but does not duplicate the large historical checkpoint archives.
