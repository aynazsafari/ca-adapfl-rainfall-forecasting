# Finite revision experiment

The accepted revision_v2 result remains the historical primary. revision_v3 is a separately identified exploratory study of a period already inspected, with 100 full-budget empirical fits. Five seeds repeat training randomness; they do not provide five independent rainfall records. The five fixed regions overlap. No untouched 2025 holdout was verified.

Use Python 3.12 and the pinned requirements-lock.txt. The recorded execution used Python3.12.14, PyTorch2.14.1+cpu, Windows11, two workers and two CPU threads per worker. A fresh reproduction records its own runtime and hashes; numerical identity across hardware/operating systems is not presumed. Environment installation on every alternative platform has not been validated.

In a fresh extracted source checkout:

```sh
python -m pip install -r requirements-lock.txt
python tools/reproduce_revision_v3.py prepare
python tools/reproduce_revision_v3.py train --execute
python tools/reproduce_revision_v3.py evaluate --execute
```

Preparation authenticates four accepted-main preprocessed inputs, rebuilds compact and matched-delay cohorts, checks the source panel and performs numerical contract fixtures. It performs zero empirical fits. Training performs25 development and75 matched-control fits, each60 rounds, three local epochs and the five paired seeds42,123,2024,7,999. All candidate/checkpoint choices use validation. Evaluation authenticates the frozen choices, all100 complete receipts, state files, source/runtime identity and separately frozen analysis code before opening the separate truth file. Independent verification recomputes reported arithmetic from raw CSVs without importing production scoring helpers.

Do not run `prepare` in an existing evidence checkout. For a fresh replication, extract the source ZIP into a new short path. The source ZIP excludes `runs/revision_v3`; the results companion supplies that directory for inspection. To verify the delivered evidence, extract both archives into the same directory and run:

```sh
python tools/reproduce_revision_v3.py verify
```

The delivered verification receipt already exists and will never be replaced. `verify` writes each new receipt into a unique timestamped directory under `runs/revision_v3/independent_rechecks`. This changes neither accepted results nor their original verification receipt.

RAW CAAdapFL and RAW FedProx include documented exact aliases of existing fits; clipped forecasts and validation-selected output-loss labels add no fits. All source/data manifests,100 fit configurations, selected checkpoints, histories, optimization/example counts, raw predictions, five-seed summaries, tail diagnostics,49 contrasts and294000 saved paired-bootstrap draws are in the results companion. The newly declared36-comparison family includes32 fresh-output comparisons and four archive-based v2 controls, with separate Holm annual-sign adjustment. Eleven other archived baseline/router and two feature contrasts have only unadjusted exploratory intervals. The accepted31 family remains unchanged.

The latency sensitivity retains target-first-day issuance and adds two months of observed-input delay. It does not establish a longer true forecast lead. FULL/COMPACT, embedding removal and adaptive-prox removal are distinct interventions. Nonnegative decoding operates in physical mm, with per-row client target scales retained in the central pool. Tail weighting uses a fixed TRAIN event-frequency normalizer.

No public repository URL, archival DOI, data permission or software licence is fabricated. The original source/citation/licensing notes remain in docs/README_ACCEPTED_V2.md and data/provenance files. This is a local GitHub-ready package; publication and licence selection remain with the authors.
