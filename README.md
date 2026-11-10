# CAAdapFL reproducible benchmark

This repository packages the evidence for **Climate Components and Information
Sharing in Federated Monthly Rainfall Forecasting across Five Indian Regions**,
manuscript Marked5. It is a local release candidate, version 1.2.0; no repository
URL, DOI or publication acceptance is implied. [راهنمای فارسی](README_FA.md).

| Study | Completed fit bundles | Final output labels | Forecast rows |
|---|---:|---:|---:|
| Accepted primary revision_v2 |283|96|230400|
| Solar, IMD and GLDAS21 controls |90|36|83100|
| Capacity, components and output revision_v3 |100|46|110400|
| Information sharing |40|9 including climatology|64800 across all endpoints|

The current paper draws on **513 completed fit bundles**. Labels, aliases,
checkpoint branches, archived carried predictions and estimator operations are
not additional fits. Information sharing has 40 trained jobs and 8 trained
policies; the ninth label is a no-fit climatology. Historical seven superseded
capacity engineering fits and the separate systems_v4 extension are excluded
from that count. This is a single-host simulation, with already inspected
2017–2024 evaluation years. See [scientific boundaries](docs/PROTOCOL.md).

## Install

Use a short writable path, e.g. `C:\CAAdapFL`. The accepted saved-state runtime
is Python 3.12.14 on Windows 11 build 26200. CPU execution requires no GPU or cloud account.
From the extracted source folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python benchmark.py doctor
python benchmark.py install --assets "C:\path\to\release_assets"
```

This pip recipe installs the declared direct dependencies for arithmetic and
fresh-run setup. It has **not** been tested as an exact recreation of the
reference interpreter/PyTorch build. The recorded torch distribution is 2.14.1
while its loaded CPU version is 2.14.1+cpu; an official `+cpu` wheel may have a
different distribution/build signature. `doctor` reports the full recorded
signature match when dependencies are available. Exact replay/source training
require that full signature, including Python compiler and PyTorch compiled
configuration, in `metadata/reference_runtime_signature.json`. A matching list
of version numbers alone is insufficient. Retain the accepted guards.

Download all six result assets listed in `metadata/release_assets.json` from
the same GitHub Release, once published. The supplied local `release_assets`
folder is the equivalent before publication. Installation authenticates every
asset before extraction, rejects escaping paths and refuses changed existing
evidence. Assets total 5.964GB compressed. Allow approximately 20GB of free
disk for source, archives, extracted results and QA; fresh fits need additional
space. Model dependencies are in `requirements.txt`; observed Windows dependency
closure is in `requirements-reference-windows.txt`. That file is not a portable
resolver lock. All exact-runtime guards remain enabled.

## Reproduce the accepted evidence

```powershell
python benchmark.py verify --suite all
python benchmark.py replay --suite all
python benchmark.py tables --output publication/tables_fresh
python benchmark.py figures --output publication/figures_fresh
python scripts/execute_notebook.py --output work/notebook_recheck.ipynb
```

`verify` independently recalculates metrics, paired uncertainty and logical
costs from released forecasts/ledgers without fitting. On the public package,
source-control QA reports **PASS_OUTPUT_ARITHMETIC** and explicitly cannot
validate the withheld IMD input/scaler admission. `replay` loads authentic saved
states for inference, without fitting. The public solar/GLDAS portion is complete;
the IMD-added replay reports **PARTIAL_UNAVAILABLE**, so `replay --suite all`
returns exit2 until IMD is imported. Other suites continue. Exit2 is a stated
missing-input scope, not full success. See [data acquisition](docs/DATA_ACCESS.md).

Primary neural/baseline replay is exact on the reference host. Classical/router
rainfall uses a stated absolute1e-10mm comparison boundary; observed maximum
difference was1.14e-13mm. Capacity and available source-control forecasts were
exact. Runtime, code, states, features, truth and cohort guards are retained;
cross-hardware bitwise equality is not promised. Capacity replay distinguishes
new inference, aliases and authenticated carried archival predictions.

The executed notebook is `notebooks/Reproduce_Paper.ipynb`. It reads all four
studies, independently checks primary aggregation and draws the accuracy/cost
comparison; it performs no training. Heavy full bootstrap and state replay are
explicit commands above. `metadata/paper_evidence_map.json` maps all 45 paper and
supplement items to source evidence. Embedded original Figures1–13 are preserved
under `publication/reference`. Numerical redraws are equivalent data renderings,
not a claim of exact original pixel layout. Methods Figures1–4 are regenerated by:

```powershell
python benchmarks/accepted/tools/make_method_figures.py --repo benchmarks/accepted --output publication/methods_fresh
```

## Fit new models

Frozen scientific files are hashed in `metadata/frozen_source_inventory.json`.
Use a **new nonexistent directory outside both frozen benchmark roots**:

```powershell
python benchmark.py train --suite primary --workdir work/new_primary
python benchmark.py train --suite primary --workdir work/new_primary --execute
python benchmark.py train --suite capacity --workdir work/new_capacity --execute
python benchmark.py train --suite sharing --workdir work/new_sharing --execute
python benchmark.py train --suite sources --workdir work/new_sources --execute
```

Without `--execute`, the command prints the plan and creates no fits. With it,
primary reruns development and final comparisons; capacity reruns 100 jobs from
authenticated archived development inputs; sharing reruns 40 jobs under the
declared amended exploratory protocol; source controls rerun 90 jobs with frozen
parent hyperparameters. Source controls additionally require local IMD import
and the recorded primary Windows runtime. `--workers` controls primary/source
jobs; the frozen capacity/sharing launchers retain their own worker policy.
Each fresh run establishes its own identity and timing. Do not replace accepted
results with its output or describe already inspected years as a new holdout.

## GitHub deposit

Apache2.0 applies to author-owned software. Data terms remain separate and
IMD input values are excluded because redistribution permission was not
established. Source packaging and Git staging use an explicit SHA256 whitelist;
`git add .` is not the release procedure. See [deposit instructions](docs/DEPOSIT.md).
The six result ZIPs belong in Release assets; the compact source belongs in Git.
No credentials, private IMD sidecar, manuscript or reviewer correspondence is
part of the public deposit. [Validation report](docs/VALIDATION.md).
