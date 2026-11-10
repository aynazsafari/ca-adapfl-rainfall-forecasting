# Reproducing inputs and inspecting completed results

The processed-snapshot model benchmark runs from the repository without Earth Engine, CDS, NASA or Drive credentials. Install the model/notebook direct pins, perform the tested editable installation and run the eleven public synthetic checks described in the main README. Source reconstruction is a separate operation; it is not a model fit or a forecast result.

The accepted primary panel contains 2,640 client-month rows for 1981–2024. Its manifest preserves original/recovered ERA5 lineage and canonical units. The completed main directory is `runs/revision_v2`, package 1.0.1; preparation, restart, actual evaluation and independent saved-result QA pass. Its run SHA256 is `c10ec7499adba7470982ec57eb2daaba0634d79adae49ccb53ebca6309a218fb`. Keep its source/code/runtime freeze unchanged. The completed run contains 283 unique fit bundles and 96 output labels; costs for shared prediction aliases are counted once.

Revision v1 was stopped and superseded before evaluation after seven fits, following a measured memory probe with only 1.94 GB remaining margin while cached round histories were loaded. Its code and fit artifacts are retained under `snapshots/revision_v1_code` and `runs/revision_v1`. This is a runtime/memory observation, not an accuracy result. Revision v2 changes optional selected-state loading and streams loaded models one at a time; it changes no numerical training, data roles, grids or scientific protocol. V2 completed with two CPU workers. The seven superseded v1 fits are disclosed separately when accounting for total operational work.

## Assemble the retained source tables

From the repository root, use the model environment and a **new, nonexistent** output directory:

```text
python tools/assemble_data.py --source-dir provenance/source_tables --output-dir work/reassembled_v2 --with-gldas
```

This command reads the packaged target/index/POWER tables, the completed ERA5/Land/GLDAS source exports, native quality metadata and the immutable recovery receipts. It contacts no source service and performs no model work. It verifies the strict source gates and all original/recovered ERA5 rows before writing six files: `monthly.csv`, `indices.csv`, `solar_supplement.csv`, `gldas21_supplement.csv`, `source_quality.csv` and `manifest.json`. The GLDAS flag adds its admitted optional table; GLDAS still remains outside primary FULL. IMD is retained separately in `data/imd_supplement.csv` and the admitted source files under `tools/derived/imd`; the assembler does not fabricate it.

The portable packaged command has passed on the reference environment. All six files are byte-identical to the accepted canonical version 2 and their packaged counterparts. Verify a new assembly without changing the release:

```python
from pathlib import Path
import hashlib

rebuilt = Path("work/reassembled_v2")
for name in ("monthly.csv", "indices.csv", "solar_supplement.csv",
             "gldas21_supplement.csv", "source_quality.csv", "manifest.json"):
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(rebuilt / name) == digest(Path("data") / name), name
print("Six retained-table outputs match the packaged snapshot.")
```

This checks the full manifest as well as table values. A fresh manual `source_gate.passed=true` flag is not a substitute for its provider, native-mask, temporal-count, unit, coverage and checksum receipts. Unit conversions occur once: ERA5/Land K to °C, Pa to hPa and m/month to mm/month. POWER precipitation stays mean mm/day. No source or prediction values are clipped.

## Reacquire the pinned native ERA5 subset

Use a **separate copy** of the repository and a separate source environment. The recovery and provider-control programs write raw caches, derived tables and receipts relative to their own directory. A new retrieval timestamp or relocated file path changes receipt bytes even when the compressed climate bytes agree. Never run these writers in the immutable benchmark release or its active model environment.

The copied recovery directory contains the frozen country geometry, native masks, coordinates, serialized metadata, source-object byte-range ledger, compressed hashes, source flags, recovery contract and accepted controls. The 644 compressed climate chunks themselves are omitted from the repository: reacquisition requests exactly 1,465,963,879 bytes from the pinned snapshot, subject to a 2,000,000,000-byte ceiling and two network workers. Source availability remains an external requirement. The product/snapshot is frozen; do not replace it with a different ERA5 provider version, CHIRPS version or filled monthly panel.

The source program imports extra libraries beyond the model direct pins. Their observed reference versions are listed below; these are direct pins for a separate reconstruction environment, with transitive versions recorded in `ENVIRONMENT.json`, rather than a guarantee of every platform's resolver output:

```text
numpy==2.3.5
pandas==3.0.1
threadpoolctl==3.7.0
icechunk==2.3.0
zarr==3.4.0
netCDF4==1.7.4
requests==2.34.2
pcodec==1.0.4
pyproj==3.8.0
```

Create the environment inside the separate repository copy. On Windows, the commands use its explicit interpreter; on POSIX replace `.source-venv/Scripts/python.exe` with `.source-venv/bin/python`:

```text
python -m venv .source-venv
.source-venv/Scripts/python.exe -m pip install numpy==2.3.5 pandas==3.0.1 threadpoolctl==3.7.0 icechunk==2.3.0 zarr==3.4.0 netCDF4==1.7.4 requests==2.34.2 pcodec==1.0.4 pyproj==3.8.0
```

Before a **fresh** reacquisition, archive the timestamped raw-retrieval manifest inside that separate copy. The program deliberately rejects changed immutable manifest bytes, so retaining the old timestamped manifest at its original name would reject newly acquired records. Keep the frozen source evidence intact:

```python
from pathlib import Path

source = Path("provenance/era5_recovery/raw_retrieval_manifest.jsonl")
backup = source.with_name("raw_retrieval_manifest.frozen.jsonl")
if backup.exists():
    raise FileExistsError("Use a fresh separate source-reconstruction copy.")
source.rename(backup)
```

Then inventory the fixed subset, perform bounded acquisition and independently recheck the March control using its relocated Earth Engine table:

```text
.source-venv/Scripts/python.exe provenance/era5_recovery/recover_selected_months.py
.source-venv/Scripts/python.exe provenance/era5_recovery/recover_selected_months.py --execute
.source-venv/Scripts/python.exe provenance/era5_recovery/compare_provider_control.py --ee-table provenance/source_tables/era5_chunks/assembled_for_admission.csv
```

The first command inventories and checks the pinned public archive; the second retrieves climate payloads and writes native reconstruction outputs; the third writes provider-comparison receipts. The fresh-copy procedure is documented from the reviewed program interface and has not been independently rerun as a second 1.46 GB acquisition. Original acquisition and accepted numerical controls have already passed. Compare the newly acquired `raw_retrieval_manifest.jsonl` to `raw_retrieval_manifest.frozen.jsonl` by each `raw_path`, compressed SHA256, byte range and length. Retrieval timestamps may differ; the compressed climate identities must agree. Retain both receipt sets. Do not overwrite the accepted source manifest with newly written receipts: admitting a new reconstruction requires an explicit source version and run identity.

The accepted composition retains 522 Earth Engine months and replaces every one of the seven native ERA5 variables and complete support metadata for 2022-04, 2022-05, 2022-07, 2022-08, 2024-01 and 2024-02. Earth Engine lacked 21 precipitation validity hours across those months, causing its strict joint mask to fail. Recovery is native source reconstruction, with zero imputed cells and no interpolation. March 2024 remains Earth Engine data and is a compatibility control. January 1981 and March 2024 regional checks do not prove authenticated CDS bitwise equality, an encoding explanation or historical operational vintages.

All selected native hours require source status zero and final expver one. State variables use monthly validity-time means; precipitation sums one-hour amounts ending at 01 UTC of the first day through 00 UTC of the next month. Native cell centers, frozen country coverage, half-open boxes and whole-cell WGS84 geodesic areas remain unchanged. The original failed Earth Engine export is preserved. Rebuilding all Earth Engine assets, historical release vintages or all CHIRPS GeoTIFFs is outside this retained-table/native-subset recipe.

## Run the admitted matched supplements

Use the **model** environment in the original repository. After primary `development/hyperparameters_frozen.json` exists, the launcher can prepare a separate run with its packaged defaults:

```text
python tools/supplement_runner.py prepare --package-root . --canonical-dir data --primary-run runs/revision_v2 --output runs/supplements_v2
python tools/supplement_runner.py run --package-root . --output runs/supplements_v2
python tools/supplement_runner.py run --execute --workers 1 --package-root . --output runs/supplements_v2
python tools/supplement_runner.py evaluate --package-root . --output runs/supplements_v2
```

Default source directories are `tools/derived/imd` and `tools/derived/gldas_full`. Default plans are IMD/solar under `tools/derived/cohorts` and GLDAS under `tools/derived/cohorts_gldas_admitted`. The older structural GLDAS placeholder is not the admitted plan. Source-only admission through these exact packaged defaults passed QA.

The dry `run` writes the 90-fit schedule without fitting. Execution uses three sources × two arms × three methods × five seeds and fixed primary LR/μ/CAAA coefficients, with no supplemental tuning. The added and reference arms use identical source-quality target keys and refit normalization, event thresholds and checkpoints. IMD has 299 train and 85 evaluation months per client; solar has 272/96 and GLDAS 80/96. The separate evaluator opens its truth only after all fits and checkpoint choices are frozen. It has no `--execute` option. These within-source comparisons do not enter the primary 31-pair family or compare against a longer unmatched primary cohort.

Coordinate a combined maximum of two model processes, each with two numerical threads. The actual supplemental run completed all 90 unique fits after primary completion, followed by evaluation and independent QA PASS: 36 output labels, 83,100 raw prediction rows, 18 within-source contrasts and 108,000 reproduced bootstrap draws. These results are retained under `runs/supplements_v2`; use a fresh directory for any rerun instead of replacing accepted artifacts. GLDAS's producer-documented preceding-three-hour semantics remain distinct from its unexplained ingestion start/end-hour metadata; source admission does not resolve that numeric time-bound limitation.

## Inspect the completed main run

`notebooks/inspect_results.ipynb` reads completed saved outputs only and defaults to `runs/revision_v2`. It refuses an absent `evaluation/complete.json` and never prepares data or trains a model. This source remains unexecuted. The actual completed-run execution is retained separately as `notebooks/inspect_results_executed.ipynb`; all seven code cells completed without error in 17.38 seconds. All three embedded plots were individually inspected at original resolution. Synthetic preparation checks are retained separately and are not study results.

The model/notebook pins include ipykernel 7.4.0 and nbclient 0.11.0. Actual headless verification used a temporary local kernel specification whose argv pointed to the matching model interpreter (`python -m ipykernel_launcher -f {connection_file}`), a temporary `KernelSpecManager`, and the repository working directory. No global kernel was registered; the kernel was shut down and channels stopped. Execution and visual receipts are retained in `provenance/notebook_execution_actual.json` and `provenance/notebook_visual_QA_actual.json`. They bind the source and completed-run identities and record that no fitting or inference occurred. A new verification should retain its own executed copy and receipt.
