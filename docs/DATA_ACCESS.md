# Data access, attribution and IMD reconstruction

Provider attribution and extraction records live under
`benchmarks/accepted/provenance/source_tables`, `data/manifest.json` and
`benchmarks/accepted/docs/DATA_ATTRIBUTION.md`. The admitted CHIRPS target, ERA5/ERA5-Land/GLDAS21
regional predictors, climate indices and solar controls remain public snapshot
inputs. Cloud credentials are unnecessary for snapshot-level replication.
Original download/export scripts and provenance are retained; re-querying
providers later can produce a different identity and cannot recreate historical
real-time vintages. Apache2.0 applies to own code, not third-party products.

No independent IMD redistribution permission was supplied. Five source files
and four result-input files are omitted, with original hashes in the manifests.
Generated model states, forecasts and score/cost outputs remain included;
their observed evaluation truth is CHIRPS. The geographic support NPZ contains
cell coordinates/masks only, with no rainfall values.

Obtain the 44 annual 1981–2024 NetCDF files yourself from the
[IMD0.25-degree official page](https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html)
under its applicable terms. Put unchanged files named
`RF25_indYYYY_rfp25.nc` in a private folder. The import command authenticates
every native file against immutable expected hashes before reconstruction.
Provider revisions must not silently be labeled as the old source snapshot.

Native source arithmetic was performed with **Python 3.11.7, NumPy 2.4.4,
pandas 3.0.2**; prepared model features were made with **Python 3.12.14,
NumPy 2.3.5, pandas 3.0.1**. The environments must be separate to reproduce the
original 17-digit CSV bytes and hashes. Using the model environment for native
aggregation caused roundoff differences despite numerically close rainfall.

```powershell
py -3.11 -m venv .venv-imd
.venv-imd\Scripts\python.exe -m pip install -r requirements-imd-source.txt
.venv-imd\Scripts\python.exe scripts/import_imd.py --repo-root . --native-dir "C:\private\IMD"
.venv\Scripts\python.exe scripts/import_imd.py --repo-root . --rebuild-prepared
.venv\Scripts\python.exe benchmark.py verify --suite sources
.venv\Scripts\python.exe benchmark.py replay --suite sources
```

The packaged geography support removes any Shapely/PyProj requirement for the
native import. With exact source bytes, this restores all five source paths and
three IMD_ADDED_FULL prepared/schema paths byte-identically. It does not rebuild
the historical `CHIRPS_IMD_same_month_pairs.csv` diagnostic; that diagnostic is
not needed for the current primary/source/capacity/sharing comparisons.
Use `--output-dir NEW_DIRECTORY` for isolated validation, followed by
`--rebuild-prepared --import-dir SOURCE_IMPORT_DIRECTORY --output-dir ANOTHER_NEW_DIRECTORY`.
Existing different inputs are never overwritten. New source identities are
explicitly reported rather than passed off as accepted hashes.

Native files and all restored/source-intermediate IMD values remain local.
They are excluded from public source packaging and Git staging even after import.
The private sidecar beside the public package is an author-only local recovery
copy; it is not a Release asset and is never required for public CHIRPS-based
primary, capacity or sharing verification/replay. A file's download availability
is not treated as permission to redistribute it.
