# Deposit procedure

1. Review root README, LICENSE/NOTICE, attribution, manifests and validation.
   Software license is Apache2.0, as authorized by the authors. Keep the IMD
   exclusions. Do not upload private data, native NetCDF or reviewer documents.
2. Extract `CAAdapFL_GitHub_Source_v1.2.0.zip` into a short clean checkout.
   Source contains an explicit `metadata/source_manifest.json`; verify it with
   `python scripts/package_source.py verify`. To stage exactly this whitelist:
   `python scripts/package_source.py stage --init-git --execute`.
   Preview with `stage` without `--execute`. The tool checks every source hash
   and force-adds only listed source paths (some frozen nested ignores include
   legitimate small source-data figures). Never use broad `git add .` here.
3. Create the desired GitHub repository, commit/push reviewed source and tag 1.2.0.
   Publication is a separate user action; it has not been performed by this
   packaging task. Fill actual repository URL and DOI in metadata/CITATION only
   after they exist; update source manifest after those authored changes.
4. Attach all six files from `release_assets` and `SHA256SUMS.txt` to the matching
   Release. Each file is below 2GiB. Source/Git files remain below 50MiB.
   [GitHub Git-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
   and [Release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
   motivated the separation. The complete source ZIP may also be a Release asset.
5. From a clean source extraction, download all six matching assets; run install,
   doctor, independent verify, available replay, notebook and exports. For full
   IMD-added replay, acquire/import IMD locally. Keep original and new receipts.
6. Record the immutable commit/tag/Release URL in the manuscript and response.
   A DOI archive such as Zenodo can be added after deposit. Do not invent a DOI.

The root GitHub Actions workflow defines bounded implementation/packaging and
source-preparation checks on Windows/Linux. It does not download 6GB results,
retrain the complete benchmark or assert hosted CI already passed. Saved-state
replay is tied to the recorded reference runtime; cross-platform exact parity
has not been empirically established.
