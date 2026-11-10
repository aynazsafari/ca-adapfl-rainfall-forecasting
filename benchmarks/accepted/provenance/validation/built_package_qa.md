# Built package QA: PASS

READ-ONLY BUILT PACKAGE QA; source assembly only, no study fits or study scores

Passed 19/19 checks. The portable assembler ran only source assembly into work/package_qa_reassembled; no study model fitting or evaluation scoring was performed.

The six assembler outputs are compared byte-for-byte with both canonical_v2 and packaged data. The full manifest includes all seven native ERA5 fields, 522 original EE months and 30 recovered rows across six months; March 2024 remains EE.

The packaged supplemental default paths pass source-only admission for IMD/solar/GLDAS cohorts. Notebook syntax, empty outputs and copied identity pass; this review does not execute it against an incomplete study run.

## Checks

- PASS: every immutable packaged file byte/hash verified
- PASS: module hashes match package manifest
- PASS: canonical manifest SHA matches package manifest
- PASS: canonical source gate passed
- PASS: portable assembler output bytes exactly equal canonical v2 and packaged data
- PASS: portable assembler reproduces full manifest including seven-field recovery lineage
- PASS: ERA5 composition scope exact
- PASS: all eight ERA5 lineage hashes bind packaged bytes
- PASS: all canonical primary and supplemental inputs match their checksums
- PASS: main exact feature widths and converted units
- PASS: portable tools/assemble_data.py uses repository/local script roots
- PASS: portable tools/supplement_runner.py uses repository/local script roots
- PASS: supplement launcher source-only admission succeeds using packaged default paths
- PASS: three packaged documentation snapshots match final accepted-source docs
- PASS: results notebook has no saved study output
- PASS: results notebook exact copied identity and repository-relative default
- PASS: all direct dependency pins present in reference task environment
- PASS: direct-pins versus transitive snapshot limitation disclosed
- PASS: prepared main run uses exact packaged neural/core code

## Remaining release notes

- Main README still describes install/public-check verification as pending; root is performing these checks and should record the actual commands/receipts before release.
- provenance/era5_recovery/README.md retains original work/env and work/era5_rebuild commands. compare_provider_control.py default EE path uses the original workspace; from repository root, a relocated control recheck needs python provenance/era5_recovery/compare_provider_control.py --ee-table provenance/source_tables/era5_chunks/assembled_for_admission.csv. This affects optional source rebuilding instructions, not processed-snapshot model reproduction or the verified portable assembler. Rechecks write control receipts, so use a separate copy when retaining the frozen snapshot.
- Fresh source reconstruction requires optional icechunk/zarr/netCDF4/requests/pcodec/pyproj dependencies beyond the model direct-pin file; the main environment snapshot records them, but a dedicated rebuild environment is still a separate procedure.
- Code/data licensing remains owner/provider-specific; no blanket license or identity was invented.
