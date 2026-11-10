# Implementation checks

The local GitHub Actions workflow runs the existing implementation-invariant
tests on Python 3.12, on Windows and Linux runners. The tests include two tiny
two-round synthetic training fixtures. Research benchmark training, climate
archive downloads and research prediction evaluation require separate CLI
operations. The action
references are pinned to commits resolved from the official v7 tags on
2026-10-08; the read-only retrieval receipt is in
`provenance/tooling/github_action_pins.json`.

The research reference execution is Windows/Python 3.12. The workflow has been
prepared and statically inspected locally; it has not been uploaded to GitHub
or executed by GitHub Actions. Linux parity is not claimed from this file.
The CPU variant `torch==2.14.1+cpu` is installed first from the official PyTorch
CPU index. The public-version requirement `torch==2.14.1` also accepts this
local version. Both Python 3.12 platform-wheel links were observed in that index
and returned successful HEAD responses; wheel downloads, integrity checks and
hosted installation remain unexecuted.
The direct dependency versions are the observed reference versions;
`requirements-lock.txt` does not pin every transitive dependency. The complete
observed environment is recorded separately.

The workflow requests read-only repository contents permission, disables
persisted checkout credentials, and has no scheduled trigger or deployment
step. An actual accuracy rerun remains an explicit benchmark CLI operation.
