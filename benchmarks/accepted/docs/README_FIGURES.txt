Publication figure source and accepted actual releases

The packaged plotting source is tools/make_figures.py.
Matplotlib is the sole plotting library. NumPy and pandas are already core
benchmark dependencies; there are no mapping, dashboard or model dependencies.
The script imports no benchmark modules, tensor/pickle loader or network client.

From the repository root, inspect the fixed eight-figure plan without reading
any run data:

python tools/make_figures.py --plan

After completed evaluation AND independent numerical QA:

python tools/make_figures.py --repo . --run runs/revision_v2 --qa-receipt runs/revision_v2/independent_qa/publication_gate.json --verify-only
python tools/make_figures.py --repo . --run runs/revision_v2 --qa-receipt runs/revision_v2/independent_qa/publication_gate.json --output figures/a_fresh_original_directory

Defaults are canonical data/ and runs/revision_v2. --repo, --run, --data,
--qa-receipt and --output accept explicit locations. Existing output directories
are refused: choose a fresh directory to preserve accepted figure releases.
No preparation, training, tuning or evaluation is started implicitly.

Required gate receipts:
  RUN/evaluation/complete.json
  RUN/independent_qa/publication_gate.json (pass explicitly with --qa-receipt)

The original independent complete.json has a structured scope and remains
unchanged. For a new run, tools/qa/make_publication_gate.py creates the textual
scope view while retaining the complete structured scope and parent audit hash.

The independent receipt must contain status="PASS", the identical run_sha256,
evaluation_complete_sha256 equal to the exact evaluation/complete.json byte hash,
and a nonempty scope string. Missing, failed or mismatched QA prevents accuracy
CSV access. The script then checks source/code/choice/result hashes, complete
5-seed x 5-client x 96-month cohorts, independently recomputes all per-client and
macro R2/RMSE and paired point differences, and verifies all 31 contrasts and
6/12/24-month interval cohorts. SHA256 checks are consistency checks, not signed
proof of authorship or proof that an untrusted QA author is truthful.

The figures are fixed before scores:
  01 FULL global neural and explicitly named classical controls;
  02 FULL personal neural and three named graph methods;
  03 all 31 paired contrasts, in protocol order;
  04 anchor-minus-removal component ablations at fixed (.3,.3,.4);
  05 original/enriched/uniform gradient-training input affinities on 0..1;
  06 actual LR/mu/coefficient development grids and saved validation curves;
  07 per-client heavy recall, AP and tail RMSE, separate global/personal columns;
  08 five separate per-client seed42 global PR examples, no pooled PR curve.

All five seed points and mean +/- sample SD describe seed spread, not confidence
intervals. Paired forests show marginal 95% joint-client circular block12
intervals; full block6/24 sensitivity tables are exported. Positive R2 A-minus-B
favors A; negative RMSE A-minus-B favors A. Anchor ablations do not substitute
the separately selected primary coefficients as their parent. Exact block-sign
tests, if present, are exported without stars; their eight-block resolution
cannot reject at .05 after Holm adjustment over 31 tests. No superiority claim
is written before reviewing actual results.

Graph figures plot saved symmetric input affinities, not row-normalized trainer
weights. Both are exported, and training-only cutoff/support/count declarations
are checked. Enriched graph preparation uses privileged aligned cleartext
training rainfall and profiles; no secure aggregation or privacy guarantee is
implied. No geographic map or 2025 prediction is included.

Every figure has PNG at 300 dpi, PDF and SVG, minimum authored 10 pt text,
neutral titles, explicit units, fixed series order and redundant markers/styles.
figure_manifest.json binds the run, completion, independent QA, exact script,
all input/output hashes, per-figure source input hashes and CSV/caption lookups.
All saved prediction and metric CSVs are copied without filtering or rounding;
all model/seed/client PR coordinates are exported. All 96 evaluation months
remain accessible. Raw negative forecasts are retained. Figure panels distinguish
macro from pooled scores and global from personal identities.

Rendering success is recorded as RENDERED_AWAITING_VISUAL_QA. It is not visual
acceptance. Inspect every actual PNG/PDF/SVG for clipping, labels, signs, units,
uncertainty and series distinctions before manuscript insertion. Preserve font
size when placing the large 31-contrast forest; use its vector version for layout.

Actual verification now passes for all eight original exports in
figures/revision_v2 and all nine native-width companions in
figures/revision_v2_publication_v2. Each actual PNG was individually inspected
at original resolution, and hash-bound independent visual receipts are retained.
The first nine-companion candidate is retained with its marker-clipping finding;
the corrected release and exact display-only repairs are documented in
docs/PUBLICATION_LAYOUT.md and docs/publication_repairs. Synthetic preparation
checks remain separate. Original exports are large standalone figures; their
10pt authoring size does not remain 10pt when shrunk into a manuscript. Use the
accepted 6.7-inch companions, native tables and subsequent document render QA.
