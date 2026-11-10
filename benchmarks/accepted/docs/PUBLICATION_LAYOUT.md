The source-only layout was approved and frozen before main evaluation. These nine companion plates preserve the original eight scientific figure scope and address manuscript legibility at a native width of 6.7 in. The layout plan remains byte-identical. Data, models, tuning, evaluation, metrics, the comparison family and source cohorts remain as frozen. A disclosed post-evaluation plotting-verifier repair checks the graph builder's exact float32 serialization; the original plotting sources are retained under docs/publication_repairs. No rendering function or figure selection changed.

Four plates show all 31 paired comparisons in exact protocol order: 1–8, 9–16, 17–24, 25–31. Each contains both macro R² and macro RMSE differences. Each metric uses one shared limit across all four plates, derived from zero and every point and 95% interval endpoint in the full 31-row family, with 8% span padding. No ranking, clipping, transformation, omitted comparisons, significance stars or result-selected panels are permitted. R² A-minus-B positive favors A; RMSE A-minus-B negative favors A. Intervals are conditional marginal 95% paired circular-block 12 intervals, not simultaneous family intervals. Block 6/24 sensitivity tables remain accessible.

The remaining plates show fixed FULL global/classical identities, fixed FULL personal identities, the five fixed-nonzero-anchor component removals, three training graph inputs, and five separate client PR examples at predeclared seed 42. The anchor remains (.3,.3,.4) and is distinct from the selected primary coefficients. Global and personal identities retain separate names and plates. All paired seed points accompany mean ± sample SD; SD is not a confidence interval. Graph colors retain raw affinities on the shared 0–1 scale; cell labels round to one decimal to prevent touching text at native width. Exact affinities and row-normalized trainer weights remain in CSV. PR uses three FULL global neural identities and all five clients, with no pooled-client curve. Full all-seed metrics and all saved source exports remain available from the original tool.

All canvases are exactly 6.7 in wide, fonts are at least 10 pt, and PNGs have 2010 px width at 300 dpi. PDF/SVG are native vector exports. Saving uses the fixed canvas without tight-bbox expansion or shrinking. A text-containment and pairwise text-box check refuses overflow/overlap. Actual rendered plots still require visual review for marker/curve legibility. The original 06 validation-grid and 07 heavy-event artifacts remain repository/supplement material; manuscript heavy-event summaries may use native tables. They should not be shrunk to 6.7 in with unreadable fonts.

Use the packaged repository, including tools/layout_plan.json and docs/publication_repairs. The repair gate verifies the original source and exact allowed plotting-verifier byte replacement, along with the unchanged plan. The 47-entry model abbreviation ledger is embedded in the frozen plan and exported alongside every companion set. The core NumPy/Pandas/Matplotlib environment suffices; this helper introduces no new model or plotting dependencies.

After the completed run has independently passed QA, render and visually accept the original eight-figure export first. Supply FIGURE_DIR/visual_qa.json (or --figure-qa PATH) with this receipt schema:

```json
{
  "status": "PASS",
  "run_sha256": "64 lowercase hexadecimal characters",
  "figure_manifest_sha256": "SHA256 of the exact original figure_manifest.json",
  "scope": "Original scientific render accepted; manuscript legibility addressed by companions."
}
```

Then, from the repository root:

```powershell
python tools/make_publication_companions.py --plan
python tools/make_publication_companions.py --repo . --run runs/revision_v2 --figures figures/revision_v2 --qa-receipt runs/revision_v2/independent_qa/publication_gate.json --verify-only
python tools/make_publication_companions.py --repo . --run runs/revision_v2 --figures figures/revision_v2 --qa-receipt runs/revision_v2/independent_qa/publication_gate.json --output figures/a_fresh_publication_directory
```

The first command reads only the adjacent frozen plan. The other two require evaluation/complete.json, hash-bound independent QA PASS and original-figure visual QA PASS before any accuracy CSV is read. The explicit publication_gate.json is a textual-scope view of the original complete.json; it retains the unchanged structured scope and parent receipt hash. The helper verifies run/code/protocol/choice identities, all completed evaluation hashes, accepted plotting-source lineage, all original figure outputs and their hashes, and exact frozen model/client/seed/pair order. It reads only accepted source exports for plotting. It performs no training, inference, source preparation, evaluation, pickle loading, network or subprocess calls. SHA256 supplies consistency checks, not cryptographic authentication of an author.

Each fresh output directory includes nine PNG/PDF/SVG sets, source CSV copies, a full abbreviation ledger, captions with source CSV lookups and run identity, frozen layout plan and companion_manifest.json. The manifest remains RENDERED_AWAITING_VISUAL_QA until all nine real plates have been independently inspected. Existing output directories are refused. The original eight figures and scientific modules are never modified.

The retained pre-evaluation synthetic integration checks exercised the gates, rejection before CSV reads, exact pair/order/seed/client schemas, full coverage, common limits, a point outside its percentile interval, three graph inputs, all five client PR examples, text geometry, fonts and physical PNG dimensions. Its watermarked fixtures contain no research results. The scratch test is tailored to this preparation workspace and is not presented as an already-packaged public test. Actual original-eight rendering and visual QA pass. All nine corrected companions in figures/revision_v2_publication_v2 were individually inspected at original resolution and independently accepted; visual_qa.json binds the immutable companion_manifest.json. The first actual candidate in figures/revision_v2_publication is retained with its failed marker-containment finding. A second disclosed display-only repair computes seed-plot padding after extending the R2 range to zero. It changes only the P05/P06 axis limits and six corresponding PNG/PDF/SVG exports; all sources, captions, identities, frozen order and seven other PNGs remain byte-identical. Repair, exact diff, issue receipt and independent acceptance are retained under docs/publication_repairs. Scientific modules remain unchanged.

Four source-only method companions in docs/method_publication_companions are independently accepted at native 6.7-inch width, with diagram labels at least 10.5 pt. The original method figures and captions are retained. Optional captioned vector pages use 10 pt captions and have mechanically checked geometry/text; visual review covered each standalone PNG. Word embedding and document page rendering remain a subsequent QA phase.
