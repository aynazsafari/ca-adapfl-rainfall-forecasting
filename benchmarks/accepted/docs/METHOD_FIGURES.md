# Source-only method figures

The four figures explain the implemented FULL architecture, declared messages,
calendar support and frozen geographic domains. They do not read evaluation
labels, make predictions or show model accuracy. The generator authenticates
the active scientific modules, prepared FULL schema, frozen development choices,
country geometry and extraction protocol. The selected primary aggregation is
(1,0,0); the component-removal anchor is separately fixed at (0.3,0.3,0.4).

From the repository root, install the optional diagram dependency in your own
environment and choose a fresh output directory:

```powershell
python -m pip install -r requirements-method-figures.txt
python tools/make_method_figures.py --repo . --run runs/revision_v2 --output figures/methods
```

Shapely 2.2.0 was used for map intersections on the reference environment. It is
separate from the model environment and does not change the eleven core direct
dependency pins. Result figures use only core NumPy, pandas and Matplotlib.

Every method figure has PNG (300 dpi), PDF and SVG plus a source-hash manifest
and caption. Generation records RENDERED_AWAITING_VISUAL_QA. Inspect all four
actual PNGs at full resolution before accepting the release. Source-only
diagram review does not constitute evaluation-result verification.

Target scaling and q80 are fitted locally. The recorded reporting workflow
exports target mean/SD for decoding (16 bytes/client), and q80 appears in the
saved schema and evaluation diagnostics. Counts and two correlations are
one-time uploads; fixed whole-TRAIN strengths are sent each round. Plain
summaries and model updates provide no secure-aggregation or differential-
privacy guarantee. Privileged graph preparation and raw-pooled controls have
their separately declared server access.
