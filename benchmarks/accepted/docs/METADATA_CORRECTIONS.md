# Frozen metadata clarification

The prepared input-scaler record in `MAIN_FULL_schema.json` and related schemas
contains a sentence stating that local target scaling and heavy thresholds
remain at clients. This sentence is too broad for the recorded reporting
workflow. It is preserved in the frozen schemas for provenance; this notice
corrects its interpretation without changing source arrays, code, model
training, hyperparameters or run identity.

Target mean, target population SD and q80 are fitted on each client's TRAIN
labels. The runner exports two float64 summaries for inverse decoding,
amounting to 16 bytes/client or 80 bytes for five clients. q80 is also present
in the saved schema and evaluation diagnostics. These reporting exports are
plain values. Their presence prevents a claim that target statistics and
thresholds never leave clients.

Input scaling itself is shared through TRAIN count/sum/squared-sum vectors.
CAAA sends two local ENSO/IOD correlations and sample counts once, fixed
whole-TRAIN strengths each round, model states and scalar diagnostics. Graph
preparation separately accesses aligned TRAIN rainfall, and centralized/router
controls use declared raw pooling. No secure aggregation or differential
privacy is implemented. Source-only method-figure captions and the cost ledger
describe the actual access and reporting boundary.

This clarification was recorded before evaluation completion. The frozen
active run is `c10ec7499adba7470982ec57eb2daaba0634d79adae49ccb53ebca6309a218fb`.
Numerical source, fit, checkpoint and metric hashes remain authoritative.
