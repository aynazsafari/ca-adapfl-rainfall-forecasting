# Observed frozen development choices

This note reports choices already frozen by the actual `revision_v2`
development execution. It does not report final evaluation performance.
The source is `runs/revision_v2/development/hyperparameters_frozen.json`,
SHA256 `680ddf67bf01d5b396bc9338c20b5173a9cdc613aaa4e53df68a5432b094083e`. Its receipt explicitly records no evaluation-label access.

The selected learning rate is 0.001. The proximal coefficient is
0.1 for each of the six methods with a tuned coefficient
(CAAA, FedProx, Ditto and the three graph variants).
The selected CAAA coefficient vector is [1.0, 0.0, 0.0], in the order
sample size, inner-monitor performance, and climate relevance. Thus the
primary aggregation has inactive performance and climate terms. Climate
inputs/embedding and adaptive proximity remain separate configured pathways.
Claims about an active climate-weighted primary aggregation would be incorrect.

The prepared main cohorts contain 299 TRAIN samples at each of the five
clients, with every client participating in every round. The selected
sample-size-only weights are therefore 299/1495 = 0.2 for each client.
This follows from the frozen inputs and implemented equation. Independent
inspection of 900 saved round rows and 4,500 client weights confirms exact 0.2
weights across the 15 unique selected-primary CORE/CLIMATE/FULL fits. Performance and climate
metadata are still computed and transmitted under the configured flags,
so inactive coefficients do not imply zero computation or communication cost.

The five component-removal experiments use the separate prespecified
[0.3, 0.3, 0.4] parent anchor. They measure changes to that anchor rather than
to the selected primary vector. These distinctions must remain visible in
tables and captions.

The feature-group selector fitted raw pooled Ridge (alpha 10) in three
chronological TRAIN-only folds. Mean fold macro R2 was
0.753891 for CORE (14 features),
0.758531 for CLIMATE (21), and
0.773728 for FULL (43).
The rule required at least 0.002 improvement over CORE and preferred fewer
features for ties. FULL was selected; `TRAIN_SELECTED` is consequently an
alias of FULL, sharing the same fit rather than incurring a duplicate fit.
The selector made nine estimator fits. These are included in whole-driver
elapsed time, but were not separately timed in the per-fit cost ledger.

These are development choices, not evidence that every feature is necessary,
that the indices are independent mechanisms, or that the selected model
will outperform controls on the final retrospective evaluation window.
The actual complete-fit and independent evaluation QA gates now pass; final
performance remains separate from these development choices.
