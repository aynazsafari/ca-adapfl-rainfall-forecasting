# Information sharing and monthly rainfall: controlled exploratory study

This is an extension of the completed CA-AdapFL Systems revision. It holds the
selected FULL12 ALCE architecture, local update budget and training randomness
fixed while changing input-statistic sharing and model synchronization cadence.
It is a one-host simulation over five overlapping Indian regions. Evaluation
years 2017–2024 were already inspected; this is not an untouched holdout.

## Design

Eight policies, five paired seeds 42/123/2024/7/999, 40 genuinely new fits:

| Policy | TRAIN input statistics | Model sharing |
|---|---|---|
| LOCAL_PRIVATE | Own region | None |
| LOCAL_STATS | Shared count/sum/squared sum | None |
| FEDAVG_PRIVATE_K1 | Own region | Every local tick |
| FEDAVG_STATS_K1 | Shared | Every local tick |
| FEDAVG_STATS_K5 | Shared | Every 5 ticks |
| FEDAVG_STATS_K20 | Shared | Every 20 ticks |
| CAAA_STATS_K1 | Shared | Every tick, selected adaptive proximal configuration |
| CENTRAL_POOL_STATS | Reconstructed from pooled TRAIN examples | Engineered example upload and final model deployment |

The common previously selected research configuration is prior coordination.
`LOCAL_PRIVATE` means no numeric TRAIN-statistic/model exchange within this
experiment, not absence of all prior shared design knowledge or a privacy guarantee.

Each distributed fit has 60 local ticks × 3 epochs, Adam 0.001 reset each tick,
batch 32, 299 TRAIN examples/client, 9000 optimizer steps and 269100 example
presentations. Cadence changes only synchronization: each client's state and
tick-specific RNG continue between synchronizations. Pooled training has the
same examples but 8460 steps due to batch boundaries; it is contextual.

The primary endpoint is tick 60. All federated primary predictions use the final
aggregated global model; local predictions use each region's own model.
Secondary checkpoint selection uses the same candidate ticks 20/40/60 in every
policy. Personal federated states are separately labeled diagnostics.

TRAIN 1983-01…2007-11:299/client; inner monitor2008-01…2010-11:35;
VAL 2011-01…2016-11:71; EVAL 2017-01…2024-12:96. December role-end labels
are embargoed as in the authenticated parent data construction. Target
normalization and q80 thresholds remain local in distributed training. All
network outputs use the same physical nonnegative head.

## Analysis and transparent amendment

`protocol.json` preserves the original prefit seven-comparison plan.
`analysis_amendment_v1.json` was recorded after training started, before any
new evaluation results were loaded or scored. Six contrasts address statistic
sharing, model exchange, cadence, and the scaling×exchange interaction. The
seventh changes proximal regularization at the same sharing policy, and is
reported as secondary context. The original Holm-seven analysis is retained
as sensitivity. This is an amended exploratory plan, not an original
confirmatory preregistration.

Primary outcome: equal-client/equal-seed MSE in mm². RMSE, R², skill against
TRAIN-only monthly climatology and tail diagnostics remain visible. The
interaction is recomputed within every paired bootstrap draw. Joint circular
date blocks 12/6/24 and paired seed resampling use 2000 draws per block length.
Clients are fixed, not sampled as independent institutions.

Annual squared-loss sign tests enumerate 256 assignments over eight years.
They assume annual sign exchangeability. The minimum two-sided p is 0.0078125:
Holm-seven cannot reach 0.05 even at that minimum; Holm-six can only narrowly
do so. Effect sizes, intervals and both multiplicity families are reported.
Five seeds do not create additional independent rainfall observations.

## Cost boundary

Required communication is the sum of uncompressed logical numeric payloads,
counting each sender/recipient, initial model delivery and final deployment.
Shared preprocessing uses int64 counts and float64 sums/squared sums/mean/std:
6920 bytes over five clients. Model states contain 17330 float32 parameters,
69320 bytes each. Inactive CAAA aggregation telemetry is omitted because the
selected aggregation weights are (1,0,0); its adaptive proximal mechanism
remains active locally.

Central upload is explicitly the unscaled engineered TRAIN x/climate tensors
and rainfall labels as float64, plus one int64 count/client. The server can
reconstruct its TRAIN scaling and q80 from those records. Overlapping windows
duplicate source information, so this is not a minimal original-product raw
upload estimate. Unused driver diagnostics need not be transmitted.

Roundwise research validation exports are priced separately. The three-point
common checkpoint selection costs 280 bytes/identity; reporting both global
and personal selections costs 560 bytes. Primary fixed-tick training needs
none of these validation exports. Final publication metrics are an audit,
not a hidden zero-cost private deployment claim.

Timing records single-process CPU training/aggregation separately from
validation audit. No WAN latency, transport/encryption overhead, energy,
differential privacy, secure aggregation or institutional deployment is
measured. Payload reductions do not imply privacy protection or universally
lower cost than centralized learning.

## Reproduce

Use Python 3.12 with `requirements.txt`. Results are immutable and guarded.
For a fresh empirical run, extract SOURCE alone into a new folder and run:

```powershell
python tools/run_experiment.py qa
python tools/run_experiment.py prepare
python tools/seal_evaluation.py
python tools/run_experiment.py train
python tools/verify_training_completion.py
python tools/evaluate_experiment.py
python tools/independent_qa.py --root . --output qa/independent_numeric_qa.json
```

Do not run `prepare` over an accepted RESULT identity. For a mathematical
replay of released results, extract SOURCE and RESULTS into the same parent,
then run the independent QA command only. To replay neural forecasts from
saved states, use `tools/replay_predictions.py` after the release is complete.
The notebook is an executed analytical companion and does not silently train
forty models when opened. Exact bitwise equality is not promised across
different hardware or library versions.

Training input/output hashes, full configurations, environment, both terminal
and selected states, round histories, local steps, message ledgers, raw
predictions, bootstrap resamples/draws, annual tests and QA receipts accompany
the release. Parent files remain unchanged.

## Literature context

- [McMahan et al., AISTATS 2017](https://proceedings.mlr.press/v54/mcmahan17a.html): model averaging and communication-efficient federation are established ideas.
- [Li et al., FedBN, ICLR 2021](https://openreview.net/pdf?id=6YEQUn0QICG): feature-distribution shifts motivate normalization concerns. This experiment does not implement FedBN; fixed input standardization is different from learned batch normalization.

The prospective contribution is case-specific evidence about information
policies and synchronization, not a new aggregation algorithm or proof that
federation is necessary. Results and reviewer implications are written only
after computation and independent QA.
