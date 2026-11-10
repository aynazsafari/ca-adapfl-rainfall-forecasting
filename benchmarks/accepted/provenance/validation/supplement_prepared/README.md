# Prepared supplement audit

PASS: all six prepared case/arm inputs satisfy the pre-fit audit. No fits, predictions, evaluation numeric arrays or sealed truth values were read.

| Case | TRAIN/MON/VAL/EVAL per client | Reference/added channels | First target | Evaluation keys |
|---|---|---|---|---:|
| IMD | 299/35/71/85 | 43/44 | 1983-01 | 425 |
| solar | 272/35/71/96 | 43/44 | 1985-04 | 480 |
| GLDAS21 | 80/35/71/96 | 43/55 | 2001-04 | 480 |

Primary choices reused exactly: LR .001, FedProx/CA mu .1, CAAA coefficients [1,0,0]. No supplemental retuning. Both identities are evaluated later for each of the 90 fits.

The source-only formula independently derives every target key from all-client valid support t-15 through t-4. IMD excludes 2024-02 through 2024-12; its retained evaluation axis is contiguous through 2024-01.

Each case/arm input scaler is recomputed from one latest row per gradient-TRAIN sample. All 12 TRAIN history rows and context arrays match the stored float32 transform exactly. Local target scalers, strict q80 labels and deseasonalized climate sensitivities were independently checked.

Twenty-four nontruth prepared files match their claimed SHA256. Three sealed evaluation truth files were not opened or independently hashed; their existing claimed hashes are recorded separately. Evaluation archive name lists, headers and date keys were checked, without loading numeric evaluation feature arrays.
