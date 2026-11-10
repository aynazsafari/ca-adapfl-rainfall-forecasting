# Final source admission audit

**PASS.** The actual ERA5 composition replaces exactly 30 client-month rows for the six frozen defective months, using all seven recovered variables. The other 2,610 rows (522 months), including March2024 control values, preserve original Earth Engine numeric values and calendar/quality metadata exactly.

The six repair months are 2022-04, 2022-05, 2022-07, 2022-08, 2024-01, 2024-02. They contain 21 unavailable TP validity records in Earth Engine; all state-band counts are complete. Replacement values come from the pinned final ERA5 public snapshot `ZFKDHBCTBVHVXM3BQFV0`. No statistical filling or monthly clipping was used.

Independent source checks rehashed all 644 compressed climate chunks (1,465,963,879 bytes), every retained metadata file, recovery artifact and receipt. Pinned coordinate and quality arrays were decoded: all 5,139 required unique boundary-inclusive timestamps have source flag0 and expver1. Native country/half-open-box masks and full-cell WGS84 areas were independently recomputed. All seven decoded hourly NetCDF files were checked numerically; 118335287 values were finite without sentinel values. The 245 monthly reductions agree within 1.89e-10 native units.

State variables use `[month start00,next month start00)`. TP uses one-hour amounts ending valid time, `(month start00,next month start00]`; no daily cumulative reset or second integration is applied.

| March2024 variable | Largest provider difference | Frozen native-unit tolerance |
|---|---:|---:|
| t2m | 9.827681424e-07 | 0.0001 |
| d2m | 5.16307864e-07 | 0.0001 |
| sp | 0.0003355585795 | 0.01 |
| msl | 0.0002499094699 | 0.01 |
| u10 | 2.214915225e-07 | 0.0001 |
| v10 | 3.052252618e-07 | 0.0001 |
| tp | 9.69195935e-07 | 1e-05 |

All 35 March numeric and 60 calendar/count/area comparisons pass. The March archive values are used only as a compatibility control and are not substituted into the composed source.

Canonical v2 preserves all five v1 CSV files byte-for-byte. Its only manifest addition is `source_gate.era5_recovery`, with every recorded receipt/source SHA256 verified against current bytes. The monthly panel has 2,640 unique complete keys, 22 finite meteorological columns and unchanged CHIRPS targets. Six corrected climate indices have 528 finite monthly rows. Native ERA5, ERA5Land and GLDAS source calendars, integer counts, physical intervals, fixed geometry and 95% coverage gates pass.

| Source | Rows | Minimum area coverage |
|---|---:|---:|
| ERA5 | 2640 | 1 |
| ERA5Land | 2640 | 0.993154792724 |
| GLDAS21 | 1500 | 0.954957663253 |

All source-to-canonical unit conversions pass, including K→degC, Pa→hPa and ERA5/Land monthly metres→mm. POWER PRECTOTCORR stays a mean daily rate in mm/day; solar stays mean daily irradiation in MJ/m²/day. Solar has structural missingness during 1981–1983 (36 months, 180 client-months); its 2,460 available client-months are finite. GLDAS begins in 2000 and remains a matched-cohort supplement. December2007/2010/2016 label embargoes and source latency assumptions remain in the manifest.

## Evidence files

- `final_source_audit.json`: exact repaired keys, 700+ hash checks, numerical reductions, geometry areas, source conversions and source calendars.
- `independent_native_aggregation.csv`: independent 245 monthly numerical comparisons.
- `audit_final_sources.py`: bounded, read-only audit program.

## Scope limits

- The retained compressed climate chunks were rehashed and their inventory/ranges verified; seven decoded hourly NetCDF arrays were fully numerically verified. The 644 compressed climate chunks were not independently redecoded in this audit.
- Independent provider agreement is regional, with frozen January1981-derived tolerances and March2024 controls; no authenticated CDS bitwise equivalence is established.
- Final revised archives and declared latencies define a retrospective availability scenario, not reconstructed historical operational vintages.
- GLDAS physical three-hour semantics and producer documentation support aggregation; raw GLDAS time bounds are not independently numerically verified here.
- This audit performs no model fitting, model selection or forecast evaluation.

No blocking source issue found. This is a source admission result, not evidence of model skill or of untouched evaluation labels.
