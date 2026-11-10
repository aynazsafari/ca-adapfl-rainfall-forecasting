# TRAIN feature diagnostic

Read-only diagnostic of active revision_v2; authentication PASS. No feature changes, fits, evaluation reads or label reads.

Only the latest row of the 12-step sequence is analyzed after raw_target_month <= 2007-11: 299 rows per client, 1495 client-month rows.

All analyzed channels are dimensionless global TRAIN z scores. The dictionary also records their original physical units and source-support basis.

| Scope | Finite cells | Constant features | Pairs with abs(r) >= 0.95 |
|---|---:|---:|---:|
| Client1_NW_Arid | 12857 | 0 | 14 |
| Client2_Indo_Gangetic | 12857 | 0 | 18 |
| Client3_NE_Himalaya | 12857 | 0 | 29 |
| Client4_South_Peninsular | 12857 | 0 | 11 |
| Client5_Central_India | 12857 | 0 | 16 |
| GLOBAL_POOLED | 64285 | 0 | 11 |

## Pooled high-correlation pairs

| Feature A | Feature B | Pearson r |
|---|---|---:|
| era5_tp_mm | era5land_tp_mm | 0.99990307 |
| era5_t2m_c | era5land_t2m_c | 0.99766965 |
| era5_d2m_c | era5land_d2m_c | 0.99577915 |
| era5land_t2m_c | era5land_skin_c | 0.99384517 |
| era5_t2m_c | era5land_skin_c | 0.98754296 |
| era5land_t2m_c | power_t2m_c | 0.98487857 |
| era5_t2m_c | power_t2m_c | 0.98485780 |
| era5land_soil_m3_m3_layer1 | era5land_soil_m3_m3_layer2 | 0.97992729 |
| era5land_skin_c | power_t2m_c | 0.97933162 |
| nino34_anomaly | oni_anomaly_3mo | 0.95960154 |
| rain_origin_lag2 | rain_roll3_mean | 0.95685833 |

## All 43 channels and raw units

| Index | Feature | Incremental group | Raw unit |
|---:|---|---|---|
| 0 | season_sin1 | CORE | dimensionless |
| 1 | season_cos1 | CORE | dimensionless |
| 2 | season_sin2 | CORE | dimensionless |
| 3 | season_cos2 | CORE | dimensionless |
| 4 | rain_origin_lag1 | CORE | mm (monthly total) |
| 5 | rain_origin_lag2 | CORE | mm (monthly total) |
| 6 | rain_origin_lag5 | CORE | mm (monthly total) |
| 7 | rain_origin_lag11 | CORE | mm (monthly total) |
| 8 | rain_roll3_mean | CORE | mm (monthly total) |
| 9 | rain_roll3_std | CORE | mm (monthly total) |
| 10 | rain_roll6_mean | CORE | mm (monthly total) |
| 11 | rain_roll6_std | CORE | mm (monthly total) |
| 12 | rain_roll12_mean | CORE | mm (monthly total) |
| 13 | rain_roll12_std | CORE | mm (monthly total) |
| 14 | nino34_anomaly | CLIMATE | degC anomaly |
| 15 | dmi_hadisst11_anomaly | CLIMATE | degC anomaly |
| 16 | soi_cpc_standardized | CLIMATE | dimensionless standardized index |
| 17 | oni_anomaly_3mo | CLIMATE | degC anomaly |
| 18 | meiv2_bimonthly | CLIMATE | dimensionless standardized index |
| 19 | amv_ersstv6_global_removed_1951_1980 | CLIMATE | degC anomaly |
| 20 | enso_x_iod | CLIMATE | degC2 |
| 21 | era5_t2m_c | METEOROLOGY | degC |
| 22 | era5_d2m_c | METEOROLOGY | degC |
| 23 | era5_sp_hpa | METEOROLOGY | hPa |
| 24 | era5_mslp_hpa | METEOROLOGY | hPa |
| 25 | era5_u10_m_s | METEOROLOGY | m s-1 |
| 26 | era5_v10_m_s | METEOROLOGY | m s-1 |
| 27 | era5_tp_mm | METEOROLOGY | mm month-1 |
| 28 | era5land_t2m_c | METEOROLOGY | degC |
| 29 | era5land_d2m_c | METEOROLOGY | degC |
| 30 | era5land_skin_c | METEOROLOGY | degC |
| 31 | era5land_soil_m3_m3_layer1 | METEOROLOGY | m3 m-3 |
| 32 | era5land_soil_m3_m3_layer2 | METEOROLOGY | m3 m-3 |
| 33 | era5land_soil_m3_m3_layer3 | METEOROLOGY | m3 m-3 |
| 34 | era5land_soil_m3_m3_layer4 | METEOROLOGY | m3 m-3 |
| 35 | era5land_tp_mm | METEOROLOGY | mm month-1 |
| 36 | era5land_evaporation_mm_signed | METEOROLOGY | mm month-1 (negative evaporation) |
| 37 | era5land_surface_runoff_mm | METEOROLOGY | mm month-1 |
| 38 | era5land_subsurface_runoff_mm | METEOROLOGY | mm month-1 |
| 39 | power_t2m_c | METEOROLOGY | degC |
| 40 | power_rh2m_pct | METEOROLOGY | percent |
| 41 | power_ws10m_m_s | METEOROLOGY | m s-1 |
| 42 | power_prectotcorr_mm_day | METEOROLOGY | mm day-1 (rate) |

## Interpretation limits

- Pooled correlations mix regional distributions; the five clients share calendar/index observations and are not 1495 independent time observations.
- Pearson correlation is descriptive and contemporaneous within eligible latest-row inputs, without p-values, causal claims or predictive selection.
- This centralized offline TRAIN audit accesses client feature rows; it is not a core federated model message or included in training communication costs.
- Raw units for the 22 meteorological channels are explicit in the canonical manifest; calendar/rain/climate units are derived from authenticated transformations and previously verified source definitions.
- Prepared float32 encoding permits tiny deviations of global standardized variance from one.
- No raw target, heavy-event label or evaluation payload member was loaded. Target-scaling statistics present in the authenticated schema were not used.

The already frozen group-selection protocol is unchanged. These correlation descriptions do not measure forecasting utility.

Reproduce using the supplied diagnose_train_features.py with --package-root, --run runs/revision_v2 and --output pointing to a separate owned audit directory.
