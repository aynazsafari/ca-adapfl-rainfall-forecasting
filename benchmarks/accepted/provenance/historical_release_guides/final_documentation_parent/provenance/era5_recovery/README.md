This bounded recovery is limited to ERA5 months 2022-04, 2022-05, 2022-07,
2022-08, 2024-01 and 2024-02, plus March 2024 as a provider compatibility control.
The original Earth Engine rows remain rejected and unchanged. Its independent
per-band diagnostic has complete state records and 21 missing precipitation
validity records across the six months.

The pinned Earthmover Icechunk snapshot is ZFKDHBCTBVHVXM3BQFV0, group
single/temporal. Frozen geometry retains 3,287 native quarter-degree cell
centers, including the C1/C5 overlap. Whole-cell WGS84 quadrilateral areas use
the same retained country geometry SHA256 and half-open client boxes as the
accepted January 1981 comparison. Source latitude, longitude, time coordinates,
serialized metadata and quality flags are preserved. Every selected required
hour must have status0 and expver1; this is final retrospective reanalysis and
does not reconstruct historical operational data availability.

Only 644 native compressed chunks are needed: two temporal chunks times 46
spatial tiles times seven variables, exactly 1,465,963,879 bytes. Source object
URLs, offsets, lengths, HTTP range metadata and SHA256 accompany immutable
compressed files. The pipeline caps climate bytes at 2GB and network workers at
two; numerical aggregation uses one thread and processes one variable at a time.

Instantaneous variables average validity times from the first day00UTC through
the final day23UTC. Hourly precipitation sums one-hour amounts ending from the
first day01UTC through the next month's first day00UTC. No daily integration,
deaccumulation or clipping is applied. Native float32 hourly subsets are saved
per month, with cell coordinates, integer native indices and normalized client
area weights. Independent reduction first averages/sums each cell over time,
then uses scalar math.fsum over cells; the main calculation weights each
spatial tile for each hour and then reduces over time.

Commands use the shared interpreter:

```
work/env/Scripts/python.exe work/era5_rebuild/recovery_6months/recover_selected_months.py
work/env/Scripts/python.exe work/era5_rebuild/recovery_6months/recover_selected_months.py --execute
work/env/Scripts/python.exe work/era5_rebuild/recovery_6months/compare_provider_control.py
```

The first command inventories exact source bytes without downloading climate
payloads. The second retrieves and verifies the fixed subset. The third checks
all 35 March2024 regional native-unit values, areas, counts and time bounds
against the genuine full Earth Engine table. Absolute tolerances were frozen
before recovery or forecast outcomes using the accepted January check:
1e-4K temperature,0.01Pa pressure,1e-4m/s wind and1e-5m monthly precipitation.
Any failed check prevents provider admission. Measured small differences do
not establish their encoding cause or bitwise provider equivalence.

Completed checks passed: all 245 independent monthly/regional reductions,
35 March provider values and 60 metadata comparisons. Maximum independent
reduction difference is 1.7462298274040222e-10 in native units. March maximum
differences are 9.83e-7K temperature,0.000336Pa pressure,3.06e-7m/s wind and
9.70e-7m monthly precipitation, all below the frozen bounds. All16,882,032
selected precipitation hour/cell values are finite and nonnegative. All source
flags are0 and expver1. Thirty of the35 monthly source rows are authorized
repairs; the five March control rows must not replace the genuine EE control.
