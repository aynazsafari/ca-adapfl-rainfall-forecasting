# GLDAS physical processing interval versus GEE asset properties

The live root inspection reports GEE `system:time_start=2000-01-01T03:00Z`,
`start_hour=3,end_hour=6`; the06UTCasset has6/9 and a midnight asset has0/3.
These custom fields therefore appear to express a nominal forward asset
duration. Their ingestion derivation has not been independently established.
They must not be described as observed physical `time_bnds` values.

Two independent primary documentation sources explicitly define the physical
GLDAS fields as backward intervals:

* NASA README revised2025, Noah Table3.3 pp21–23: `_tavg` averages and `_acc`
  accumulates the preceding3hours; `_inst` is instantaneous.
  [NASA README](https://docserver.gesdisc.eosdis.nasa.gov/public/project/hydrology/README_GLDAS2.pdf).
* Google's own catalog source explicitly states that `_tavg` and `_acc` cover
  the past3hours and `_inst` is instantaneous. The same file only calls the two
  custom properties “Start hour” and “End hour”, without defining their physical
  role or provenance.
  [Official Google catalog source](https://github.com/google/earthengine-catalog/blob/main/catalog/NASA/NASA_GLDAS_V021_NOAH_G025_T3H.jsonnet#L54).

NASA CMR UMM-G confirms the exact source granule
`GLDAS_NOAH025_3H.A20000101.0300.021.nc4`, with both temporal extent timestamps
equal03UTC. This confirms its label, not its averaging bounds.
[Exact NASA granule metadata](https://cmr.earthdata.nasa.gov/search/concepts/G1414071265-GES_DISC.umm_json).

The public NASA summary lists `time` units as minutes since2000-01-01 03:00
and includes `time_bnds`, but supplies no numeric bounds. The exact raw
`time_bnds` values could not be independently obtained anonymously. The old
hydro1 OPeNDAP endpoint returns410 (retired). NASA CMR supplies the replacement
cloud OPeNDAP endpoint below; anonymous `.das`, `.dds` and
`.ascii?time,time_bnds` requests redirect to EarthdataLogin and end401. No
credentials were accessed, no alternative product was substituted and no bounds
were invented.
[Current NASA endpoint](https://opendap.earthdata.nasa.gov/collections/C1342986035-GES_DISC/granules/GLDAS_NOAH025_3H.2.1%3AGLDAS_NOAH025_3H.A20000101.0300.021.nc4).

The scientifically documented source contract therefore supports the current
flow window: validity03UTC on the first day through00UTC on the following
month's first day. Rate sums×10800 and accumulation sums then cover the month.
Switching to a forward window solely from the unexplained custom properties
would contradict both producer and consumer physical documentation. This is a
documented-contract interpretation; it is not a claim that raw bounds or
Google's ingestion implementation have been observed. The exact origin of the
custom-property discrepancy remains unresolved and should remain recorded.

Primary source bytes and hashes are retained in `source_evidence` and the
independent research child's audit directory. Full exports were held while
the separate coastal spatial-mask failure was investigated.
