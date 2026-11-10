# Data sources and attribution

This benchmark contains regional aggregations, corrected calendar alignments
and derived features. These products are identified separately from the
providers' native archives in `data/manifest.json` and the source receipts.
Retrieval through Earth Engine, NOAA or an AWS mirror does not transfer data
ownership to the benchmark authors. No provider endorsement is implied.

Provider pages were checked on 8 October 2026. This record does not change the
frozen data or upgrade CHIRPS v2 to the newer CHIRPS v3 product.

| Input | Provider and terms evidence | Attribution and release treatment |
|---|---|---|
| CHIRPS v2 monthly final | [UCSB Climate Hazards Center](https://chc.ucsb.edu/data/chirps) states that CHIRPS data are public domain. | Credit UCSB CHC and the CHIRPS scientific publication. The supplied regional panel is a documented rebuild, not the missing original export. |
| ERA5 and ERA5-Land | The [ERA5](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY) and [ERA5-Land](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR) catalogs point to the [Copernicus product licence](https://apps.ecmwf.int/datasets/licences/copernicus/), which permits distribution and adaptation with attribution. The current [CDS ERA5-Land page](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) also identifies a CC-BY licence. | Credit ECMWF and C3S, cite the source products, identify conversion and regional aggregation, and retain the attribution below. Record Earth Engine and the pinned Earthmover AWS archive as access paths, with their exact provenance. |
| NASA POWER | [NASA Earthdata use and citation guidance](https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy) describes NASA-led data as CC0 unless a restriction is marked and asks users to cite source datasets. [POWER](https://power.larc.nasa.gov/) identifies its solar and meteorological sources. | Acknowledge the NASA POWER Project and distinguish MERRA-2 meteorology, SRB and CERES SYN radiation. Regional reduction and unit conversion were performed for this benchmark. NASA's general policy does not override any separately marked third-party restriction. |
| GLDAS-2.1 Noah | The [product catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H) identifies NASA GES DISC distribution and the NASA data policy. | Credit NASA GES DISC and GLDAS, cite Rodell et al. and the product identifier. This is an optional, separately matched sensitivity cohort. |
| NOAA climate indices and ERSSTv6 | [NOAA guidance](https://www.noaa.gov/office-education/outreach-communication/faq) distinguishes NOAA public-domain material from third-party content and requests source credit. | Identify CPC, PSL and NCEI for each retained series. The AMV reconstruction is a benchmark derivation from ERSSTv6 and is not an official NOAA AMO series. HadISST-based DMI source lineage is retained separately; no blanket NOAA licence is assigned to all upstream third-party data. |
| Natural Earth 5.1.1 country geometry | [Natural Earth terms](https://www.naturalearthdata.com/about/terms-of-use/) place the map data in the public domain. | Made with Natural Earth. The frozen geometry is cartographic and uses its de facto boundary convention. Five study boxes neither partition nor cover all India. |
| IMD daily gridded rainfall | The original files are supplied research materials from IMD; official product links and all 44 file hashes are retained in the local source evidence. An explicit redistribution licence for this supplied snapshot was not established by the checked pages. | Credit IMD and the gridded-rainfall dataset publication. Do not assign CC0 or an authors' software licence to this source. Raw annual NetCDF files are outside the core repository; include optional derived IMD material in a public release only under its applicable source terms. The local matched experiment remains documented and reproducible from the source pointers. |

Contains modified Copernicus Climate Change Service information (2026).
Neither the European Commission nor ECMWF is responsible for uses of these
derived data. Regional means, unit conversions and six repaired ERA5 months
are documented modifications; provider-native values are not described as
author observations.

The software's authorship and licence are separate from dataset terms. No
software licence has been selected on the authors' behalf. Large raw source
archives, authentication material, downloaded literature PDFs, renderer
executables, local environments and study checkpoints are excluded from the
small source release. Exact study-result artifacts and their hashes belong
to a separate results release after completion and numerical/visual QA.
