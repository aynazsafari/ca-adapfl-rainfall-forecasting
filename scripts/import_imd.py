#!/usr/bin/env python3
"""Import user-obtained frozen IMD native files without downloading or fitting.

Example: python tools/import_imd.py --native-dir /private/IMD --repo-root .
Use --output-dir NEW_DIRECTORY for an isolated validation. Existing files are
never overwritten. Native import uses numpy/pandas; shapely/pyproj are needed
only when frozen geography-only support has not been packaged. Exact source
bytes were validated with numpy2.4.4/pandas3.0.2. Use the reference model
environment separately for --rebuild-prepared (numpy2.3.5/pandas3.0.1).

Native identities and source-output hashes are public metadata, not source
values. An optional provenance/imd_native_files.json may supply those hashes;
the immutable defaults below ensure it cannot silently admit a new snapshot.
An optional provenance/imd_admission_receipt_template.json contains the original
receipt with invalid_source_rows removed. Reconstructing its omitted row allows
byte authentication without distributing that row in the public repository.
Without that template, the importer creates an honestly identified new receipt.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import platform
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NATIVE_HASHES = """
1981 554a4aea83b1c50344ef7b90d058fd63f390542045f32774633facb7bd639a04
1982 e6a21b89c097af05219d8cc9eeb345d792ecbc7f521b087aeb183e39db94c5c8
1983 9f943927a346a09ecba08d2bd3faa9bf20eaeee3a10520ea436a1ddee0e03438
1984 236b7e747e06553b058b038be381e3d0cf9f9c640e0ce6832ccba08b728cd4ce
1985 d89fee68898501abe1f943caefeb29297466419826304fccd4cfe6f7b9f315f3
1986 8aceeab94d1ea259e9088747c27e4616c2567a3cbf949cc769b7b0502435f949
1987 9930656c8f4c6b6623dc29342dcbf4afdbc0ab04ed311fa7ddb659b8d1423f82
1988 54a18b3d93ac466cfe1a7208080698fe12c63306e4a359228273e9d428e3e290
1989 e4804b5f5c44c463688f7bf2680e50d78b04ddbd53f9ffac7ff77a542f237540
1990 ba93db1499b6de8a079ea3ece83cdd2ea3b3c4b27842a18c0ed3caa1d113f2cf
1991 fd5f8309d02f0875fbb5b43663d7d7f38a5d1e7911ac4b72dc80897f82895377
1992 e5d0c389573e7d006923b9ef16bc67f631061a10da03b25949551a03dff7da61
1993 2fedcb184f08989672037d06485666b18165f244aead36c46e8a6a8df58b8f3c
1994 153f742dc9596d413af259e114d83aa415c0ba3ef7f22ed3f60d8854d71a92f3
1995 ee769ab54c2d02e7f6db23fd440ceef107f4a3d79308ee34d5ad1015df813e0c
1996 606d5eaa75bf96cf698330ae137a42f244b1e0850178f14460c3f8d3a47ec589
1997 0edb75fb4fb6f215811e34480d09e1d8d6487d7a0166cedb3d05b38f2e439724
1998 dbcc01ec2d44f22f94d74009da38c7f936920534bf6d7206ea73fc1bf5c0fd7d
1999 636961db4399dbb3c51caba7c815ccfecb361cd271cce767825862e2e6920884
2000 eee32b28de22dee336d1b414a8b6a3f420e3dfcfb7ca6186d56a3a02dd344782
2001 f3e0536f688908cfbcd5bc8272362c4bf238aac198e607220071867899b7f175
2002 4c61fdae1ea69b733cdfe2a27ebadbfa79f11aea2ae5ef4cf2e99624f7f1efdb
2003 f215e6182fb15f1e2b59da09e408fa10d23cca6c3041aea54adb878d784b64d2
2004 1ab45a65b10277e415dc06965008156637777a79b86367cf45329b2718545935
2005 c6de4f7b35238f6c9fca22950934390ac8f08f630fe0bfb3c90cc62849f9cc50
2006 8caf5b4fa2900a2be070f87b760a84297d686e74ee0dacded6c2944451e1d9f1
2007 f4430feb443d8f8e8d8882e8c52ebe738f20aefea269bd7a10aa14e5da799ef8
2008 345079a1a4c84a3d0ca198f46693384767e9639c05f131e40de4d5a649931279
2009 109c02519016c3e12e248d8032f6baebc4a3635a163dbf13d374b582f43e0486
2010 fe7ea2aba9dab02865684040e9c1abfd1402481feac214c78b5bd94cf881b84b
2011 925e06db22b6797afd393b781e834ee8016cf8850b2501f73dbb13e6c2c936de
2012 7708a4846e8ef0664962c7042e886f3a5f055af365d536782eff63760dc3f666
2013 2042932a2aea8288bbf45482272f65b9225e8aafa9a44cf06c65962007f8bc79
2014 9ecca0e9049f0e6cf366a44420e2e7361bff67bf97ae43c6502999b25da76457
2015 c276ce7ebff1d74d79123a7c777d9ef61c3c5765aa3b6db464ff6224b30a3926
2016 e5b0ba1feb2bf10a3347091867237b1c480a71cbd8087037ef846e4346cbcdef
2017 49786e2d2b661c5d3bcfb3ffd90385c1133ec27a8a04bf272ff2df5029106a9c
2018 26bd53aeb2d6f3f7d39516c41606db005dede474906b33462d1a0caef69cd6ec
2019 c551c563b3514d492a2de94c706c1c254d5826656738b4a8e7dbd83841344a5b
2020 2e567f0f71f3a1936ba11d61a0214fc3927c7b969ee5669d14959aa0b6b026dc
2021 fb0f291fa7e214a7811023a9309cead67233b6b4e0bfcc5b277643f976f68374
2022 83ae0d4e48f3a408eec3b54844476d406faf70928dc13c79f0bd085e3b0224b8
2023 1fa0cbcb56769fd3cd2702e36dc3ee1b81b74755b77c7f058c70dfc3afb82831
2024 1ef02aeba5694dbb57a6cca23a3c2cc11740affb185137c1eacbeab59893228a
"""
EXPECTED_FILES = {f"RF25_ind{y}_rfp25.nc": h for y, h in
                  (line.split() for line in NATIVE_HASHES.strip().splitlines())}
GEOMETRY_SHA256 = "4cdbe7cba953c55e4c6591b779128484c2e203ff0c3433b1826bafca95a4d090"
SUPPORT_SHA256 = "aad62872a5a0bf7b60deb55eb6ee63d19d16dda0555bc0cebe1b11050bd7fe58"
SOURCE_SHA256 = "eac5b7555a70263d289d118f81ddfc54de9415bab535a6fc96abf4da304b4fee"
EXPECTED_OUTPUTS = {
    "imd_supplement.csv": "93adacbec2a1aa4c240f756c02d440ad4fd8a27f6795731653c5d700f28e5933",
    "imd_supplement_quality.csv": "a968989b0194b5671f1c4a9f803215abcd18952ee0b6d2d90277faba42b9e7a9",
    "manifest.json": "8c14f84a86e41cd3a847d26692a3f0bee2a2a4e2e33c6504202c01ac1b49af76",
}
EXPECTED_CLIENTS = {
    "Client1_NW_Arid": [22., 30., 68., 76.],
    "Client2_Indo_Gangetic": [24., 31., 76., 88.],
    "Client3_NE_Himalaya": [23., 29., 88., 97.5],
    "Client4_South_Peninsular": [8., 15.5, 75., 80.5],
    "Client5_Central_India": [18., 24., 74., 83.],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


class NC3:
    """Bounded classic NetCDF decoder copied from the independent audit logic."""
    TYPES = {1: ("i1", 1), 2: ("S1", 1), 3: (">i2", 2),
             4: (">i4", 4), 5: (">f4", 4), 6: (">f8", 8)}

    def __init__(self, data):
        self.data, self.pos = data, 0
        self.magic = self.take(4)
        require(self.magic in (b"CDF\x01", b"CDF\x02"), "Expected classic NetCDF CDF1/CDF2")
        self.nrecords = self.u32()
        self.dims = self.items(10, self.dim)
        self.global_attrs = self.attrs()
        self.variables = self.items(11, self.var)
        self.record_size = sum(v["vsize"] for v in self.variables if v["record"])

    def take(self, count):
        result = self.data[self.pos:self.pos + count]
        self.pos += count
        require(len(result) == count, "Truncated NetCDF")
        return result

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def string(self):
        count = self.u32()
        value = self.take(count).decode()
        self.take((-count) % 4)
        return value

    def items(self, tag, reader):
        actual, count = self.u32(), self.u32()
        require((actual == 0 and count == 0) or actual == tag, "Changed NetCDF list tag")
        return [reader() for _ in range(count)]

    def dim(self):
        return {"name": self.string(), "length": self.u32()}

    def attr(self):
        name, kind, count = self.string(), self.u32(), self.u32()
        dtype, size = self.TYPES[kind]
        value = self.take(count * size)
        self.take((-count * size) % 4)
        return name, value.decode() if kind == 2 else np.frombuffer(value, dtype=dtype).tolist()

    def attrs(self):
        return dict(self.items(12, self.attr))

    def var(self):
        name, count = self.string(), self.u32()
        ids = [self.u32() for _ in range(count)]
        attrs, kind, size = self.attrs(), self.u32(), self.u32()
        begin = self.u32() if self.magic == b"CDF\x01" else struct.unpack(">Q", self.take(8))[0]
        record = bool(ids and self.dims[ids[0]]["length"] == 0)
        dims = [self.dims[i]["name"] for i in ids]
        shape_ = [self.nrecords if self.dims[i]["length"] == 0 else self.dims[i]["length"] for i in ids]
        return dict(name=name, dimensions=dims, shape=shape_, attrs=attrs, type=kind,
                    vsize=size, begin=begin, record=record)

    def array(self, var):
        dtype, size = self.TYPES[var["type"]]
        if not var["record"]:
            return np.ndarray(var["shape"], dtype=dtype, buffer=self.data, offset=var["begin"])
        strides = [self.record_size] + [int(np.prod(var["shape"][i + 1:])) * size
                                       for i in range(1, len(var["shape"]))]
        return np.ndarray(var["shape"], dtype=dtype, buffer=self.data,
                          offset=var["begin"], strides=strides)


def accepted_root(repo):
    repo = Path(repo).resolve()
    nested = repo / "benchmarks/accepted"
    return nested if nested.is_dir() else repo


def geometry(root, lat, lon):
    source = root / "provenance/source_tables/india_country_ne_5_1_1.geojson"
    require(sha_file(source) == GEOMETRY_SHA256, "Frozen public country geometry differs")
    protocol = json.loads((root / "provenance/source_tables/extraction_protocol.json").read_text())
    require(protocol["clients"] == EXPECTED_CLIENTS, "Frozen client boxes differ")
    support = root / "provenance/imd_native_support.npz"
    if support.exists():
        # This artifact contains geographic coordinates, cell areas and masks;
        # it contains no provider rainfall values or observation support arrays.
        require(sha_file(support) == SUPPORT_SHA256, "Frozen native geography support differs")
        with np.load(support, allow_pickle=False) as saved:
            require(np.array_equal(lat, saved["latitude"]) and np.array_equal(lon, saved["longitude"]), "Frozen geographic coordinates differ")
            return saved["geodesic_cell_area_m2"].copy(), {c: saved[c].copy() for c in EXPECTED_CLIENTS}
    import pyproj
    import shapely
    from pyproj import Geod
    from shapely.geometry import Point, shape
    from shapely.prepared import prep
    region = shape(json.loads(source.read_text())["geometry"])
    require(region.is_valid, "Invalid country geometry")
    xx, yy = np.meshgrid(lon, lat)
    prepared = prep(region)
    country = np.array([prepared.covers(Point(x, y)) for x, y in
                        zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
    geod, x = Geod(ellps="WGS84"), float(lon[0])
    row_areas = np.array([abs(geod.polygon_area_perimeter(
        [x - .125, x + .125, x + .125, x - .125],
        [y - .125, y - .125, y + .125, y + .125])[0]) for y in lat])
    areas = np.repeat(row_areas[:, None], len(lon), axis=1)
    masks = {c: country & (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
             for c, (y0, y1, x0, x1) in EXPECTED_CLIENTS.items()}
    require(all(mask.any() for mask in masks.values()), "Empty client mask")
    return areas, masks


def reconstruct(native, root):
    files = {p.name: p for p in Path(native).glob("*.nc")}
    require(set(files) == set(EXPECTED_FILES), "Provide exactly the 44 frozen RF25_ind1981..2024_rfp25.nc files")
    ledger = root / "provenance/imd_native_files.json"
    if ledger.exists():
        supplied = json.loads(ledger.read_text())
        records = supplied["files"] if isinstance(supplied, dict) else supplied
        found = {r.get("file", r.get("path")): r["sha256"] for r in records}
        require(found == EXPECTED_FILES, "Public native hash ledger differs from immutable accepted snapshot")
    # Authenticate every file before reading any source value or writing outputs.
    observed = []
    for name, expected in EXPECTED_FILES.items():
        path = files[name]
        require(sha_file(path) == expected, f"Native SHA256 differs: {name}")
        observed.append(dict(file=name, bytes=path.stat().st_size, sha256=expected))
    rows, audit_rows = [], []
    lat_ref = lon_ref = None
    for name in EXPECTED_FILES:
        nc = NC3(files[name].read_bytes())
        variables = {v["name"]: v for v in nc.variables}
        require(set(variables) == {"LONGITUDE", "LATITUDE", "TIME", "RAINFALL"}, "Changed variables")
        arrays = {name_: nc.array(var) for name_, var in variables.items()}
        lat, lon = arrays["LATITUDE"], arrays["LONGITUDE"]
        rain = np.array(arrays["RAINFALL"], copy=True)
        year = int(name[8:12])
        dates = pd.DatetimeIndex(pd.Timestamp("1900-12-31") + pd.to_timedelta(arrays["TIME"], unit="D"))
        require(dates.equals(pd.date_range(f"{year}-01-01", f"{year}-12-31")), f"Changed calendar: {name}")
        require(variables["RAINFALL"]["attrs"]["units"] == "mm", "Changed rainfall units")
        require(variables["TIME"]["attrs"]["units"] == "days since 1900-12-31 00:00:00", "Changed time units")
        require(variables["RAINFALL"]["dimensions"] == ["TIME", "LATITUDE", "LONGITUDE"], "Changed dimension order")
        if lat_ref is None:
            require(np.all(np.diff(lat) == .25) and np.all(np.diff(lon) == .25), "Changed native grid")
            lat_ref, lon_ref = lat.copy(), lon.copy()
            areas, masks = geometry(root, lat, lon)
        require(np.array_equal(lat, lat_ref) and np.array_equal(lon, lon_ref), "Changed annual grid")
        valid = np.isfinite(rain) & (rain != -999)
        require(not ((rain < 0) & valid).any(), "Negative nonmissing native rainfall")
        require(np.isfinite(rain).all(), "Nonfinite native rainfall")
        for month in range(1, 13):
            block = rain[dates.month == month]
            complete = valid[dates.month == month].all(axis=0)
            independent_totals = np.sum(block, axis=0, dtype=np.float64)
            audit_totals = np.full(complete.shape, np.nan, dtype=np.float64)
            audit_totals[complete] = block[:, complete].sum(axis=0, dtype=np.float64)
            for client, mask in masks.items():
                selected = mask & complete
                weights = areas[selected]
                require(len(weights) > 0, "Empty complete source support")
                mean = float(np.dot(independent_totals[selected], weights) / weights.sum())
                audit_mean = float(np.sum(audit_totals[selected] * weights) / float(weights.sum()))
                key = dict(client=client, month=f"{year}-{month:02d}")
                rows.append({**key, "imd_monthly_mm": mean, "complete_cells": int(selected.sum()),
                             "weighted_coverage": float(weights.sum() / areas[mask].sum())})
                audit_rows.append({**key, "imd_monthly_mm": audit_mean})
        print(f"Authenticated and reconstructed {name}", flush=True)
    independent = pd.DataFrame(rows).sort_values(["client", "month"])
    audit = pd.DataFrame(audit_rows).sort_values(["client", "month"])
    # Preserve the original audit15-significant-digit CSV and pandas C-parser
    # rounding before computing the independent17-digit diagnostic difference.
    csv15 = audit.to_csv(index=False, float_format="%.15g", lineterminator="\r\n")
    parsed = pd.read_csv(io.StringIO(csv15))
    independent["saved_value_delta_mm"] = independent.imd_monthly_mm.to_numpy() - parsed.imd_monthly_mm.to_numpy()
    source = independent.to_csv(index=False, float_format="%.17g", lineterminator="\r\n").encode("utf-8")
    require(len(independent) == 2640, "Incomplete reconstructed panel")
    return source, observed


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def admitted(source, root):
    features, quality, invalid = [], [], []
    for row in csv.DictReader(io.StringIO(source.decode())):
        mean, coverage, count = float(row["imd_monthly_mm"]), float(row["weighted_coverage"]), int(row["complete_cells"])
        ok = coverage >= .95
        feature = dict(client=row["client"], month=row["month"], imd_rainfall_mm=format(mean, ".17g") if ok else "")
        q = dict(client=row["client"], month=row["month"], physical_complete_cell_mean_mm=format(mean, ".17g"),
                 complete_cells=count, weighted_coverage=format(coverage, ".17g"), feature_admitted=int(ok),
                 quality_reason="PASS" if ok else "weighted_area_coverage_below_0.95",
                 saved_value_delta_mm=row["saved_value_delta_mm"])
        features.append(feature)
        quality.append(q)
        if not ok:
            invalid.append(q)
    require({(r["client"], r["month"]) for r in invalid} == {("Client1_NW_Arid", "2023-10")}, "Changed invalid-source pattern")
    payload = {"imd_supplement.csv": csv_bytes(features, list(features[0])),
               "imd_supplement_quality.csv": csv_bytes(quality, list(quality[0]))}
    template = root / "provenance/imd_admission_receipt_template.json"
    source_exact = sha_bytes(source) == SOURCE_SHA256 and all(sha_bytes(data) == EXPECTED_OUTPUTS[name] for name, data in payload.items())
    if template.exists() and source_exact:
        receipt = json.loads(template.read_text())
        require(receipt.get("invalid_source_rows") in (None, []), "Receipt template contains omitted provider values")
        receipt["invalid_source_rows"] = invalid
        receipt["source_sha256"] = sha_bytes(source)
        receipt["outputs"] = {name: dict(sha256=sha_bytes(data), bytes=len(data)) for name, data in payload.items()}
    else:
        receipt = dict(status="RECONSTRUCTED_SOURCE_NEW_RECEIPT_IDENTITY", source_sha256=sha_bytes(source),
                       reconstructed_rows=len(features), admitted_feature_rows=len(features) - len(invalid),
                       invalid_source_rows=invalid,
                       source_mean_definition="monthly rainfall complete-day/complete-cell weighted regional mean from the independent native IMD reconstruction",
                       feature_name="imd_rainfall_mm", unit="mm month-1", minimum_weighted_area_coverage=.95,
                       physical_mean_location="imd_supplement_quality.csv", missing_numeric_encoding="empty CSV field (NA); no filling",
                       feature_support=dict(sequence_length=12, row_origin_month="s", target_month="t=s+1",
                                            met_source_per_row="s-3=t-4", sequence_oldest_source="t-15", sequence_latest_source="t-4",
                                            interpretation="prespecified conservative availability scenario applied to revised archives; no historical real-time vintage claim"),
                       outputs={name: dict(sha256=sha_bytes(data), bytes=len(data)) for name, data in payload.items()},
                       generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # The accepted Windows receipt used CRLF and no final newline.
    payload["manifest.json"] = json.dumps(receipt, indent=2, allow_nan=False).replace("\n", "\r\n").encode()
    return payload


def rebuild_prepared(root, imported, destination):
    """Rebuild omitted IMD feature archives from authenticated imported inputs.

    This uses accepted preparation interfaces and performs zero model fitting,
    checkpoint loading, evaluation scoring or network requests.
    """
    expected = {
        "IMD_ADDED_FULL_fit.npz": "07822be743330dab4cbee5d4ac858e0ec1774280f1ea2c243fb510594f95b501",
        "IMD_ADDED_FULL_evaluation_features.npz": "78c08cb80ccb2add2eb861a04578c01fda047b1549a9c7ba9262efc453373c1f",
        "IMD_ADDED_FULL_schema.json": "48bfa75f9f3d82f872df3779a0c0a899f5c36835ea83b6de36a8f731585c4a16",
    }
    data_dir = destination / "runs/supplements_v2/datasets"
    require(not any((data_dir / name).exists() for name in expected), "Existing prepared evidence must never be overwritten")
    code_hashes = {
        "caadapfl/features.py": "0381556297d12e669dcf7aee191c611e259d1c6385315e61fa08d688653bd28f",
        "caadapfl/runner.py": "79be46b566da5070ce785519cfe3674748295e1756abd7e2d1fc2d1a01b76047",
    }
    for name, digest in code_hashes.items():
        require(sha_file(root / name) == digest, f"Accepted preparation code differs: {name}")
    for name in ("imd_supplement.csv", "imd_supplement_quality.csv"):
        require(sha_file(imported / "tools/derived/imd" / name) == EXPECTED_OUTPUTS[name], f"Imported admitted source differs: {name}")
    manifest = json.loads((root / "data/manifest.json").read_text())
    for name in ("monthly", "indices"):
        require(sha_file(root / "data" / (name + ".csv")) == manifest["inputs"][name]["sha256"], f"Changed canonical {name}")
    plan_path = root / "tools/derived/cohorts/IMD_cohort_plan.json"
    require(sha_file(plan_path) == "c131916d44aeeaf1125f690e63b392f9d041ef862bf7a3379a1a4a3f921ddd50", "Changed frozen IMD cohort")
    plan = json.loads(plan_path.read_text())
    sys.path.insert(0, str(root))
    from caadapfl import features as f, runner as r
    met = tuple(manifest["feature_contract"]["met_columns"])
    require(len(met) == 22, "Changed primary feature contract")
    panel = f.checked_panel(root / "data/monthly.csv", manifest["inputs"]["monthly"]["sha256"], ["rainfall_mm", *met])
    indices = pd.read_csv(root / "data/indices.csv")
    extra = pd.read_csv(imported / "tools/derived/imd/imd_supplement.csv")
    extra["month"] = pd.PeriodIndex(extra["month"], freq="M")
    extra = extra.set_index(["client", "month"]).sort_index()
    current = panel.join(extra[["imd_rainfall_mm"]], validate="one_to_one")
    rows, feature_set = f.build_rows(current, indices, met_columns=(*met, "imd_rainfall_mm"), with_climate=True)
    raw = f.samples(current, rows, feature_set, target_months=tuple(plan["target_months"]),
                    expected_evaluation_months=tuple(plan["expected_evaluation_months"]))
    scaled, stats = f.train_transform(raw, feature_set)
    declared = {k: v for k, v in plan["role_counts_per_client"].items() if k != "embargo"}
    require(all(v["role_counts"] == declared for v in stats["clients"].values()), "Changed matched role support")
    data_dir.mkdir(parents=True, exist_ok=True)
    # Preparation writes only the three omitted input artifacts. It does not
    # open saved predictions or modify choices, completed runs or their receipts.
    r._array_archive(data_dir / "IMD_ADDED_FULL_fit.npz", raw, scaled, feature_set)
    r._array_archive(data_dir / "IMD_ADDED_FULL_evaluation_features.npz", raw, scaled, feature_set, evaluation=True)
    schema = dict(names=feature_set.names, climate_names=feature_set.climate_names, support=feature_set.support,
                  met_columns=feature_set.met_columns, extra_lag=0, statistics=stats)
    schema_bytes = json.dumps(schema, indent=2, sort_keys=True, allow_nan=False).replace("\n", "\r\n").encode()
    with (data_dir / "IMD_ADDED_FULL_schema.json").open("xb") as stream:
        stream.write(schema_bytes)
    checks = {name: dict(expected_sha256=h, observed_sha256=sha_file(data_dir / name),
                         byte_identical=sha_file(data_dir / name) == h, bytes=(data_dir / name).stat().st_size)
              for name, h in expected.items()}
    report = dict(status="ACCEPTED_FROZEN_PREPARED_BYTES_RESTORED" if all(c["byte_identical"] for c in checks.values()) else "REBUILT_PREPARED_NEW_IDENTITY",
                  output_checks=checks, numpy=np.__version__, pandas=pd.__version__, empirical_fits=0,
                  inference_calls=0, network_requests=0, original_receipts_modified=False)
    receipt_path = destination / "provenance/imd_import/prepared_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(status=report["status"], receipt=str(receipt_path),
                          outputs_exact={name: c["byte_identical"] for name, c in checks.items()}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="New isolated output directory; default installs missing source files into accepted benchmark")
    parser.add_argument("--rebuild-prepared", action="store_true", help="Rebuild the three omitted IMD prepared files with the reference model environment")
    parser.add_argument("--import-dir", type=Path, help="Directory containing imported source files; defaults to accepted benchmark root")
    args = parser.parse_args()
    root = accepted_root(args.repo_root)
    destination = args.output_dir.resolve() if args.output_dir else root
    if args.output_dir:
        require(not destination.exists(), "Use a fresh nonexistent output directory")
    if args.rebuild_prepared:
        require(args.native_dir is None, "--rebuild-prepared uses imported sources, not native files")
        rebuild_prepared(root, args.import_dir.resolve() if args.import_dir else root, destination)
        return
    require(args.native_dir is not None, "Native import requires --native-dir")
    targets = [Path("data") / name for name in ("imd_supplement.csv", "imd_supplement_quality.csv")]
    targets += [Path("tools/derived/imd") / name for name in EXPECTED_OUTPUTS]
    require(not any((destination / path).exists() for path in targets), "Existing source files must never be overwritten")
    source, native_records = reconstruct(args.native_dir.resolve(), root)
    payload = admitted(source, root)
    comparisons = {name: dict(expected_sha256=EXPECTED_OUTPUTS[name], observed_sha256=sha_bytes(data),
                             byte_identical=sha_bytes(data) == EXPECTED_OUTPUTS[name], bytes=len(data))
                   for name, data in payload.items()}
    exact = sha_bytes(source) == SOURCE_SHA256 and all(x["byte_identical"] for x in comparisons.values())
    status = "ACCEPTED_FROZEN_BYTES_RESTORED" if exact else "RECONSTRUCTED_NEW_SOURCE_IDENTITY"
    for path in targets:
        (destination / path).parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation, including during install, protects existing evidence.
        with (destination / path).open("xb") as stream:
            stream.write(payload[path.name])
    audit_dir = destination / "provenance/imd_import"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "independent_monthly_reconstruction.csv").write_bytes(source)
    report = dict(status=status, native_files=native_records, native_bytes=sum(r["bytes"] for r in native_records),
                  independent_source_expected_sha256=SOURCE_SHA256, independent_source_observed_sha256=sha_bytes(source),
                  output_checks=comparisons, geometry_sha256=GEOMETRY_SHA256, source_rows=2640, admitted_rows=2639,
                  environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                   geographic_support="frozen geography-only NPZ" if (root / "provenance/imd_native_support.npz").exists() else "shapely/pyproj native reconstruction",
                                   operating_system=platform.platform()),
                  empirical_fits=0, network_requests=0, existing_evidence_overwritten=False,
                  identity_policy="Only observed byte-identical hashes restore the frozen identity. Otherwise new source identity is required; original receipts are not changed.")
    report_path = audit_dir / "receipt.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(status=status, receipt=str(report_path), source_exact=sha_bytes(source) == SOURCE_SHA256,
                          outputs_exact={name: check["byte_identical"] for name, check in comparisons.items()}), indent=2))


if __name__ == "__main__":
    main()
