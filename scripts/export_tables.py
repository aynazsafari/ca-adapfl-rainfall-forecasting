"""Export Marked5 tables from saved evidence, without training or source rebuild.

Tables1--5 are explicitly static display transcriptions. Tables6--11 and
IS1--IS4 are calculated/selected from released saved result ledgers and checked
against metadata/paper_reference.json. Supplement CSVs preserve authoritative
numeric evidence, rather than claiming to reproduce Word table styling.

python scripts/export_tables.py --repo-root . --output publication/tables_fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

A = "benchmarks/accepted/"
H = "benchmarks/sharing/"
SRC = A + "figures/revision_v2/source_data/"
V3 = A + "publication/revision_v3/"
SUP = A + "runs/supplements_v2/publication_tables/"
SH = H + "evaluation/"
SEEDS = [7, 42, 123, 999, 2024]
POLICIES = ["LOCAL_PRIVATE", "LOCAL_STATS", "FEDAVG_PRIVATE_K1", "FEDAVG_STATS_K1",
            "FEDAVG_STATS_K5", "FEDAVG_STATS_K20", "CAAA_STATS_K1", "CENTRAL_POOL_STATS"]
POLICY_LABELS = ["Local / own", "Local / shared", "FedAvg / own K1", "FedAvg / shared K1",
                 "FedAvg / shared K5", "FedAvg / shared K20", "CA-AdapFL / shared K1", "Central pooled / shared"]
MAIN_MODELS = ["CAAA_GLOBAL_FULL", "FEDAVG_GLOBAL_FULL", "FEDPROX_GLOBAL_FULL", "CENTRAL_GLOBAL_FULL",
               "raw_pooled_ET_FULL_latest", "local_ET_FULL_latest", "client_normalized_pooled_ET_FULL_latest",
               "training_seasonal_mean", "CAAA_PERSONAL_FULL", "DITTO_PERSONAL_FULL", "LOCAL_PERSONAL_FULL",
               "GRAPH_ORIGINAL_PERSONAL_FULL", "GRAPH_ENRICHED_PERSONAL_FULL", "GRAPH_UNIFORM_PERSONAL_FULL"]
MAIN_LABELS = ["CAAA global", "FedAvg global", "FedProx global", "Central neural", "Pooled ET latest",
               "Local ET latest", "Client-normalized pooled ET", "TRAIN monthly climatology", "CAAA personal",
               "Ditto personal", "Local neural", "Original graph personal", "Enriched graph personal", "Uniform graph personal"]


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Evidence:
    def __init__(self, root):
        self.root, self.used, self.cache = root, {}, {}

    def read(self, name, kind):
        if name not in self.cache:
            path = self.root / name
            require(path.is_file(), "Missing released evidence: " + name)
            if kind == "csv":
                value = pd.read_csv(path, float_precision="round_trip")
                record = {"sha256": sha(path), "rows": len(value)}
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                record = {"sha256": sha(path)}
            self.cache[name], self.used[name] = value, record
        return self.cache[name]

    def csv(self, name):
        return self.read(name, "csv")

    def json(self, name):
        return self.read(name, "json")


def one(frame, **keys):
    part = frame
    for key, value in keys.items():
        part = part[part[key] == value]
    require(len(part) == 1, "Expected one row for " + repr(keys))
    return part.iloc[0]


def mean_sd(mean, sd, digits):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def interval(row, point_key="delta_a_minus_b", digits=6, newline=False):
    return f"{row[point_key]:.{digits}f}" + ("\n" if newline else " ") + f"[{row.CI95_low:.{digits}f}, {row.CI95_high:.{digits}f}]"


def references(paper):
    """Accept explicit main/supplement sections or a combined table list."""
    result = {}
    for key, prefix in [("tables", "Table"), ("information_sharing_tables", "IS"),
                        ("supplement_tables", "S"), ("sharing_tables", "IS")]:
        for table in paper.get(key, []):
            raw = str(table.get("id", table.get("table", "")))
            identity = raw if raw.startswith(("Table", "IS", "S")) else prefix + raw
            require(identity not in result, "Duplicate displayed table reference: " + identity)
            require(isinstance(table.get("rows"), list) and table["rows"], "Missing rows: " + identity)
            result[identity] = table["rows"]
    return result


def normalize(value):
    return re.sub(r"\s+", "", str(value)).replace("−", "-").replace("–", "-")


def numbers(value):
    value = re.sub(r"(?<=\d),(?=\d)", "", str(value)).replace("−", "-")
    return re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)


def validate_display(identity, actual, reference):
    if reference is None:
        return {"status": "REFERENCE_MISSING", "discrepancies": [{"message": "Displayed reference rows absent from metadata/paper_reference.json"}]}
    issues, compared = [], 0
    if len(actual) != len(reference):
        issues.append({"message": "Row count differs", "actual": len(actual), "reference": len(reference)})
    for i, (row, expected) in enumerate(zip(actual, reference)):
        if len(row) != len(expected):
            issues.append({"row": i, "message": "Column count differs", "actual": len(row), "reference": len(expected)})
        for j, (cell, target) in enumerate(zip(row, expected)):
            if i == 0:
                continue  # Headers are formatting/units, not result values.
            left, right = numbers(cell), numbers(target)
            if j == 0 or not left and not right:
                if normalize(cell) != normalize(target):
                    issues.append({"row": i, "column": j, "message": "Identity/structural text differs", "actual": str(cell), "reference": str(target)})
                continue
            if len(left) != len(right):
                issues.append({"row": i, "column": j, "message": "Numeric token count differs", "actual": str(cell), "reference": str(target)})
                continue
            for k, (value, wanted) in enumerate(zip(left, right)):
                decimals = len(wanted.partition(".")[2]) if "." in wanted else 0
                tolerance = .500001 * 10 ** -decimals if "." in wanted else 1e-9
                compared += 1
                if abs(float(value) - float(wanted)) > tolerance:
                    issues.append({"row": i, "column": j, "token": k, "message": "Rounded displayed number differs", "actual": value, "reference": wanted, "absolute_tolerance": tolerance})
    return {"status": "PASS" if not issues else "DISCREPANCY", "numeric_tokens_checked": compared, "discrepancies": issues}


def build(e, refs):
    tables = {}

    def add(identity, data, sources, formula, display=None, role="saved_numeric_evidence"):
        tables[identity] = {"data": data, "display": display, "sources": sources, "formula": formula, "role": role}

    for index in range(1, 6):
        identity = "Table" + str(index)
        require(identity in refs, "Static displayed metadata absent: " + identity)
        rows = refs[identity]
        add(identity, pd.DataFrame(rows[1:], columns=rows[0]), ["metadata/paper_reference.json"],
            "Exact static display transcription, including stated Table4 affinities and Table5 results. No independent numeric recomputation claimed.", rows, "static_display_transcription")

    summary = e.csv(SRC + "seed_summary_ALL.csv")
    require(not summary.model.duplicated().any(), "Duplicate primary model summaries")
    rows = [["Fixed identity", "R² (=NSE)", "RMSE mm", "MAE mm", "KGE", "PBIAS %", "|PBIAS| %", "Pearson r"]]
    for key, label in zip(MAIN_MODELS, MAIN_LABELS):
        r = one(summary, model=key)
        require(np.isclose(r.R2_mean, r.NSE_mean, atol=1e-14, rtol=0), "R2/NSE identity mismatch")
        rows.append([label, mean_sd(r.R2_mean, r.R2_std, 4), mean_sd(r.RMSE_mm_mean, r.RMSE_mm_std, 2),
                     f"{r.MAE_mm_mean:.2f}", f"{r.KGE_2009_mean:.4f}", f"{r.PBIAS_signed_percent_mean:.2f}",
                     f"{r.PBIAS_absolute_percent_mean:.2f}", f"{r.Pearson_r_mean:.4f}"])
    add("Table6", summary.set_index("model").loc[MAIN_MODELS].reset_index(), [SRC + "seed_summary_ALL.csv"],
        "Select the fourteen fixed identities in manuscript order. Saved equal-client macro metric means then five-seed mean/sample SD (ddof=1).", rows)

    ci = e.csv(SRC + "paired_block_bootstrap_summary_ALL.csv")
    sign = pd.DataFrame(e.json(SRC + "block_sign_randomization_Holm.json"))
    pairs = [("CAAA_GLOBAL_FULL", "FEDAVG_GLOBAL_FULL", "CAAA global − FedAvg global"),
             ("CAAA_GLOBAL_FULL", "FEDPROX_GLOBAL_FULL", "CAAA global − FedProx global"),
             ("CAAA_PERSONAL_FULL", "FEDAVG_PERSONAL_FULL", "CAAA personal − FedAvg personal"),
             ("CAAA_PERSONAL_FULL", "FEDPROX_PERSONAL_FULL", "CAAA personal − FedProx personal")]
    rows = [["Fixed a − b", "ΔR² [95% CI]", "ΔRMSE mm [95% CI]", "Holm p (31-family)"]]
    numeric = []
    for a, b, label in pairs:
        r = one(ci, model_a=a, model_b=b, metric="macro_R2", block_length_months=12)
        s = one(ci, model_a=a, model_b=b, metric="macro_RMSE_mm", block_length_months=12)
        p = one(sign, model_a=a, model_b=b)
        rows.append([label, interval(r), interval(s, digits=3), f"{p.Holm_p:.15g}"])
        numeric.append({"model_a": a, "model_b": b, "delta_R2": r.delta_a_minus_b, "R2_CI95_low": r.CI95_low, "R2_CI95_high": r.CI95_high,
                        "delta_RMSE_mm": s.delta_a_minus_b, "RMSE_CI95_low": s.CI95_low, "RMSE_CI95_high": s.CI95_high, "Holm_p_31": p.Holm_p})
    add("Table7", pd.DataFrame(numeric), [SRC + "paired_block_bootstrap_summary_ALL.csv", SRC + "block_sign_randomization_Holm.json"],
        "Four fixed A-minus-B pairs, block12 marginal percentile intervals and retained Holm31 annual sign enumeration; no resampling rerun.", rows)

    contrast = e.csv(SUP + "paired_contrasts_18.csv")
    supplement = e.csv(SUP + "model_summary_36.csv")
    anchors = [("CAAA_GLOBAL_FULL_ANCHOR", "CAAA_GLOBAL_FULL_NO_CLIMATE_WEIGHT", "Anchor − No climate weight"),
               ("CAAA_GLOBAL_FULL_ANCHOR", "CAAA_GLOBAL_FULL_NO_PERFORMANCE_WEIGHT", "Anchor − No performance weight"),
               ("CAAA_GLOBAL_FULL_ANCHOR", "CAAA_GLOBAL_FULL_NO_PROX", "Anchor − No proximity"),
               ("CAAA_GLOBAL_FULL_ANCHOR", "CAAA_GLOBAL_FULL_NO_AUX", "Anchor − No auxiliary head"),
               ("CAAA_GLOBAL_FULL_ANCHOR", "CAAA_GLOBAL_FULL_NO_EMBEDDING", "Anchor − No embedding"),
               ("CAAA_GLOBAL_CORE", "CAAA_GLOBAL_CORE_CLIMATE_FREE", "CORE − climate-free CORE"),
               ("CAAA_GLOBAL_LATENCY_LAG1", "CAAA_GLOBAL_LATENCY_REFERENCE", "Extra lag − matched reference")]
    rows = [["Declared comparison", "CohortTRAIN / EVAL", "R² of a / b", "ΔR² a−b", "12-month 95% CI"]]
    numeric = []
    for a, b, label in anchors:
        r = one(ci, model_a=a, model_b=b, metric="macro_R2", block_length_months=12)
        ra, rb = one(summary, model=a), one(summary, model=b)
        train = 298 if "LATENCY" in a else 299
        rows.append([label, f"{train} / 96", f"{ra.R2_mean:.4f} / {rb.R2_mean:.4f}", f"{r.delta_a_minus_b:.6f}", f"[{r.CI95_low:.6f}, {r.CI95_high:.6f}]"])
        numeric.append({"comparison": label, "model_a": a, "model_b": b, "TRAIN_months": train, "EVAL_months": 96, "R2_a": ra.R2_mean,
                        "R2_b": rb.R2_mean, "delta_R2": r.delta_a_minus_b, "CI95_low": r.CI95_low, "CI95_high": r.CI95_high})
    for case in ["IMD", "solar", "GLDAS21"]:
        r = one(contrast, case=case, method="CAAA", identity="GLOBAL")
        ra, rb = one(supplement, model=r.model_ADDED), one(supplement, model=r.model_REFERENCE)
        label = case + " added − own reference"
        rows.append([label, f"{r.TRAIN_months_per_client} / {r.evaluation_months_per_client}", f"{ra.R2_mean:.4f} / {rb.R2_mean:.4f}",
                     f"{r.delta_macro_R2:.6f}", f"[{r.macro_R2_CI95_low_block12:.6f}, {r.macro_R2_CI95_high_block12:.6f}]"])
        numeric.append({"comparison": label, "model_a": r.model_ADDED, "model_b": r.model_REFERENCE, "TRAIN_months": r.TRAIN_months_per_client,
                        "EVAL_months": r.evaluation_months_per_client, "R2_a": ra.R2_mean, "R2_b": rb.R2_mean,
                        "delta_R2": r.delta_macro_R2, "CI95_low": r.macro_R2_CI95_low_block12, "CI95_high": r.macro_R2_CI95_high_block12})
    add("Table8", pd.DataFrame(numeric), [SRC + "seed_summary_ALL.csv", SRC + "paired_block_bootstrap_summary_ALL.csv", SUP + "paired_contrasts_18.csv", SUP + "model_summary_36.csv"],
        "Fixed anchor/CORE/latency selections and within-source CAAA GLOBAL added-minus-reference selections. Cohort lengths from frozen protocol/source ledgers; derived IMD scores require no provider-valued input file.", rows)

    oni = e.csv(A + "runs/revision_v2/descriptive_diagnostics/oni_five_seed_summary.csv")
    selected = oni[(oni.model.isin([MAIN_MODELS[0], MAIN_MODELS[1], MAIN_MODELS[2], "CAAA_PERSONAL_FULL"])) & (oni.scope == "macro_equal_five_clients")]
    rows = [["Fixed main identity", "Negative n=34 R²", "RMSE mm", "Neutral n=35 R²", "RMSE mm", "Positive n=27 R²", "RMSE mm"]]
    for key, label in zip([MAIN_MODELS[0], MAIN_MODELS[1], MAIN_MODELS[2], "CAAA_PERSONAL_FULL"], [MAIN_LABELS[0], MAIN_LABELS[1], MAIN_LABELS[2], "CAAA personal"]):
        result = [label]
        for bin_name, count in [("negative_ONI_value", 34), ("neutral_ONI_value", 35), ("positive_ONI_value", 27)]:
            r = one(selected, model=key, ONI_value_bin=bin_name)
            require(r.unique_dates == count, "ONI bin support changed")
            result += [f"{r.R2_mean:.4f}", f"{r.RMSE_mm_mean:.2f}"]
        rows.append(result)
    add("Table9", selected.copy(), [A + "runs/revision_v2/descriptive_diagnostics/oni_five_seed_summary.csv"],
        "Four fixed identities; target-minus-two period-end ONI value bins; saved five-seed summaries; descriptive strata, not independent event tests.", rows)

    per = e.csv(SRC + "per_client_ALL.csv")
    selected = per[per.model == "CAAA_GLOBAL_FULL"].sort_values(["client", "seed"])
    rows = [["Client", "TRAIN q80 mm", "Observed n/96", "Precision", "Recall", "F1", "Tail RMSE mm"]]
    stats = []
    for client, label in zip(sorted(selected.client.unique()), ["C1 NW", "C2 IGP", "C3 NE", "C4 South", "C5 Central"]):
        p = selected[selected.client == client]
        require(sorted(p.seed.astype(int)) == SEEDS and p.n.eq(96).all(), "Heavy-event client support changed")
        require(p.heavy80_training_threshold_mm.nunique() == p.actual_heavy_n.nunique() == 1, "Seed-varying heavy support")
        row = [label, f"{p.heavy80_training_threshold_mm.iloc[0]:.2f}", str(int(p.actual_heavy_n.iloc[0]))]
        record = {"client": client, "q80_mm": p.heavy80_training_threshold_mm.iloc[0], "observed_n": int(p.actual_heavy_n.iloc[0])}
        for metric, digits in [("heavy_precision", 4), ("heavy_recall", 4), ("heavy_F1", 4), ("tail80_RMSE_mm", 2)]:
            row.append(mean_sd(p[metric].mean(), p[metric].std(ddof=1), digits))
            record[metric + "_mean"], record[metric + "_std"] = p[metric].mean(), p[metric].std(ddof=1)
        rows.append(row); stats.append(record)
    macro = selected.groupby("seed")[["heavy_precision", "heavy_recall", "heavy_F1", "tail80_RMSE_mm"]].mean()
    total = sum(r["observed_n"] for r in stats)
    row = ["Equal-client macro", "Client-specific", f"{total} total; not {total} flood events"]
    record = {"client": "macro_equal_five_clients", "q80_mm": np.nan, "observed_n": total}
    for metric, digits in [("heavy_precision", 4), ("heavy_recall", 4), ("heavy_F1", 4), ("tail80_RMSE_mm", 2)]:
        row.append(mean_sd(macro[metric].mean(), macro[metric].std(ddof=1), digits))
        record[metric + "_mean"], record[metric + "_std"] = macro[metric].mean(), macro[metric].std(ddof=1)
    rows.append(row); stats.append(record)
    add("Table10", pd.DataFrame(stats), [SRC + "per_client_ALL.csv"], "CAAA_GLOBAL_FULL only; per-client five-seed mean/sample SD. Macro averages client metrics within each seed before computing seed SD; no pooled precision/recall.", rows, "derived_numeric_evidence")

    principal = e.csv(SH + "principal_comparison.csv")
    cost = e.csv(SH + "fit_cost_40.csv")
    require(len(cost) == 40 and not cost.duplicated(["model", "seed"]).any(), "Sharing fit inventory changed")
    cost_mean = cost.groupby("model", sort=False).mean(numeric_only=True)
    rows = [["Policy", "MSEmm²", "RMSE mmmean ± SD", "R²", "MSESS", "PayloadMiB"]]
    numeric = []
    for key, label in zip(POLICIES + ["TRAIN_CLIMATOLOGY"], POLICY_LABELS + ["TRAIN climatology"]):
        r = one(principal, model=key)
        payload = cost_mean.loc[key, "logical_required_payload_bytes"] / 2 ** 20 if key != "TRAIN_CLIMATOLOGY" else 0
        rows.append([label, f"{r.MSE_mm2_mean:,.2f}", mean_sd(r.RMSE_mm_mean, r.RMSE_mm_std, 2) if key != "TRAIN_CLIMATOLOGY" else f"{r.RMSE_mm_mean:.2f}",
                     f"{r.R2_mean:.4f}", f"{r.MSESS_mean:.4f}", f"{payload:.4f}" if key != "TRAIN_CLIMATOLOGY" else "0"])
        numeric.append({"model": key, **r.to_dict(), "required_payload_MiB": payload})
    add("Table11", pd.DataFrame(numeric), [SH + "principal_comparison.csv", SH + "fit_cost_40.csv"],
        "Primary fixed tick60 ledger; macro MSE/R2/MSESS average clients then seeds; RMSE mean/sample SD; mean per-fit required bytes divided by 2^20. Macro MSE is not squared macro RMSE.", rows, "derived_numeric_evidence")

    direct = {"S1": A + "docs/feature_diagnostics/feature_dictionary.csv", "S2": SRC + "seed_summary_ALL.csv",
              "S3": A + "figures/revision_v2_publication_v2/model_abbreviation_ledger.csv", "S4": SRC + "paired_block_bootstrap_summary_ALL.csv",
              "S6": SUP + "model_summary_36.csv", "S7": SUP + "paired_contrasts_18.csv", "S8": SRC + "06_validation_candidates_ALL.csv",
              "S9": A + "docs/literature/comparison_table.csv", "S11": V3 + "S11_development.csv",
              "S12": V3 + "S12_components.csv", "S13": V3 + "S13_output_loss.csv", "S14": V3 + "S14_delay.csv",
              "S15": V3 + "S15_new_CI_36_all_blocks.csv", "S16": V3 + "S16_archived_CI_13_all_blocks.csv"}
    for identity, source in direct.items():
        add(identity, e.csv(source).copy(), [source], "Preserve authoritative saved rows, columns and ordering. Numeric CSV evidence export; original Word abbreviations/line breaks are not reconstructed.")
    add("S5", sign.copy(), [SRC + "block_sign_randomization_Holm.json"], "Flatten the full 31-pair exact annual sign/Holm31 JSON; no p-values recalculated.")
    fit_inventory = e.csv(V3 + "fit_inventory_100.csv")
    capacity_protocol = e.json(A + "runs/revision_v3/protocol.json")
    choices = e.json(A + "runs/revision_v3/choices_frozen.json")
    main_schema = e.json(A + "runs/revision_v3/datasets/MAIN_FULL_schema.json")
    protocol_rows = [{"item": key, "definition": json.dumps(value, ensure_ascii=False, sort_keys=True), "source_key": "protocol." + key}
                     for key, value in capacity_protocol.items() if key not in ["uncertainty"]]
    for key, value in choices.items():
        if key not in ["all_jobs", "development_jobs", "fair_jobs", "aliases", "fit_complete_hashes"]:
            protocol_rows.append({"item": "Frozen choice: " + key, "definition": json.dumps(value, ensure_ascii=False, sort_keys=True), "source_key": "choices_frozen." + key})
    protocol_rows += [{"item": "Client split counts", "definition": json.dumps({c: value["role_counts"] for c, value in main_schema["statistics"]["clients"].items()}, sort_keys=True), "source_key": "MAIN_FULL_schema.statistics.clients.role_counts"},
                      {"item": "Actual unique fits", "definition": str(len(fit_inventory)), "source_key": "len(fit_inventory_100)"},
                      {"item": "Actual aggregate optimizer steps", "definition": str(int(fit_inventory.optimizer_steps.sum())), "source_key": "sum(fit_inventory_100.optimizer_steps)"},
                      {"item": "Actual aggregate presented examples", "definition": str(int(fit_inventory.examples.sum())), "source_key": "sum(fit_inventory_100.examples)"}]
    require(len(fit_inventory) == 100 and not fit_inventory.id.duplicated().any(), "Capacity unique fit inventory changed")
    add("S10", pd.DataFrame(protocol_rows), [A + "runs/revision_v3/protocol.json", A + "runs/revision_v3/choices_frozen.json", A + "runs/revision_v3/datasets/MAIN_FULL_schema.json", V3 + "fit_inventory_100.csv"],
        "Protocol/choice definitions and schema counts, with unique fit count and aggregate optimizer steps/examples calculated from the immutable 100-fit inventory. Original prose layout is represented as explicit source-key rows.", role="derived_protocol_inventory")
    add("S10_fits", fit_inventory.copy(), [V3 + "fit_inventory_100.csv"], "All 100 unique per-fit costs and selected checkpoints; aliases/projections do not add fits.")
    cohort_columns = ["case", "TRAIN_start", "TRAIN_end", "TRAIN_months_per_client", "monitor_months_per_client", "validation_months_per_client", "evaluation_start", "evaluation_end", "evaluation_months_per_client", "REFERENCE_features", "ADDED_features"]
    cohorts = supplement[cohort_columns].drop_duplicates()
    require(len(cohorts) == 3, "Expected three matched-source cohorts")
    add("S6_cohorts", cohorts, [SUP + "model_summary_36.csv"], "Distinct frozen source-specific cohort/feature support; each source is compared only within its matched cohort.")
    for identity, source, selector in [("S7_long", SUP + "paired_contrasts_long_108_saved_rows.csv", None),
                                       ("S8_rounds", SRC + "06_validation_rounds_ALL.csv", None),
                                       ("S12_contrasts", V3 + "S15_new_CI_36_all_blocks.csv", "components"),
                                       ("S14_contrasts", V3 + "S15_new_CI_36_all_blocks.csv", "delay"),
                                       ("S15_multiplicity", A + "runs/revision_v3/evaluation/annual_sign_Holm36.csv", None)]:
        data = e.csv(source).copy()
        if selector == "components": data = data[data.contrast_id.isin(["C05", "C06", "C07", "C08"])]
        if selector == "delay": data = data[data.model_a.str.contains("DELAY2") | data.model_b.str.contains("DELAY2")]
        add(identity, data, [source], "Explicit companion numeric rows; retains all saved block lengths and both metrics where applicable.")
    v3per = e.csv(A + "runs/revision_v3/evaluation/per_client.csv")
    aux = []
    for model in tables["S13"]["data"].model:
        selected = v3per[v3per.model == model]
        require(len(selected) == 25, "Expected 25 client/seed rows for " + model)
        for metric in ["MAE_mm", "NSE", "KGE_2009", "PBIAS_signed_percent", "PBIAS_absolute_percent", "Pearson_r", "heavy_precision", "heavy_recall", "heavy_F1", "heavy_average_precision", "heavy_ROC_AUC"]:
            values = selected.groupby("seed")[metric].mean()
            aux.append({"model": model, "metric": metric, "mean_of_equal_client_seed_means": values.mean(), "sample_seed_SD": values.std(ddof=1),
                        "undefined_client_seed_rows": int(selected[metric].isna().sum()), "defined_seed_means": int(values.notna().sum())})
    add("S13b", pd.DataFrame(aux), [A + "runs/revision_v3/evaluation/per_client.csv", V3 + "S13_output_loss.csv"],
        "Twelve output/loss identities; per-seed means over defined client metrics then seed mean/sample SD. Undefined client/seed metrics are counted explicitly.", role="derived_numeric_evidence")

    protocol = e.json(H + "protocol.json")
    rows = [["Archive identity", "Input scaler", "K", "Aggregations", "Proximal term", "Steps / fit"]]
    numeric = []
    for key in POLICIES:
        arm = next(a for a in protocol["arms"] if a["label"] == key)
        k = arm["cadence"]; group = cost[cost.model == key]
        require(sorted(group.seed.astype(int)) == SEEDS, "Expected five sharing seeds")
        for col in ["optimizer_steps", "aggregation_count", "examples"]: require(group[col].nunique() == 1, "Seed-varying policy budget")
        steps = protocol["fixed_budget"]["steps_per_pooled_fit" if arm["method"] == "central" else "steps_per_distributed_fit"]
        aggregations = protocol["fixed_budget"]["ticks"] // k if k else 0
        require(group.optimizer_steps.iloc[0] == steps and group.aggregation_count.iloc[0] == aggregations, "Protocol/fit budget mismatch")
        rows.append([key, "Own" if arm["scaling"] == "local" else "Shared", str(k) if k else "None", str(aggregations),
                     "Adaptive μ=0.1" if key == "CAAA_STATS_K1" else "None", f"{steps:,}"])
        numeric.append({**arm, "aggregations": aggregations, "optimizer_steps_per_fit": steps, "examples_per_fit": int(group.examples.iloc[0]), "seeds": len(group)})
    add("IS1", pd.DataFrame(numeric), [H + "protocol.json", SH + "fit_cost_40.csv"], "Policy rows from frozen arms; K aggregation count=ticks/K; optimizer steps from fixed_budget reconciled against all forty fit ledgers.", rows, "derived_numeric_evidence")

    ici = e.csv(SH + "paired_CI_7.csv"); tests = e.csv(SH + "annual_MSE_sign_tests.csv")
    defs = ["Local/shared − Local/own", "FedAvg/shared K1 − Local/shared", "FedAvg/own K1 − Local/own", "FedAvg/shared K5 − K1",
            "FedAvg/shared K20 − K1", "S02 − S03 interaction", "CA-AdapFL/shared K1 − FedAvg/shared K1"]
    rows = [["Contrast", "ΔMSE mm²\n95% interval", "ΔRMSE mm\n95% interval", "Raw p", "Holm6", "Holm7"]]
    numeric = []
    for index, label in enumerate(defs, 1):
        cid = f"S{index:02d}"; r = one(ici, contrast_id=cid, metric="macro_MSE_mm2", block_length_months=12)
        s = one(ici, contrast_id=cid, metric="macro_RMSE_mm", block_length_months=12); t = one(tests, contrast_id=cid)
        rows.append([cid + "\n" + label, interval(r, "delta", 2, True), interval(s, "delta", 2, True),
                     f"{t.two_sided_p:.7f}", f"{t.holm_p_6_primary:.7f}" if pd.notna(t.holm_p_6_primary) else "—", f"{t.holm_p_7_sensitivity:.7f}"])
        numeric.append({"contrast_id": cid, "delta_MSE_mm2": r.delta, "MSE_CI95_low": r.CI95_low, "MSE_CI95_high": r.CI95_high,
                        "delta_RMSE_mm": s.delta, "RMSE_CI95_low": s.CI95_low, "RMSE_CI95_high": s.CI95_high,
                        "raw_p": t.two_sided_p, "Holm6": t.holm_p_6_primary, "Holm7": t.holm_p_7_sensitivity})
    add("IS2", pd.DataFrame(numeric), [SH + "paired_CI_7.csv", SH + "annual_MSE_sign_tests.csv"],
        "Seven fixed contrasts, marginal block12 saved MSE/RMSE intervals; annual MSE raw/retained Holm7/amended Holm6 p-values. S07 is outside Holm6.", rows)
    add("IS2_all_blocks", ici.copy(), [SH + "paired_CI_7.csv"], "All 84 saved intervals at block6/12/24; no uncertainty resampling.")

    endpoints = e.csv(SH + "model_summary_ALL_endpoints.csv")
    rows = [["Policy", "Tick60 global/local", "VAL global/local", "VAL personal"]]; numeric = []
    for key, label in zip(POLICIES, POLICY_LABELS):
        fixed = one(endpoints, model=key, endpoint="fixed60"); val = one(endpoints, model=key, endpoint="VAL20_40_60")
        personal = endpoints[(endpoints.model == key) & endpoints.endpoint.str.contains("personal") & endpoints.endpoint.str.contains("VAL")]
        require(len(personal) <= 1, "Ambiguous VAL personal endpoint")
        pvalue = float(personal.RMSE_mm_mean.iloc[0]) if len(personal) else np.nan
        rows.append([label, f"{fixed.RMSE_mm_mean:.2f}", f"{val.RMSE_mm_mean:.2f}", f"{pvalue:.2f}" if np.isfinite(pvalue) else "—"])
        numeric.append({"model": key, "fixed60_global_local_RMSE_mm": fixed.RMSE_mm_mean, "VAL_global_local_RMSE_mm": val.RMSE_mm_mean, "VAL_personal_RMSE_mm": pvalue})
    add("IS3", pd.DataFrame(numeric), [SH + "model_summary_ALL_endpoints.csv"], "Same fitted policies, separately declared fixed60/global/local, common VAL20/40/60 global/local and personal diagnostic paths; no endpoint selection by EVAL.", rows)
    rows = [["Policy", "Required\nbytes", "MiB", "CPU s", "Wall s", "VAL3\nbytes", "Audit\nbytes"]]
    for key, label in zip(POLICIES, POLICY_LABELS):
        r = cost_mean.loc[key]
        rows.append([label, f"{r.logical_required_payload_bytes:.0f}", f"{r.logical_required_payload_bytes / 2 ** 20:.4f}",
                     f"{r.train_cpu_seconds:.2f}", f"{r.train_wall_seconds:.2f}", f"{r.three_checkpoint_VAL_selection_bytes_all_identities:.0f}", f"{r.all_round_VAL_audit_bytes:.0f}"])
    add("IS4", cost_mean.loc[POLICIES].reset_index(), [SH + "fit_cost_40.csv"], "Arithmetic means over five per-fit ledgers. MiB=bytes/2^20. Required, alternative VAL3 selection and round-wise research audit bytes remain separate; timings describe recorded simulation.", rows, "derived_numeric_evidence")
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh nonexistent directory, resolved from current cwd")
    args = parser.parse_args(argv)
    root, output = args.repo_root.resolve(), args.output.resolve()
    require(not output.exists(), "Refusing to overwrite existing output: " + str(output))
    for protected in [root / "benchmarks", root / "metadata", root / "publication/reference"]:
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output), "Output overlaps frozen evidence/reference: " + str(protected))
    e = Evidence(root); paper = e.json("metadata/paper_reference.json"); refs = references(paper)
    tables = build(e, refs)
    expected = {"Table" + str(i) for i in range(1, 12)} | {"S" + str(i) for i in range(1, 17)} | {"IS" + str(i) for i in range(1, 5)}
    require(expected.issubset(tables), "Missing required publication table")
    reports = {identity: validate_display(identity, tables[identity]["display"], refs.get(identity))
               for identity in ["Table" + str(i) for i in range(6, 12)] + ["IS" + str(i) for i in range(1, 5)]}
    status = "PASS" if all(r["status"] == "PASS" for r in reports.values()) else "DISCREPANCY_OR_MISSING_REFERENCE"
    # All source reads, selectors and validation complete before the first write.
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for identity, table in tables.items():
        path = output / (identity + ".csv")
        table["data"].to_csv(path, index=False, float_format="%.17g", encoding="utf-8")
        files = {path.name: sha(path)}
        if table["display"] is not None:
            display_path = output / (identity + "_display.csv")
            pd.DataFrame(table["display"][1:], columns=table["display"][0]).to_csv(display_path, index=False, encoding="utf-8")
            files[display_path.name] = sha(display_path)
        record = {"id": identity, "role": table["role"], "rows": len(table["data"]), "columns": list(table["data"].columns),
                  "sources": {p: e.used[p] for p in table["sources"]}, "formula_or_selector": table["formula"], "files_sha256": files,
                  "validation_against_displayed_paper": reports.get(identity, {"status": "STATIC_TRANSCRIPTION" if table["role"] == "static_display_transcription" else "NOT_REQUESTED_NUMERIC_EVIDENCE_EXPORT"})}
        (output / (identity + "_provenance.json")).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        records[identity] = record
    receipt = {"status": status, "scope": "Rounded displayed numerical agreement for Tables6--11 and IS1--4; no model fitting/inference/source rebuild/resampling.",
               "required_tables": len(expected), "exported_tables_and_companions": len(tables), "table_checks": reports,
               "paper_reference_sha256": e.used["metadata/paper_reference.json"]["sha256"], "exporter_sha256": sha(Path(__file__).resolve())}
    (output / "validation.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"status": status, "tables": records, "source_files": e.used,
        "runtime": {"numpy": np.__version__, "pandas": pd.__version__}}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.txt").write_text("Tables1--5 are static display transcriptions, explicitly labelled in provenance.\nTables6--11 and IS1--4 have independently selected/calculated saved-result exports and displayed forms.\nS1--16 preserve authoritative saved CSV rows; companion files expose supporting block, development and auxiliary diagnostics.\nValidation compares displayed rounded results; it is not a new independent experiment or statistical refit.\nNo provider-valued IMD input files are opened or copied. Derived IMD experiment scores remain labelled as such.\n", encoding="utf-8")
    print(f"{status}: exported {len(expected)} publication tables plus {len(tables)-len(expected)} companions into {output}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
