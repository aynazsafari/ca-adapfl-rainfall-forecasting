"""Read-only independent audit of the information-sharing experiment.

No production module is imported and no training or inference is performed.
NumPy/Pandas read the numeric exports; Torch only deserializes saved tensors.
Only the requested JSON receipt is written. A failed assertion exits nonzero.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch


FLOAT_RTOL = 5e-12
FLOAT_ATOL = 5e-10
METRICS = ("macro_MSE_mm2", "macro_RMSE_mm", "macro_R2", "macro_MSESS")
VALUES = ("MSE_mm2", "RMSE_mm", "R2", "MSESS", "bias_mm", "MAE_mm",
          "tail80_bias_mm", "tail80_RMSE_mm")
ARM_NAMES = {
    "LOCAL_PRIVATE", "LOCAL_STATS", "FEDAVG_PRIVATE_K1", "FEDAVG_STATS_K1",
    "FEDAVG_STATS_K5", "FEDAVG_STATS_K20", "CAAA_STATS_K1", "CENTRAL_POOL_STATS",
}
EXPECTED_TERMS = {
    "S01": {"LOCAL_STATS": 1., "LOCAL_PRIVATE": -1.},
    "S02": {"FEDAVG_STATS_K1": 1., "LOCAL_STATS": -1.},
    "S03": {"FEDAVG_PRIVATE_K1": 1., "LOCAL_PRIVATE": -1.},
    "S04": {"FEDAVG_STATS_K5": 1., "FEDAVG_STATS_K1": -1.},
    "S05": {"FEDAVG_STATS_K20": 1., "FEDAVG_STATS_K1": -1.},
    "S06": {"FEDAVG_STATS_K1": 1., "LOCAL_STATS": -1.,
            "FEDAVG_PRIVATE_K1": -1., "LOCAL_PRIVATE": 1.},
    "S07": {"CAAA_STATS_K1": 1., "FEDAVG_STATS_K1": -1.},
}


class Audit:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.checks = 0
        self.scalar_comparisons = 0
        self.hash_count = 0
        self.max_abs_float_error = 0.
        self.categories = Counter()
        self.details = {}
        self.hashes = {}

    def check(self, condition, label: str, category="structure"):
        if not bool(condition):
            raise AssertionError(label)
        self.checks += 1
        self.categories[category] += 1

    def close(self, actual, expected, label: str, *, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        self.check(actual.shape == expected.shape, label + ": shape", "numeric")
        self.check(np.isfinite(actual).all() and np.isfinite(expected).all(),
                   label + ": finite", "numeric")
        error = np.abs(actual - expected)
        tolerance = atol + rtol * np.abs(expected)
        if not np.all(error <= tolerance):
            bad = int(np.argmax((error - tolerance).ravel()))
            raise AssertionError(f"{label}: error={error.ravel()[bad]:.17g}, "
                                 f"allowed={tolerance.ravel()[bad]:.17g}, "
                                 f"actual={actual.ravel()[bad]:.17g}, "
                                 f"expected={expected.ravel()[bad]:.17g}")
        self.scalar_comparisons += int(actual.size)
        self.max_abs_float_error = max(self.max_abs_float_error,
                                      float(error.max()) if error.size else 0.)
        self.checks += 1
        self.categories["numeric"] += 1

    def sha(self, path: Path):
        path = path.resolve()
        if path not in self.hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            self.hashes[path] = digest.hexdigest()
            self.hash_count += 1
        return self.hashes[path]

    def hash_matches(self, path: Path, expected: str):
        self.check(self.sha(path) == expected, f"SHA256 mismatch: {path}", "identity")

    @staticmethod
    def load(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def csv(path: Path):
        return pd.read_csv(path, float_precision="round_trip")


def indexed(frame, keys, audit, label):
    audit.check(not frame.duplicated(keys).any(), label + ": duplicate keys")
    return frame.set_index(keys).sort_index()


def check_identity(a):
    p = a.load(a.root / "protocol.json")
    identity = a.load(a.root / "run/identity.json")
    a.hash_matches(a.root / "protocol.json", identity["protocol_sha256"])
    a.check(identity["inputs"] == p["inputs"], "protocol and identity input agreement", "identity")
    for name, value in {**identity["inputs"], **identity["code"]}.items():
        a.hash_matches(a.root / name, value)
    a.hash_matches(a.root / "run/implementation_qa.json", identity["implementation_qa_sha256"])
    implementation = a.load(a.root / "run/implementation_qa.json")
    a.check(implementation["status"] == "PASS", "recorded implementation QA status", "identity")
    a.details["recorded_implementation_qa"] = {
        "check_count": len(implementation["checks"]),
        "scope": implementation["scope"],
        "evidence": "Authenticated pre-fit receipt; this audit does not rerun those tests.",
    }
    seeds = sorted(p["seeds"])
    a.check(seeds == [7, 42, 123, 999, 2024], "five specified seeds")
    a.check(len(p["jobs"]) == p["new_fit_count"] == 40, "protocol 40 fits")
    a.check({q["label"] for q in p["arms"]} == ARM_NAMES, "eight experiment arms")
    a.check({q["id"]: q["terms"] for q in p["primary_contrasts"]} == EXPECTED_TERMS,
            "seven frozen contrasts including interaction")
    a.check(p["uncertainty"]["blocks"] == [12, 6, 24] and
            p["uncertainty"]["repetitions"] == 2000 and p["uncertainty"]["seed"] == 731,
            "frozen bootstrap specification")
    frozen = a.load(a.root / "run/jobs_frozen.json")
    a.check(frozen["evaluation_labels_used"] is False, "job freeze labels declaration", "identity")
    jobs = frozen["jobs"]
    a.check(len(jobs) == 40 and {(q["label"], q["seed"]) for q in jobs} ==
            set(itertools.product(ARM_NAMES, seeds)), "complete 8 by 5 job grid", "identity")
    for job, expected in zip(jobs, p["jobs"]):
        source = {k: v for k, v in job.items() if k != "id"}
        a.check(source == expected, "frozen job equals protocol", "identity")
        digest = hashlib.sha256(json.dumps({"job": source,
                 "identity_sha256": a.sha(a.root / "run/identity.json")},
                 sort_keys=True).encode()).hexdigest()
        a.check(job["id"] == digest, "job identity digest", "identity")
    choices = a.load(a.root / "run/choices_frozen.json")
    a.check(choices["jobs"] == jobs and choices["EVAL_used_for_selection"] is False,
            "pre-evaluation fit choices frozen", "identity")
    start = a.load(a.root / "run/training_started.json")
    a.hash_matches(a.root / "run/jobs_frozen.json", start["jobs_sha256"])
    a.check(start["EVAL_labels_read"] is False, "fit launch labels declaration", "identity")
    launch = a.load(a.root / "run/evaluation_prelaunch.json")
    a.hash_matches(a.root / "run/evaluation_identity.json", launch["evaluation_identity_sha256"])
    sealed = a.load(a.root / "run/evaluation_identity.json")
    a.check(sealed["training_identity_sha256"] == a.sha(a.root / "run/identity.json") and
            sealed["new_EVAL_outputs_inspected"] is False, "separate evaluation identity", "identity")
    for name, value in sealed["files"].items():
        a.hash_matches(a.root / name, value)
    a.check(datetime.fromisoformat(sealed["utc"]) < datetime.fromisoformat(launch["utc"]),
            "evaluation implementation sealed before launch", "identity")
    a.hash_matches(a.root / "tools/evaluate_experiment.py", launch["evaluator_sha256"])
    a.hash_matches(a.root / "run/choices_frozen.json", launch["choices_sha256"])
    a.hash_matches(a.root / "protocol.json", launch["protocol_sha256"])
    amendment = a.load(a.root / "analysis_amendment_v1.json")
    a.hash_matches(a.root / "analysis_amendment_v1.json", launch["analysis_amendment_sha256"])
    a.check(amendment["original_protocol_sha256"] == identity["protocol_sha256"] and
            amendment["original_primary_family"] == list(EXPECTED_TERMS) and
            amendment["amended_primary_family"] == list(EXPECTED_TERMS)[:6] and
            amendment["secondary_contextual"] == ["S07"] and
            amendment["new_EVAL_outputs_inspected"] is False and
            amendment["training_jobs_unchanged"] is True, "transparent pre-EVAL analysis amendment", "identity")
    origin = a.load(a.root / "ANALYSIS_ORIGIN.json")
    a.check(origin["amendment_sha256"] == a.sha(a.root / "analysis_amendment_v1.json"),
            "original amendment binding", "identity")
    a.check(datetime.fromisoformat(amendment["utc"]) < datetime.fromisoformat(launch["utc"]),
            "analysis amendment precedes evaluation", "identity")
    if a.sha(a.root / "run/identity.json") == origin["training_identity_sha256"]:
        a.check(datetime.fromisoformat(start["utc"]) < datetime.fromisoformat(amendment["utc"]),
                "original release amendment after training start", "identity")
        timing_mode = "original_amended_after_start_before_evaluation"
    else:
        a.check(datetime.fromisoformat(amendment["utc"]) <= datetime.fromisoformat(identity["utc"]) <=
                datetime.fromisoformat(start["utc"]), "inherited analysis plan precedes new fit", "identity")
        timing_mode = "inherited_before_new_fit"
    minimum_p = 2/(2**8)
    a.close([amendment["minimum_two_sided_p"], amendment["minimum_Holm6"], amendment["minimum_Holm7"]],
            [minimum_p, 6*minimum_p, 7*minimum_p], "exact8-block multiplicity resolution", atol=0., rtol=0.)
    a.details["analysis_amendment"] = {
        "primary_family": amendment["amended_primary_family"], "secondary_contextual": ["S07"],
        "original_Holm7_retained_as_sensitivity": True,
        "minimum_two_sided_exact_p_8_blocks": minimum_p,
        "minimum_Holm6": 6*minimum_p, "minimum_Holm7": 7*minimum_p,
        "Holm7_rejection_at_05_possible": False,
        "timing_evidence": "Authenticated amendment and prelaunch timestamps; declaration of no new EVAL inspection.",
        "timing_mode": timing_mode,
    }
    complete = a.load(a.root / "evaluation/complete.json")
    a.check(complete["status"] in {"COMPLETE_PENDING_INDEPENDENT_QA", "COMPLETE"},
            "evaluation completed before QA")
    for name, value in complete["files"].items():
        a.hash_matches(a.root / "evaluation" / name, value)
    a.check(complete["new_fits"] == 40 and complete["primary_forecast_rows"] == 21600
            and complete["unique_observations"] == 480 and complete["reported_contrasts"] == 7
            and complete["primary_sharing_contrasts"] == 6 and complete["secondary_contextual_contrasts"] == 1
            and complete["bootstrap_draw_rows"] == 42000, "evaluation completion counters")
    a.details["identity_declarations"] = {
        "note": "Hashes establish file identity; numeric and semantic checks below are independent.",
        "code_files": len(identity["code"]), "input_files": len(identity["inputs"]),
        "evaluation_files": len(complete["files"]), "fit_complete_files": 40,
    }
    return p, jobs, choices


def raw_training(a):
    schema = a.load(a.root / "datasets/MAIN_FULL_schema.json")
    clients = sorted(schema["statistics"]["clients"])
    a.check(len(clients) == 5 and len(schema["names"]) == 43 and
            len(schema["climate_names"]) == 7, "five clients and 43/7 input dimensions")
    raw, stats, climatology = {}, {}, {}
    with np.load(a.root / "datasets/MAIN_FULL_fit.npz", allow_pickle=False) as src:
        for client in clients:
            values = {k: src[client + "::raw_" + k].copy()
                      for k in ("x", "climate", "target_month")}
            values["y"] = src[client + "::y"].copy()
            months = values["target_month"]
            masks = {
                "train": months <= "2007-11",
                "monitor": (months >= "2008-01") & (months <= "2010-11"),
                "validation": (months >= "2011-01") & (months <= "2016-11"),
            }
            a.check([int(masks[k].sum()) for k in masks] == [299, 35, 71],
                    f"{client}: 299/35/71 split counts", "budget")
            a.check(sum(masks.values()).min() == 1 and sum(masks.values()).max() == 1,
                    f"{client}: mutually exclusive fit roles")
            for key in ("x", "climate", "y"):
                a.check(values[key].dtype == np.float64 and np.isfinite(values[key]).all(),
                        f"{client}: raw float64 finite {key}")
            mask = masks["train"]
            x = values["x"][mask, -1, :]
            y = values["y"][mask]
            mean = x.sum(0) / len(x)
            variance = np.maximum((x * x).sum(0) / len(x) - mean * mean, 0.)
            sd = np.sqrt(variance)
            sd[sd < 1e-12] = 1.
            stats[client] = {"TRAIN_count": len(x), "TRAIN_sum": x.sum(0),
                "TRAIN_squared_sum": (x*x).sum(0), "input_mean": mean, "input_std": sd,
                "target_mean": y.mean(), "target_std": y.std(ddof=0),
                "q80_mm": np.quantile(y, .8, method="linear")}
            for calendar_month in range(1, 13):
                yy = y[np.array([int(month[-2:]) == calendar_month for month in months[mask]])]
                climatology[client, calendar_month] = {"TRAIN_count": len(yy),
                    "climatology_mm": yy.mean(), "q80_mm": stats[client]["q80_mm"]}
            values["train"] = mask
            raw[client] = values
    total = sum(s["TRAIN_count"] for s in stats.values())
    sums = np.sum([s["TRAIN_sum"] for s in stats.values()], axis=0)
    squares = np.sum([s["TRAIN_squared_sum"] for s in stats.values()], axis=0)
    shared_mean = sums / total
    shared_sd = np.sqrt(np.maximum(squares / total - shared_mean * shared_mean, 0.))
    shared_sd[shared_sd < 1e-12] = 1.
    a.check(total == 1495, "1495 pooled TRAIN examples", "budget")
    with np.load(a.root / "datasets/MAIN_FULL_evaluation_features.npz", allow_pickle=False) as src:
        a.check(not any(k.endswith(("::y", "::y_z", "::heavy")) for k in src.files),
                "EVAL feature archive is label-free")
        for client in clients:
            dates = src[client + "::raw_target_month"]
            a.check(len(dates) == 96 and len(set(dates)) == 96 and min(dates) == "2017-01"
                    and max(dates) == "2024-12", f"{client}: EVAL feature dates")
    central_bytes = {
        c: sum(np.asarray(raw[c][key][raw[c]["train"]], dtype=np.float64).nbytes
               for key in ("x", "climate", "y")) + np.dtype(np.int64).itemsize
        for c in clients
    }
    a.details["payload_definition"] = {
        "shared_scaler_setup_bytes": 5 * (8 + 16 * 43) + 5 * 16 * 43,
        "central_train_upload_bytes": sum(central_bytes.values()),
        "central_deployment_downlink_bytes": 5 * 69320,
        "central_per_client_bundle_bytes": central_bytes,
        "dtype": "float64 x/climate/labels; int64 client count; float32 model parameters",
    }
    return clients, raw, stats, climatology, shared_mean, shared_sd, central_bytes


def model_parameter_formula(features, context, config):
    h, layers, emb, att = (config[k] for k in ("hidden", "layers", "embedding", "attention_hidden"))
    # Bidirectional LSTM: each direction has W_ih, W_hh and two gate biases.
    lstm = sum(2 * (4*h * (features if layer == 0 else 2*h) + 4*h*h + 8*h)
               for layer in range(layers))
    linear = lambda inp, out: inp*out + out
    width = 2*h + emb
    return (lstm + linear(max(context, 1), 32) + linear(32, emb)
            + linear(width, att) + att + linear(width, 64) + linear(64, 32)
            + linear(32, 1) + linear(width, 32) + linear(32, 1))


def message_signature(row):
    return (int(row.get("round", -1)), row["client_id"], row["kind"],
            row["direction"], int(row["payload_bytes"]))


def inspect_fits(a, p, jobs, choices, clients, raw, stats, shared_mean, shared_sd, central_bytes):
    budget = p["fixed_budget"]
    a.check({k: budget[k] for k in ("ticks", "epochs_per_tick", "batch_size")} ==
            {"ticks": 60, "epochs_per_tick": 3, "batch_size": 32}, "fixed update budget", "budget")
    cost_rows, selected = [], {}
    checkpoints_tested = 0
    tensor_scalars_tested = 0
    for job in jobs:
        directory = a.root / "run/fits" / job["id"]
        receipt = a.load(directory / "complete.json")
        a.hash_matches(directory / "complete.json", choices["fit_complete_hashes"][job["id"]])
        a.check(receipt["job"] == job, "fit receipt job identity", "identity")
        declared = receipt["files"]
        actual = {q.relative_to(directory).as_posix() for q in directory.rglob("*")
                  if q.is_file() and q != directory / "complete.json"}
        a.check(actual == set(declared), f"{job['id']}: fit file manifest exhaustive", "identity")
        for name, value in declared.items():
            a.hash_matches(directory / name, value)
        cost = a.load(directory / "cost.json")
        a.check(cost == receipt["cost"], "fit cost file equals receipt", "identity")
        is_fed = job["method"] in {"fedavg", "caaa"}
        identities = (["global", "personal"] if is_fed else
                      ["global"] if job["method"] == "central" else ["personal"])
        terminal = a.load(directory / "terminal/manifest.json")
        secondary = a.load(directory / "validation_selected/manifest.json")
        a.check(terminal["config"] == secondary["config"] == job["config"],
                "saved config equals frozen config", "identity")
        a.check(terminal["selected_rounds"] == {key: 60 for key in identities},
                "primary identities at final tick60")
        a.check((terminal["features"], terminal["context"], terminal["window"]) == (43, 7, 12),
                "saved model input dimensions")
        for client in clients:
            manifest = terminal["data_manifest"][client]
            for role, count in (("train", 299), ("monitor", 35), ("validation", 71)):
                a.check(manifest[role]["count"] == len(manifest[role]["keys"]) == count,
                        "fit role manifest counts", "budget")
        scaler = a.load(directory / "scaler_statistics.json")
        a.check(scaler["mode"] == job["scaling"] and scaler["fit_role"] == "TRAIN only",
                "scaler role/scaling mode")
        for client in clients:
            for key, value in stats[client].items():
                wanted = (shared_mean if key == "input_mean" else shared_sd
                          if key == "input_std" else value) if job["scaling"] == "shared" else value
                a.close(scaler["clients"][client][key], wanted,
                        f"{job['label']}/{job['seed']}/{client}: TRAIN {key}")
        steps = a.load(directory / "terminal/local_steps.json")
        secondary_steps = a.load(directory / "validation_selected/local_steps.json")
        a.check(steps == secondary_steps, "terminal and VAL branch use same fit update logs", "budget")
        streams = ["__central_pooled__"] if job["method"] == "central" else clients
        keys = [(int(row["round"]), row["client_id"]) for row in steps]
        a.check(len(keys) == len(set(keys)) and set(keys) ==
                set(itertools.product(range(1, 61), streams)), "complete client/tick update grid", "budget")
        examples_expected, steps_expected = 0, 0
        for row in steps:
            count = 1495 if job["method"] == "central" else stats[row["client_id"]]["TRAIN_count"]
            ee = count * 3
            ss = math.ceil(count/32) * 3
            a.check(row["examples"] == ee and row["optimizer_steps"] == ss and
                    row["optimizer"] == "adam" and row["sgd_steps"] == 0,
                    "per-tick TRAIN example/Adam budgets", "budget")
            examples_expected += ee
            steps_expected += ss
        a.check(examples_expected == receipt["examples"] == 269100 and
                steps_expected == receipt["optimizer_steps"] ==
                (8460 if job["method"] == "central" else 9000), "fit total example/step budget", "budget")
        parameter_count = model_parameter_formula(43, 7, job["config"])
        a.check(parameter_count == cost["parameters"] == 17330 and cost["model_state_bytes"] == 69320,
                "independent architecture parameter/byte formula", "payload")
        expected_messages = []
        def add(client, kind, direction, size, rd=None):
            q = dict(client_id=client, kind=kind, direction=direction, payload_bytes=size)
            if rd is not None:
                q["round"] = rd
            expected_messages.append(q)
        if job["method"] == "central":
            for client in clients:
                add(client, "engineered_TRAIN_example_bundle", "uplink", central_bytes[client])
                add(client, "deployed_model", "downlink", 69320)
        else:
            if job["scaling"] == "shared":
                for client in clients:
                    add(client, "TRAIN_scaler_count_sum_squaredsum", "uplink", 696)
                    add(client, "TRAIN_scaler_mean_std", "downlink", 688)
            if is_fed:
                for client in clients:
                    add(client, "initial_model", "downlink", 69320, 0)
                    add(client, "train_count", "uplink", 8, 0)
                for tick in range(job["cadence"], 61, job["cadence"]):
                    for client in clients:
                        add(client, "model_update", "uplink", 69320, tick)
                        add(client, "aggregated_model", "downlink", 69320, tick)
        messages = a.load(directory / "required_training_messages.json")
        a.check(Counter(map(message_signature, messages)) == Counter(map(message_signature, expected_messages)),
                "exact logical message inventory including final deployment", "payload")
        training_messages = a.load(directory / "terminal/messages.json")
        a.check(Counter(map(message_signature, training_messages)) ==
                Counter(message_signature(row) for row in expected_messages if "round" in row),
                "engine messages agree with required training exchange", "payload")
        totals = {
            "logical_required_payload_bytes": sum(row["payload_bytes"] for row in expected_messages),
            "uplink_bytes": sum(row["payload_bytes"] for row in expected_messages if row["direction"] == "uplink"),
            "downlink_bytes": sum(row["payload_bytes"] for row in expected_messages if row["direction"] == "downlink"),
            "scaler_setup_bytes": sum(row["payload_bytes"] for row in expected_messages if "scaler" in row["kind"]),
            "model_tensor_bytes": sum(row["payload_bytes"] for row in expected_messages if row["kind"] in
                                      {"initial_model", "model_update", "aggregated_model", "deployed_model"}),
            "aggregation_count": 60//job["cadence"] if is_fed else 0,
            "three_checkpoint_VAL_selection_bytes_per_identity": 5*3*16 + 5*8,
            "three_checkpoint_VAL_selection_bytes_all_identities": (5*3*16 + 5*8)*len(identities),
            "all_round_VAL_audit_bytes": (5*60*16 + 5*8)*len(identities),
        }
        for key, wanted in totals.items():
            a.check(cost[key] == wanted, f"{job['label']}: recomputed {key}", "payload")
        if job["scaling"] == "shared" and job["method"] != "central":
            a.check(cost["scaler_setup_bytes"] == 6920, "shared statistics setup6920", "payload")
        curves = a.load(directory / "terminal/curves.json")
        a.check(curves == a.load(directory / "validation_selected/curves.json"),
                "same fit curves for both saved branches")
        aggregations = [row for row in curves if row["kind"] == "aggregation"]
        expected_ticks = list(range(job["cadence"], 61, job["cadence"])) if is_fed else []
        a.check([row["round"] for row in aggregations] == expected_ticks,
                "aggregation ticks exactly match cadence", "aggregation")
        for row in aggregations:
            a.check(set(row["weights"]) == set(clients), "aggregation client set", "aggregation")
            a.close([row["weights"][client] for client in clients], np.full(5, .2),
                    "uniform count aggregation weights", atol=1e-15, rtol=0.)
        for identity in identities:
            evaluations = [row for row in curves if row["kind"] == "evaluation" and row["identity"] == identity]
            a.check([row["round"] for row in evaluations] == list(range(1, 61)),
                    "60 VAL audit ticks per identity", "validation")
            for row in evaluations:
                a.check(set(row["validation"]) == set(clients) and
                        all(row["validation"][c]["n"] == 71 for c in clients),
                        "VAL macro uses all five clients/71 months", "validation")
                a.close(row["validation_macro_r2"],
                        np.mean([row["validation"][c]["r2"] for c in clients]),
                        "VAL macro computed from equal-client R2")
            best, chosen = -np.inf, None
            for tick in (20, 40, 60):
                score = next(row["validation_macro_r2"] for row in evaluations if row["round"] == tick)
                if score > best + 1e-12:
                    best, chosen = score, tick
            a.check(receipt["selected_rounds"][identity] == secondary["selected_rounds"][identity] == chosen,
                    "earliest best common VAL20/40/60 checkpoint", "validation")
        selected[job["id"]] = secondary["selected_rounds"]
        audit_messages = a.load(directory / "research_audit_messages.json")
        expected_audit = [dict(round=tick, client_id=c, kind="VAL_"+identity,
                         direction="uplink", payload_bytes=16)
                         for tick in range(1, 61) for identity in identities for c in clients]
        expected_audit.extend(dict(round=secondary["selected_rounds"][identity], client_id=c,
                              kind="chosen_round_"+identity, direction="downlink", payload_bytes=8)
                              for identity in identities for c in clients)
        a.check(Counter(map(message_signature, audit_messages)) == Counter(map(message_signature, expected_audit)),
                "VAL audit/selection sharing inventory", "validation")
        saved = torch.load(directory / "terminal/selected_states.pt", map_location="cpu", weights_only=True)
        states = ([saved["global"]] if saved["global"] is not None else []) + list(saved["personal"].values())
        for state in states:
            a.check(sum(t.numel() for t in state.values()) == 17330 and
                    sum(t.numel()*t.element_size() for t in state.values()) == 69320 and
                    all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in state.values()),
                    "saved tensor parameter count/dtype/size/finiteness", "tensor")
        history = torch.load(directory / "terminal/round_states.pt", map_location="cpu", weights_only=True)
        a.check([row["round"] for row in history] == [1, 10, 20, 30, 40, 50, 60],
                "saved state checkpoint grid", "tensor")
        last = history[-1]
        for identity in ("global", "personal"):
            endpoint_states = {"global": saved["global"]} if identity == "global" else saved["personal"]
            history_states = {"global": last["global_state"]} if identity == "global" else last["personal_states"]
            a.check(set(endpoint_states) == set(history_states), "final history identity set", "tensor")
            for client, state in endpoint_states.items():
                if state is None:
                    a.check(history_states[client] is None, "absent global state", "tensor")
                    continue
                a.check(all(torch.equal(tensor, history_states[client][key]) for key, tensor in state.items()),
                        "terminal tensors exactly equal final checkpoint", "tensor")
        if is_fed:
            for row in history:
                if row["round"] % job["cadence"]:
                    continue
                a.check(set(row["personal_states"]) == set(clients), "checkpoint personal client states", "tensor")
                for key, global_tensor in row["global_state"].items():
                    local = np.stack([row["personal_states"][c][key].numpy().astype(np.float64) for c in clients])
                    wanted = local.mean(0)
                    actual_tensor = global_tensor.numpy().astype(np.float64)
                    # Five float32 products and four additions: bound roundoff in
                    # the stored accumulation against an independently computed float64 mean.
                    gamma9 = 9*np.finfo(np.float32).eps/(1-9*np.finfo(np.float32).eps)
                    bound = gamma9*np.mean(np.abs(local), axis=0) + np.finfo(np.float32).tiny
                    a.check(np.all(np.abs(actual_tensor-wanted) <= bound),
                            "saved global equals uniform personal-state mean within float32 operation bound", "tensor")
                    tensor_scalars_tested += int(wanted.size)
                checkpoints_tested += 1
        cost_rows.append(dict(model=job["label"], seed=job["seed"], job_id=job["id"],
                              optimizer_steps=steps_expected, examples=examples_expected, **cost))
    total = a.load(a.root / "run/fit_complete.json")
    a.check(total["status"] == "COMPLETE" and total["unique_new_fits"] == 40 and
            total["total_examples"] == 40*269100 and total["total_steps"] == 35*9000+5*8460,
            "all40 fit aggregate budgets", "budget")
    exported = indexed(a.csv(a.root / "evaluation/fit_cost_40.csv"), ["model", "seed"], a, "exported costs")
    recomputed = indexed(pd.DataFrame(cost_rows), ["model", "seed"], a, "recomputed costs")
    a.check(exported.index.equals(recomputed.index), "fit cost export grid")
    a.check(exported.job_id.tolist() == recomputed.job_id.tolist(), "fit cost export identities")
    for key in recomputed.select_dtypes(include=[np.number]).columns:
        a.close(exported[key].to_numpy(), recomputed[key].to_numpy(), "fit cost export " + key)
    a.details["fit_semantics"] = {"fits_checked": 40, "examples_per_fit": 269100,
        "distributed_steps_per_fit": 9000, "central_steps_per_fit": 8460,
        "parameters": 17330, "model_state_bytes": 69320,
        "aggregation_checkpoints_recomputed": checkpoints_tested,
        "aggregation_tensor_scalars_recomputed": tensor_scalars_tested,
        "validation_selection_bytes_per_identity": 280, "federated_all_identity_selection_bytes": 560,
        "scope": "Saved update logs, selected checkpoints, and tensors; no optimizer/inference rerun."}
    return selected


def principal_predictions(a, p, jobs, clients, raw, stats, climatology, selected):
    directory = a.root / "evaluation"
    primary = a.csv(directory / "raw_predictions_PRIMARY.csv")
    keys = ["endpoint", "model", "seed", "client", "target_month"]
    indexed(primary, keys, a, "primary predictions")
    months = list(pd.period_range("2017-01", "2024-12", freq="M").astype(str))
    models = sorted(ARM_NAMES | {"TRAIN_CLIMATOLOGY"})
    seeds = sorted(p["seeds"])
    a.check(len(primary) == 21600 and primary.endpoint.unique().tolist() == ["fixed60"],
            "21600 fixed60 primary rows")
    a.check(set(map(tuple, primary[["model", "seed", "client", "target_month"]].to_numpy())) ==
            set(itertools.product(models, seeds, clients, months)), "complete 9 by 5 by 5 by96 prediction grid")
    a.check(primary[["client", "target_month"]].drop_duplicates().shape[0] == 480 and
            sorted(primary.target_month.unique()) == months, "480 unique outcomes and96months")
    for key in ("y", "prediction", "climatology_mm", "q80_mm", "checkpoint_tick"):
        a.check(np.isfinite(primary[key]).all(), "finite primary " + key)
    a.check((primary.prediction >= 0).all() and (primary.y >= 0).all(), "nonnegative primary rainfall")
    probabilities = primary.loc[primary.model != "TRAIN_CLIMATOLOGY", "heavy_probability"].to_numpy()
    a.check(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "finite learned heavy probabilities in[0,1]")
    a.check(primary.loc[primary.model == "TRAIN_CLIMATOLOGY", "heavy_probability"].isna().all(),
            "climatology probability intentionally undefined")
    panel = indexed(a.csv(a.root / "datasets/monthly.csv"), ["client", "month"], a, "monthly source")
    source_keys = pd.MultiIndex.from_arrays([primary.client, primary.target_month], names=["client", "month"])
    a.close(primary.y.to_numpy(), panel.loc[source_keys, "rainfall_mm"].to_numpy(), "primary truth source alignment")
    expected_clim = np.array([climatology[c, int(m[-2:])]["climatology_mm"]
                              for c, m in zip(primary.client, primary.target_month)])
    expected_tail = np.array([stats[c]["q80_mm"] for c in primary.client])
    a.close(primary.climatology_mm.to_numpy(), expected_clim, "TRAIN monthly climatology")
    a.close(primary.q80_mm.to_numpy(), expected_tail, "TRAIN q80 thresholds")
    reference = primary.model == "TRAIN_CLIMATOLOGY"
    a.close(primary.loc[reference, "prediction"].to_numpy(), expected_clim[reference], "climatology reference predictions")
    job_lookup = {(q["label"], q["seed"]): q for q in jobs}
    for (model, seed), group in primary.groupby(["model", "seed"]):
        if model == "TRAIN_CLIMATOLOGY":
            a.check(set(group.identity) == {"fixed_reference"} and set(group.job_id) == {"NO_FIT"}
                    and set(group.checkpoint_tick) == {0}, "reference identity/checkpoint")
        else:
            job = job_lookup[model, seed]
            a.check(set(group.job_id) == {job["id"]} and set(group.checkpoint_tick) == {60} and
                    set(group.identity) == {job["config"]["primary_identity"]}, "primary fit identity/final tick")
    all_endpoints = a.csv(directory / "raw_predictions_ALL_endpoints.csv")
    indexed(all_endpoints, keys, a, "all-endpoint predictions")
    expected_all = set()
    for job in jobs:
        identities = (["global", "personal"] if job["method"] in {"fedavg", "caaa"} else
                      ["global"] if job["method"] == "central" else ["personal"])
        for branch in ("fixed60", "VAL20_40_60"):
            for identity in identities:
                endpoint = branch if identity == job["config"]["primary_identity"] else branch+"_personal_diagnostic"
                expected_all.update((endpoint, job["label"], job["seed"], c, month)
                                    for c, month in itertools.product(clients, months))
    expected_all.update(("fixed60", "TRAIN_CLIMATOLOGY", seed, c, month)
                        for seed, c, month in itertools.product(seeds, clients, months))
    a.check(len(all_endpoints) == 64800 and set(map(tuple, all_endpoints[keys].to_numpy())) == expected_all,
            "64800 complete terminal/VAL/global/personal diagnostic endpoint grid")
    fixed = indexed(all_endpoints[all_endpoints.endpoint == "fixed60"], keys, a, "all-endpoint fixed60 subset")
    primary_indexed = indexed(primary, keys, a, "principal index")
    a.check(fixed.index.equals(primary_indexed.index), "ALL endpoint primary keys match")
    for key in ("y", "prediction", "climatology_mm", "q80_mm", "checkpoint_tick"):
        a.close(fixed[key].to_numpy(), primary_indexed[key].to_numpy(), "ALL/PRIMARY " + key, atol=0., rtol=0.)
    for key in ("y", "prediction", "climatology_mm", "q80_mm", "checkpoint_tick"):
        a.check(np.isfinite(all_endpoints[key]).all(), "finite all endpoint " + key)
    a.check((all_endpoints.prediction >= 0).all(), "nonnegative all endpoint predictions")
    learned_probabilities = all_endpoints.loc[all_endpoints.model != "TRAIN_CLIMATOLOGY", "heavy_probability"].to_numpy()
    a.check(np.isfinite(learned_probabilities).all() and
            ((learned_probabilities >= 0) & (learned_probabilities <= 1)).all(),
            "finite all learned endpoint heavy probabilities in[0,1]")
    for (model, seed, identity, endpoint), group in all_endpoints[all_endpoints.model != "TRAIN_CLIMATOLOGY"].groupby(
            ["model", "seed", "identity", "endpoint"]):
        job = job_lookup[model, seed]
        wanted_tick = selected[job["id"]][identity] if endpoint.startswith("VAL") else 60
        a.check(set(group.job_id) == {job["id"]} and set(group.checkpoint_tick) == {wanted_tick},
                "ALL endpoint fitted job/identity/checkpoint alignment")
    a.check(len(all_endpoints) == a.load(directory / "complete.json")["all_endpoint_rows"],
            "ALL endpoint completion row count")
    for _, group in all_endpoints[all_endpoints.endpoint.str.startswith("VAL")].groupby(["job_id", "identity"]):
        job_id, identity = group.iloc[0].job_id, group.iloc[0].identity
        a.check(set(group.checkpoint_tick) == {selected[job_id][identity]}, "VAL export uses selected checkpoint")
    train_clim = indexed(a.csv(directory / "TRAIN_climatology.csv"), ["client", "calendar_month"], a, "TRAIN climatology table")
    a.check(set(train_clim.index) == set(climatology), "60 TRAIN climatology cells")
    for key, value in climatology.items():
        for column, expected in value.items():
            a.close(train_clim.loc[key, column], expected, "TRAIN climatology table " + column)
    a.details["prediction_grid"] = {"primary_rows": len(primary), "all_endpoint_rows": len(all_endpoints),
        "models": models, "training_seeds": seeds, "fixed_clients": clients,
        "months": 96, "unique_outcomes": 480, "negative_predictions": 0,
        "note": "Seed repetitions are paired bookkeeping, not additional unique outcomes."}
    return primary, models, seeds, clients, months


def score_exports(a, primary):
    rows = []
    for (endpoint, model, seed, client), group in primary.groupby(["endpoint", "model", "seed", "client"]):
        truth = group.y.to_numpy(dtype=np.float64)
        prediction = group.prediction.to_numpy(dtype=np.float64)
        errors = prediction - truth
        squared = errors**2
        tail = truth > group.q80_mm.to_numpy(dtype=np.float64)
        total_loss = squared.sum()
        variance_sum = ((truth-truth.mean())**2).sum()
        climatology_loss = ((truth-group.climatology_mm.to_numpy(dtype=np.float64))**2).sum()
        a.check(variance_sum > 0 and climatology_loss > 0 and tail.any(), "defined skill/tail denominators")
        rows.append(dict(endpoint=endpoint, model=model, seed=int(seed), client=client,
            n=len(truth), MSE_mm2=total_loss/len(truth), RMSE_mm=math.sqrt(total_loss/len(truth)),
            R2=1-total_loss/variance_sum, MSESS=1-total_loss/climatology_loss,
            bias_mm=errors.mean(), MAE_mm=np.abs(errors).mean(),
            tail80_bias_mm=errors[tail].mean(), tail80_RMSE_mm=np.sqrt((errors[tail]**2).mean()),
            tail_n=int(tail.sum()), negative_predictions=int((prediction<0).sum())))
    computed = pd.DataFrame(rows)
    saved = a.csv(a.root / "evaluation/per_client_seed_ALL_endpoints.csv")
    saved = saved[saved.endpoint == "fixed60"]
    keys = ["endpoint", "model", "seed", "client"]
    aa, bb = indexed(saved, keys, a, "per-client export"), indexed(computed, keys, a, "independent per-client metrics")
    a.check(aa.index.equals(bb.index) and len(bb) == 225, "225 principal client-seed metric rows")
    for key in (*VALUES, "n", "tail_n", "negative_predictions"):
        a.close(aa[key].to_numpy(), bb[key].to_numpy(), "per-client independent " + key)
    by_seed = computed.groupby(["endpoint", "model", "seed"])[list(VALUES)].mean().sort_index()
    saved = a.csv(a.root / "evaluation/macro_per_seed_ALL_endpoints.csv")
    saved = indexed(saved[saved.endpoint == "fixed60"], ["endpoint", "model", "seed"], a, "macro per-seed export")
    a.check(saved.index.equals(by_seed.index), "45 macro seed metric rows")
    for key in VALUES:
        a.close(saved[key].to_numpy(), by_seed[key].to_numpy(), "macro independent " + key)
    principal = by_seed.droplevel("endpoint").groupby("model").agg(["mean", "std"])
    principal.columns = ["_".join(pair) for pair in principal.columns]
    saved = indexed(a.csv(a.root / "evaluation/principal_comparison.csv"), ["model"], a, "principal comparison")
    a.check(saved.index.equals(principal.index), "nine model principal rows")
    for key in principal.columns:
        a.close(saved[key].to_numpy(), principal[key].to_numpy(), "principal independent " + key)
    a.details["metric_recomputation"] = {"client_seed_rows": 225, "macro_seed_rows": 45,
        "principal_model_rows": 9, "metrics": list(VALUES),
        "macro_RMSE_definition": "Mean of client-seed RMSE, never square root of macro MSE."}
    return by_seed


def endpoint_statistics(truth, predictions, climatology):
    """Independent fixed-client arithmetic, including resampled denominators."""
    residuals = predictions - truth[None, None]
    loss_sums = np.sum(np.square(residuals), axis=-1, dtype=np.float64)
    n = truth.shape[-1]
    mse = loss_sums / n
    observed_ss = np.sum(np.square(truth - np.mean(truth, axis=-1, keepdims=True)), axis=-1)
    climate_ss = np.sum(np.square(truth-climatology), axis=-1)
    if not (np.all(observed_ss > 0.) and np.all(climate_ss > 0.)):
        raise AssertionError("Undefined independently resampled skill denominator")
    return np.column_stack((np.mean(mse, axis=(1, 2)), np.mean(np.sqrt(mse), axis=(1, 2)),
        np.mean(1.-loss_sums/observed_ss, axis=(1, 2)),
        np.mean(1.-loss_sums/climate_ss, axis=(1, 2))))


def uncertainty(a, p, primary, models, seeds, clients, months, by_seed):
    directory = a.root / "evaluation"
    family = a.load(a.root / "analysis_amendment_v1.json")["amended_primary_family"]
    primary_indexed = primary.set_index(["model", "seed", "client", "target_month"])
    prediction = primary_indexed.loc[pd.MultiIndex.from_product([models, seeds, clients, months]),
                                    "prediction"].to_numpy().reshape(9, 5, 5, 96)
    base = primary[(primary.model == models[0]) & (primary.seed == seeds[0])].sort_values(["client", "target_month"])
    truth = base.y.to_numpy().reshape(5, 96)
    climate = base.climatology_mm.to_numpy().reshape(5, 96)
    contrasts = p["primary_contrasts"]
    ids = [row["id"] for row in contrasts]
    # Explicit term summation, independent of the production matrix product.
    def apply_contrasts(statistics):
        return np.stack([np.sum([coefficient*statistics[models.index(model)]
                         for model, coefficient in row["terms"].items()], axis=0) for row in contrasts])
    point = apply_contrasts(endpoint_statistics(truth, prediction, climate))
    seed_summary = by_seed.droplevel("endpoint").groupby("model").mean()
    for i, row in enumerate(contrasts):
        expected = np.sum([coefficient*seed_summary.loc[model, list(VALUES[:4])].to_numpy()
                           for model, coefficient in row["terms"].items()], axis=0)
        a.close(point[i], expected, "principal contrast from metric summaries " + row["id"])
    a.close(point[5], point[1]-point[2], "interaction equals shared minus private exchange effect")
    samples = np.load(directory / "bootstrap_resamples.npz", allow_pickle=False)
    a.check(set(samples.files) == {"block", "replicate", "seeds", "dates"}, "saved resample fields")
    block, repeat, seed_ix, date_ix = (samples[key] for key in ("block", "replicate", "seeds", "dates"))
    a.check(block.shape == repeat.shape == (6000,) and seed_ix.shape == (6000, 5) and
            date_ix.shape == (6000, 96), "6000 complete joint resample shapes")
    a.check(all(np.issubdtype(q.dtype, np.integer) for q in (block, repeat, seed_ix, date_ix)),
            "integer joint resample indices")
    a.check((seed_ix >= 0).all() and (seed_ix < 5).all() and
            (date_ix >= 0).all() and (date_ix < 96).all(), "joint index bounds")
    receipt = a.load(directory / "uncertainty_receipt.json")
    a.check(receipt["models"] == models and receipt["seeds"] == seeds and receipt["clients"] == clients
            and receipt["months"] == months and receipt["common_resampling"] is True
            and receipt["interaction_recomputed_in_each_draw"] is True and receipt["clients_fixed"] is True
            and receipt["primary_family"] == family and receipt["original_family_Holm7_retained"] is True,
            "resampling receipt axis/paired declarations")
    saved_draws = indexed(a.csv(directory / "bootstrap_draws_7.csv"),
                         ["block_length_months", "replicate", "contrast_id"], a, "bootstrap draw export")
    a.check(len(saved_draws) == 42000 and set(saved_draws.index) ==
            set(itertools.product([12, 6, 24], range(2000), ids)), "42000 complete bootstrap draws")
    replayed = np.empty((6000, 7, 4), dtype=np.float64)
    rng = np.random.default_rng(731)
    expected_order = list(itertools.product([12, 6, 24], range(2000)))
    for row, (length, repetition) in enumerate(expected_order):
        a.check((int(block[row]), int(repeat[row])) == (length, repetition), "joint resample block/repetition ordering")
        starts = rng.integers(0, 96, size=math.ceil(96/length))
        dates = np.concatenate([(np.arange(start, start+length) % 96) for start in starts])[:96]
        ss = rng.integers(0, 5, size=5)
        a.check(np.array_equal(date_ix[row], dates) and np.array_equal(seed_ix[row], ss),
                "independently regenerated circular dates and paired seed indices", "resampling")
        sampled = prediction[:, seed_ix[row]][..., date_ix[row]]
        replayed[row] = apply_contrasts(endpoint_statistics(truth[:, date_ix[row]], sampled, climate[:, date_ix[row]]))
        exported = saved_draws.loc[(length, repetition, ids), list(METRICS)].to_numpy()
        a.close(exported, replayed[row], "independently replayed seven joint contrast draws")
    a.close(replayed[:, 5], replayed[:, 1]-replayed[:, 2], "interaction identity in every saved draw")
    ci = indexed(a.csv(directory / "paired_CI_7.csv"), ["block_length_months", "contrast_id", "metric"], a, "CI export")
    a.check(len(ci) == 84 and set(ci.index) == set(itertools.product([12, 6, 24], ids, METRICS)),
            "84 complete CI rows")
    for length in [12, 6, 24]:
        subset = replayed[block == length]
        bounds = np.quantile(subset, [.025, .975], axis=0, method="linear")
        for ci_ix, contrast_id in enumerate(ids):
            for metric_ix, metric in enumerate(METRICS):
                row = ci.loc[length, contrast_id, metric]
                a.close([row.delta, row.CI95_low, row.CI95_high],
                        [point[ci_ix, metric_ix], bounds[0, ci_ix, metric_ix], bounds[1, ci_ix, metric_ix]],
                        "independent point and percentile95 CI")
                a.check(row.repetitions == 2000 and row.interval_role ==
                        ("primary_block" if length == 12 else "block_sensitivity") and row.metric_role ==
                        ("primary" if metric_ix == 0 and contrast_id in family else "secondary"), "CI role/repetition declarations")
    loss_month = np.mean((prediction-truth[None, None])**2, axis=(1, 2))
    monthly = apply_contrasts(loss_month)
    annual = monthly.reshape(7, 8, 12).sum(axis=2)/96
    annual_export = indexed(a.csv(directory / "annual_loss_contributions.csv"), ["contrast_id", "year"], a, "annual contributions")
    a.check(set(annual_export.index) == set(itertools.product(ids, range(2017, 2025))), "56 annual contribution rows")
    tests = indexed(a.csv(directory / "annual_MSE_sign_tests.csv"), ["contrast_id"], a, "annual sign test")
    a.check(set(tests.index) == set(ids) and len(tests) == 7, "seven annual sign tests")
    sign_vectors = np.array(list(itertools.product((-1., 1.), repeat=8)))
    a.check(sign_vectors.shape == (256, 8), "all256 sign assignments")
    pvalues = {}
    for j, contrast_id in enumerate(ids):
        a.close(annual_export.loc[contrast_id, "contribution_to_full_MSE_delta"].to_numpy(), annual[j],
                "independent annual loss contributions")
        a.close(annual_export.loc[contrast_id, "annual_mean_delta"].to_numpy(), 8*annual[j],
                "annual mean contrast loss")
        observed = annual[j].sum()
        a.close(observed, point[j, 0], "annual contributions sum to principal MSE contrast")
        null = sign_vectors @ annual[j]
        # Frozen evaluator uses this absolute tie tolerance at the observed boundary.
        value = float(np.count_nonzero(np.abs(null) >= abs(observed)-1e-10)/256)
        pvalues[contrast_id] = value
        row = tests.loc[contrast_id]
        a.close(row.delta_macro_MSE_mm2, observed, "annual observed MSE delta")
        a.close(row.two_sided_p, value, "exact annual sign p", atol=0., rtol=0.)
        a.check(row.n_annual_blocks == 8 and row.sign_assignments == 256, "annual sign count declarations")
    ordered = sorted(ids, key=pvalues.get)
    adjusted, running = {}, 0.
    for rank, contrast_id in enumerate(ordered):
        running = max(running, min(1., (7-rank)*pvalues[contrast_id]))
        adjusted[contrast_id] = running
    for contrast_id in ids:
        row = tests.loc[contrast_id]
        a.close(row.holm_p_7_sensitivity, adjusted[contrast_id], "independent Holm7 sensitivity p", atol=0., rtol=0.)
        a.check(row.holm_p_7_sensitivity >= 7*2/256, "Holm7 resolution precludes05rejection")
    ordered6 = sorted(family, key=pvalues.get)
    adjusted6, running = {}, 0.
    for rank, contrast_id in enumerate(ordered6):
        running = max(running, min(1., (6-rank)*pvalues[contrast_id]))
        adjusted6[contrast_id] = running
    for contrast_id in ids:
        row = tests.loc[contrast_id]
        if contrast_id in family:
            a.close(row.holm_p_6_primary, adjusted6[contrast_id], "independent Holm6 primary p", atol=0., rtol=0.)
            a.check(bool(row.primary_reject_05) == (adjusted6[contrast_id] < .05) and
                    row.contrast_role == "amended_primary_sharing", "Holm6 decision/primary classification")
        else:
            a.check(pd.isna(row.holm_p_6_primary) and not bool(row.primary_reject_05) and
                    row.contrast_role == "secondary_proximal_context", "S07 secondary classification")
    a.details["uncertainty_recomputation"] = {
        "joint_resamples_regenerated_and_replayed": 6000, "contrast_draw_rows": 42000,
        "draw_metric_values": 168000, "CI_rows": 84, "annual_contribution_rows": 56,
        "annual_tests": 7, "sign_assignments_per_test": 256,
        "Holm_primary_family_size": 6, "Holm_sensitivity_family_size": 7,
        "interaction_draws_checked": 6000, "blocks": [12, 6, 24], "repetitions_per_block": 2000,
        "annual_boundary_tie_tolerance": 1e-10,
        "point_MSE_deltas": {ids[j]: float(point[j, 0]) for j in range(7)},
        "exact_p": pvalues, "holm_p_7_sensitivity": adjusted, "holm_p_6_primary": adjusted6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Experiment benchmark directory")
    parser.add_argument("--output", required=True, type=Path, help="Requested JSON QA receipt path")
    args = parser.parse_args()
    audit = Audit(args.root)
    status, failure = "PASS", None
    started = datetime.now(timezone.utc).isoformat()
    try:
        protocol, jobs, choices = check_identity(audit)
        clients, raw, stats, climatology, mean, sd, central_bytes = raw_training(audit)
        selected = inspect_fits(audit, protocol, jobs, choices, clients, raw, stats, mean, sd, central_bytes)
        primary, models, seeds, clients, months = principal_predictions(
            audit, protocol, jobs, clients, raw, stats, climatology, selected)
        by_seed = score_exports(audit, primary)
        uncertainty(audit, protocol, primary, models, seeds, clients, months, by_seed)
    except Exception as exc:
        status = "FAIL"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc(file=sys.stderr)
    receipt = {
        "status": status, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(audit.root), "script_sha256": audit.sha(Path(__file__)),
        "independent": True, "production_helpers_imported": False,
        "training_or_inference_performed": False,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "pandas": pd.__version__, "torch": torch.__version__},
        "successful_assertions": audit.checks, "assertions_by_category": dict(audit.categories),
        "numeric_scalars_compared": audit.scalar_comparisons, "distinct_files_hashed": audit.hash_count,
        "float64_comparison": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL,
            "max_absolute_error": audit.max_abs_float_error,
            "note": "Round-trip CSV parsing. Exact index/count/p-value checks; explicit float32 forward-error bound only for saved model averaging."},
        **audit.details,
    }
    if failure:
        receipt["failure"] = failure
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": status, "receipt": str(args.output.resolve()),
                      "assertions": audit.checks, "numeric_scalars": audit.scalar_comparisons}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
