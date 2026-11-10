"""Authenticate frozen evidence and replay selected saved states; never fit models.

Run from any directory. --output is a fresh directory outside the accepted
benchmark. Original runtime guards remain strict: this is an exact reference
environment replay, not a promise of bitwise equality on other hardware.
Primary classical/router rainfall comparisons additionally report a 1e-10 mm
absolute tolerance, matching the accepted independent numerical QA boundary.
For portable arithmetic checks use the separately supplied independent QA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

# Importing frozen modules must not create cache files inside accepted evidence.
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(root, name):
    """Normalize recorded Windows separators and reject escaping identities."""
    root = Path(root).resolve()
    target = (root / str(name).replace("\\", "/")).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Manifest path escapes its evidence root: " + str(name))
    return target


class Audit:
    def __init__(self):
        self.bound = {}

    def check(self, path, expected):
        path = Path(path).resolve()
        if path in self.bound:
            if self.bound[path] != expected:
                raise ValueError("Conflicting hash bindings: " + str(path))
            return
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Missing or changed bound evidence: " + str(path))
        self.bound[path] = expected

    def files(self, root, files):
        for name, expected in files.items():
            self.check(relative(root, name), expected)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def job_core(value):
    # Original launchers define relocation-only fields as provenance.
    if isinstance(value, dict):
        return {k: job_core(v) for k, v in value.items() if k not in {"root", "package_root"}}
    if isinstance(value, list):
        return [job_core(v) for v in value]
    return value


def authenticate_fits(audit, run, jobs, choices, short=False):
    key = "job_sha256" if short else "id"
    require(len({j[key] for j in jobs}) == len(jobs), "Duplicate frozen fit identity")
    require(set(choices["fit_complete_hashes"]) == {j[key] for j in jobs},
            "Fit completion bindings do not cover the frozen jobs exactly")
    for job in jobs:
        folder = run / "fits" / (job[key][:16] if short else job[key])
        audit.check(folder / "complete.json", choices["fit_complete_hashes"][job[key]])
        complete = read(folder / "complete.json")
        require(job_core(complete["job"]) == job_core(job), "Frozen job/configuration mismatch")
        audit.files(folder, complete["files"])


def compare(replayed, expected, classical_models=()):
    import numpy as np
    keys = ["model", "seed", "client", "target_month"]
    for name, table in [("replayed", replayed), ("released", expected)]:
        require(not table.duplicated(keys).any(), "Duplicate " + name + " forecast keys")
    a = replayed.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    require(len(a) == len(b) and a[keys].equals(b[keys]), "Forecast key/calendar mismatch")
    columns = ["prediction", "y"]
    for optional in ["heavy_probability", "checkpoint_round"]:
        if optional in a.columns and optional in b.columns:
            columns.append(optional)
    differences, comparisons = {}, {}
    differing_rows = np.zeros(len(a), dtype=bool)
    classical_models = set(classical_models)
    for col in columns:
        x, y = a[col].to_numpy(float), b[col].to_numpy(float)
        equal = (x == y) | (np.isnan(x) & np.isnan(y))
        differing_rows |= ~equal
        valid = np.isfinite(x) & np.isfinite(y)
        differences[col] = float(np.max(np.abs(x[valid] - y[valid]))) if valid.any() else None
        if not equal.all():
            allowed = a.model.isin(classical_models).to_numpy() if col == "prediction" else np.zeros(len(a), dtype=bool)
            accepted = equal | (allowed & valid & (np.abs(x-y) <= 1e-10))
            bad = np.flatnonzero(~accepted)
            require(not len(bad), "Saved-state replay differs outside declared policy in " + col +
                    "; max absolute difference=" + str(differences[col]) +
                    "; first failing key=" + (str(a.loc[bad[0], keys].to_dict()) if len(bad) else "none"))
        comparisons[col] = {"exact": bool(equal.all()), "nonexact_rows": int((~equal).sum()),
                            "nonexact_models": sorted(set(a.loc[~equal, "model"]))}
    nonexact_models = set(a.loc[differing_rows, "model"])
    return {"forecast_rows_checked": len(a), "forecast_rows_exact": int((~differing_rows).sum()),
            "models_checked": int(a.model.nunique()), "models_exact": int(a.model.nunique())-len(nonexact_models),
            "maximum_absolute_differences": differences, "column_comparisons": comparisons,
            "all_keyed_values_exact": bool(not differing_rows.any()),
            "all_keyed_values_within_declared_comparison": True,
            "comparison_policy": {"all_other_values": "exact", "classical_router_rainfall_atol_mm": 1e-10 if classical_models else 0.,
                                  "rtol": 0., "tolerance_eligible_models": sorted(classical_models)}}


def runtime_matches(recorded, runner):
    actual = runner.runtime_environment()
    require(actual == recorded,
            "Reference runtime differs from the frozen identity. No guard was bypassed. "
            "Use the recorded Python/packages/OS/PyTorch build for exact saved-state replay; "
            "independent arithmetic QA supports broader environments.")
    return actual


def neural_rows(fit, data, neural, model, seed, identity, truth, climate_free=False):
    import numpy as np
    rows = []
    for client, d in data.items():
        drivers = np.zeros_like(d["drivers"]) if climate_free else d["drivers"]
        batch = neural.FeatureBatch(d["scaled_x"], d["scaled_climate"], drivers,
                                    tuple(month + "-01" for month in d["target_month"]))
        prediction = fit.predict(client, batch, identity)
        require(list(d["target_month"]) == [k[:7] for k in prediction.keys], "Inference calendar changed")
        for key, rainfall, probability in zip(prediction.keys, prediction.rainfall_mm,
                                              prediction.heavy_probability):
            month = key[:7]
            rows.append(dict(model=model, seed=seed, client=client, target_month=month,
                             y=float(truth.loc[(client, month)]), prediction=float(rainfall),
                             heavy_probability=float(probability), checkpoint_round=prediction.checkpoint_round))
    return rows


def capacity(root, audit, runner):
    import numpy as np
    import pandas as pd
    neural = importlib.import_module("caadapfl.neural_revision")
    run = root / "runs/revision_v3"
    identity = read(run / "run_identity.json")
    runtime = runtime_matches(identity["environment"], runner)
    audit.files(root, identity["code"])
    audit.files(root, identity["datasets"])
    audit.files(root, read(run / "prepared.json")["files"])
    audit.check(run / "protocol.json", identity["protocol_sha256"])
    choices = read(run / "choices_frozen.json")
    audit.check(run / "run_identity.json", choices["run_identity_sha256"])
    require(choices["evaluation_label_used"] is False, "Evaluation labels used for frozen choices")
    jobs = choices["all_jobs"]
    require(len(jobs) == 100 and read(run / "fit_complete.json")["unique_fits"] == 100,
            "Capacity study is not a completed 100-fit study")
    authenticate_fits(audit, run, jobs, choices)
    complete = read(run / "evaluation/complete.json")
    require(complete["status"] == "COMPLETE", "Evaluation is incomplete")
    audit.check(run / "choices_frozen.json", complete["choices_sha256"])
    audit.check(run / "evaluation_prelaunch_identity.json", complete["evaluation_prelaunch_sha256"])
    launch = read(run / "evaluation_prelaunch_identity.json")
    audit.files(root, launch["evaluation_code"])
    audit.check(run / "archive_analysis_protocol.json", complete["archive_protocol_sha256"])
    audit.check(run / "archive_feature_CI_protocol.json", complete["archive_feature_protocol_sha256"])
    audit.files(run / "evaluation", complete["files"])
    truth = pd.read_csv(run / "datasets/SEALED_evaluation_truth.csv", float_precision="round_trip").set_index(
        ["client", "target_month"])["observed_mm"]
    frames, cache = [], {}
    datasets = {}
    for i, job in enumerate(jobs, 1):
        fit = neural.load_fit(run / "fits" / job["id"], load_round_history=False)
        cohort = (job["scenario"], job["group"])
        if cohort not in datasets:
            datasets[cohort] = runner._load_dataset(run, *cohort, evaluation=True)[0]
        part = pd.DataFrame(neural_rows(fit, datasets[cohort], neural, job["label"],
                                       job["seed"], "global", truth))
        cache[job["id"]] = part
        frames.append(part)
        if i % 20 == 0:
            print(f"Capacity selected-state inference: {i}/100 fits", flush=True)
    inferred_rows = sum(len(f) for f in frames)
    for alias in choices["aliases"]:
        part = cache[alias["job"]["id"]].copy()
        part["model"] = alias["label"]
        frames.append(part)
    base = pd.concat(frames, ignore_index=True)
    for method in ["CAAA", "FEDAVG", "FEDPROX", "CENTRAL"]:
        part = base[base.model == method + "_RAW"].copy()
        require(len(part) == 2400, "RAW alias population mismatch")
        part["prediction"] = np.maximum(part.prediction, 0.)
        part["model"] = method + "_CLIPPED_RAW"
        frames.append(part)
        part = base[base.model == method + "_" + choices["selected_common_loss"]["variant"]].copy()
        require(len(part) == 2400, "Selected-loss alias population mismatch")
        part["model"] = method + "_SELECTED_LOSS"
        frames.append(part)
    generated_rows = sum(len(f) for f in frames)
    archived = read(run / "archive_analysis_protocol.json")
    feature = read(run / "archive_feature_CI_protocol.json")
    legacy_path = root / "runs/accepted_reference/raw_predictions.csv"
    audit.check(legacy_path, archived["legacy_raw_sha256"])
    legacy = pd.read_csv(legacy_path, float_precision="round_trip")
    inverse = {v: k for k, v in {**archived["legacy_label_mapping"],
                                 **feature["additional_legacy_label_mapping"]}.items()}
    keep = set(inverse) | set(archived["router_labels"])
    old = legacy[legacy.model.isin(keep)]
    require(set(old.model) == keep, "Authenticated archival comparison labels incomplete")
    archived_rows = 0
    for label, part in old.groupby("model", sort=True):
        part = part.copy()
        canonical = np.array([truth.loc[(c, m)] for c, m in zip(part.client, part.target_month)])
        require(float(np.max(np.abs(part.y.to_numpy() - canonical))) < 1e-10,
                "Archival target differs from the frozen canonical target")
        part["y"] = canonical
        part["model"] = inverse.get(label, label)
        frames.append(part)
        archived_rows += len(part)
    result = pd.concat(frames, ignore_index=True)
    expected = pd.read_csv(run / "evaluation/raw_predictions.csv", float_precision="round_trip")
    receipt = compare(result, expected)
    receipt.update(status="PASS", unique_fits_replayed=100, selected_fit_forecast_rows=inferred_rows,
                   alias_projection_rows=generated_rows-inferred_rows, authenticated_archival_rows=archived_rows,
                   archival_scope="Archived controls matched to their independently hash-bound parent forecasts; their models are not loaded in this suite.",
                   runtime=runtime, raw_predictions_sha256=sha(run / "evaluation/raw_predictions.csv"))
    return result, receipt


def primary(root, audit, runner):
    import joblib
    import pandas as pd
    neural = importlib.import_module("caadapfl.neural_benchmark")
    run = root / "runs/revision_v2"
    runner._check_runtime_code(run)  # Original read-only source/runtime guard.
    identity = read(run / "run_identity.json")
    audit.files(root, {"caadapfl/" + name + ".py": h for name, h in identity["code"].items()})
    audit.files(run, read(run / "prepared.json")["files"])
    choices = read(run / "choices_frozen.json")
    require(choices["run_sha256"] == identity["run_sha256"], "Primary choices belong to another run")
    require(choices["evaluation_labels_accessed"] is False and
            choices["choices_including_checkpoints_and_router_weights_frozen"] is True,
            "Primary fitting/selection freeze is incomplete")
    require(len(choices["all_fit_jobs"]) == 283, "Primary accepted fit inventory is incomplete")
    authenticate_fits(audit, run, choices["all_fit_jobs"], choices, short=True)
    complete = read(run / "evaluation/complete.json")
    audit.check(run / "choices_frozen.json", complete["choices_frozen_sha256"])
    audit.files(run / "evaluation", complete["files"])
    # Match the frozen evaluator's default truth parser exactly. The released
    # output is read with round_trip below to preserve its saved float values.
    truth = pd.read_csv(run / "datasets/evaluation_truth.csv").set_index(
        ["client", "target_month"])["y"]
    datasets, predictions, frames = {}, {}, []
    def data_for(scenario, group):
        key = (scenario, group)
        if key not in datasets:
            data = runner._load_dataset(run, scenario, group, evaluation=True)[0]
            for c, d in data.items():
                d["y"] = truth.loc[(c, d["target_month"])].to_numpy(float)
            datasets[key] = data
        return datasets[key]
    used = set()
    for i, output in enumerate(choices["outputs"], 1):
        key = output["job_sha256"]
        folder = runner.fit_path(run, key)
        cached = (key, output.get("identity"), output.get("suite"))
        if cached not in predictions:
            if output["kind"] == "neural":
                fit = neural.load_fit(folder / "model", load_round_history=False)
                job = read(folder / "complete.json")["job"]
                frame = pd.DataFrame(neural_rows(fit, data_for(output["scenario"], output["group"]),
                    neural, output["label"], output["seed"], output["identity"], truth,
                    climate_free=job.get("climate_free", False)))
            elif output["kind"] == "classical":
                frame = joblib.load(folder / "model.joblib").predict(
                    data_for(output["scenario"], "FULL"), "2017-01", "2024-12")
            elif output["kind"] == "router":
                router, wide = joblib.load(folder / "model.joblib"), None
                for expert_id in output["expert_jobs"]:
                    used.add(expert_id)
                    expert = joblib.load(runner.fit_path(run, expert_id) / "model.joblib")
                    part = expert.predict(data_for("MAIN", "FULL"), "2017-01", "2024-12")
                    label = part.model.iloc[0]
                    part = part[["client", "target_month", "y", "prediction"]].rename(columns={"prediction": label})
                    wide = part if wide is None else wide.merge(part, on=["client", "target_month", "y"], validate="one_to_one")
                routed = router.predict(wide, "2017-01", "2024-12")
                frame = routed[["client", "target_month", "y", output["suite"]]].rename(columns={output["suite"]: "prediction"})
            else:
                raise ValueError("Unknown primary output kind: " + output["kind"])
            predictions[cached] = frame
            used.add(key)
        frame = predictions[cached].copy()
        frame["model"], frame["seed"] = output["label"], output["seed"]
        frames.append(frame)
        if i % 40 == 0:
            print(f"Primary inference: {i}/{len(choices['outputs'])} output identities", flush=True)
    baseline_key = choices["baseline_job_sha256"]
    used.add(baseline_key)
    baseline = joblib.load(runner.fit_path(run, baseline_key) / "model.joblib")
    for seed in runner.SEEDS:
        frames.append(baseline.predict(data_for("MAIN", "FULL"), "2017-01", "2024-12", seed))
    result = pd.concat(frames, ignore_index=True)
    expected = pd.read_csv(run / "evaluation/raw_predictions.csv", float_precision="round_trip")
    classical_models = {o["label"] for o in choices["outputs"] if o["kind"] in {"classical", "router"}}
    receipt = compare(result, expected, classical_models)
    receipt.update(status="PASS", accepted_fit_bundles_authenticated=283,
                   saved_model_bundles_used_for_outputs=len(used),
                   scope="Replays every released primary forecast using frozen neural/classical/router/baseline models. Development-only and OOF fit artifacts are authenticated; fitting is never repeated.",
                   runtime=identity["runtime_environment"], raw_predictions_sha256=sha(run / "evaluation/raw_predictions.csv"))
    return result, receipt


def sources(root, audit, runner):
    import pandas as pd
    sys.path.insert(0, str(root / "tools"))
    supplement = importlib.import_module("supplement_runner")
    neural = importlib.import_module("caadapfl.neural_benchmark")
    run = root / "runs/supplements_v2"
    identity = read(run / "run_identity.json")
    supplement.check_code(run, runner, runner._mods())  # Original strict read-only guard.
    audit.check(root / "tools/supplement_runner.py", identity["launcher_sha256"])
    audit.files(root, {"caadapfl/" + name + ".py": h for name, h in identity["module_code"].items()})
    prepared = read(run / "prepared.json")
    unavailable = {}
    for name, expected in prepared["files"].items():
        path = relative(run, name)
        if not path.is_file():
            case = path.name.split("_", 1)[0]
            require(case == "IMD", "Missing non-IMD prepared input: " + str(path))
            unavailable.setdefault(case, []).append(str(path.relative_to(run)))
        else:
            audit.check(path, expected)
    choices = read(run / "choices_frozen.json")
    require(choices["run_sha256"] == identity["run_sha256"] and prepared["run_sha256"] == identity["run_sha256"],
            "Source experiment identities differ")
    require(choices["all_checkpoints_frozen"] is True and choices["evaluation_labels_accessed"] is False,
            "Source fit/checkpoint freeze is incomplete")
    require(len(choices["jobs"]) == 90, "Source fit inventory is incomplete")
    authenticate_fits(audit, run, choices["jobs"], choices, short=True)
    complete = read(run / "evaluation/complete.json")
    require(complete["complete_fits"] == 90, "Source evaluation is incomplete")
    audit.files(run / "evaluation", complete["files"])
    frames, cache, truth_cache, replayed_fits = [], {}, {}, 0
    for case in supplement.CASES:
        required = [run / "datasets" / (case + "_evaluation_truth.csv")]
        for job in choices["jobs"]:
            if job["case"] == case:
                required += [run / "datasets" / (job["scenario"] + "_FULL" + suffix)
                             for suffix in ["_evaluation_features.npz", "_schema.json"]]
        missing = [str(p.relative_to(run)) for p in required if not p.is_file()]
        if missing:
            require(case == "IMD", "Missing non-IMD source cohort")
            unavailable.setdefault(case, []).extend(missing)
        if case in unavailable:
            print("IMD saved-state replay unavailable: restricted prepared inputs are absent.", flush=True)
            continue
        # The executed supplement evaluator used pandas' default truth parser.
        truth_cache[case] = pd.read_csv(required[0]).set_index(["client", "target_month"])["y"]
        for job in choices["jobs"]:
            if job["case"] != case:
                continue
            fit = neural.load_fit(supplement.fit_path(run, job["job_sha256"]) / "model", load_round_history=False)
            if job["scenario"] not in cache:
                cache[job["scenario"]] = runner._load_dataset(run, job["scenario"], "FULL", evaluation=True)[0]
            for endpoint in ["global", "personal"]:
                label = supplement.label(job["method"], endpoint, case, job["arm"])
                part = pd.DataFrame(neural_rows(fit, cache[job["scenario"]], neural, label,
                                               job["config"]["seed"], endpoint, truth_cache[case]))
                part["case"] = case
                frames.append(part)
            replayed_fits += 1
        print(f"Source selected-state inference complete: {case}", flush=True)
    require(bool(frames), "No source cohort can be replayed from available inputs")
    result = pd.concat(frames, ignore_index=True)
    expected = pd.read_csv(run / "evaluation/raw_predictions_all_cases.csv", float_precision="round_trip")
    expected = expected[expected.case.isin(result.case.unique())]
    receipt = compare(result, expected)
    receipt.update(status="PARTIAL_UNAVAILABLE" if unavailable else "PASS",
                   source_fit_bundles_authenticated=90, unique_fits_replayed=replayed_fits,
                   source_cases_replayed=sorted(result.case.unique()),
                   unavailable_cases={k: {"reason": "Restricted IMD prepared inputs are not in the public package; restore authenticated local sidecar inputs to complete replay.",
                                           "missing_files": sorted(set(v))} for k, v in unavailable.items()},
                   runtime=identity["runtime_environment"],
                   raw_predictions_sha256=sha(run / "evaluation/raw_predictions_all_cases.csv"))
    return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--suite", choices=["capacity", "primary", "sources"], required=True)
    parser.add_argument("--output", type=Path, help="Fresh output directory outside accepted benchmark")
    parser.add_argument("--accepted-root", type=Path, help="Optional original accepted source root for a local validation")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    root = args.accepted_root.resolve() if args.accepted_root else repo / "benchmarks/accepted"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() if args.output else repo / "work" / (args.suite + "_state_replay_" + stamp)
    require(root.is_dir(), "Accepted benchmark root does not exist: " + str(root))
    require(not output.is_relative_to(root), "Replay output must be outside accepted evidence")
    require(not output.exists(), "Replay output already exists; accepted or previous results are never replaced")
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    receipt = dict(status="FAIL", suite=args.suite, started_utc=datetime.now(timezone.utc).isoformat(),
                   accepted_root=str(root), output=str(output), replay_script_sha256=sha(Path(__file__)),
                   new_empirical_fits=0, accepted_evidence_written=False,
                   exact_match_scope="Strictly authenticated original runtime. Capacity/source and primary neural/baseline values require exact equality; primary classical/router rainfall permits absolute1e-10mm and reports every difference. Cross-hardware bitwise equality is not promised.")
    exit_code = 1
    try:
        sys.path.insert(0, str(root))
        runner = importlib.import_module("caadapfl.runner")
        require(Path(runner.__file__).resolve().is_relative_to(root), "Imported another caadapfl installation")
        predictions, result = {"capacity": capacity, "primary": primary, "sources": sources}[args.suite](root, audit, runner)
        predictions.to_csv(output / "replayed_predictions.csv", index=False, float_format="%.17g")
        receipt.update(result)
        receipt["replayed_predictions_sha256"] = sha(output / "replayed_predictions.csv")
        exit_code = 2 if result["status"] == "PARTIAL_UNAVAILABLE" else 0
    except Exception as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), authenticated_file_count=len(audit.bound))
    (output / "replay_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "suite": args.suite, "receipt": str(output / "replay_receipt.json"),
                      "new_empirical_fits": 0, "authenticated_file_count": len(audit.bound)}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
