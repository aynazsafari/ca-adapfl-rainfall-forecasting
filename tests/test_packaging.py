"""Packaging and preparation checks; no empirical fitting and no accepted writes.

Run with the benchmark's Python environment, without pytest:
    python tests/test_packaging.py --repo-root . -v
The original CLI is copied into temporary fixtures. Real primary preparation is
executed only against authenticated temporary copies of its code and inputs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import uuid
import warnings
import zipfile

sys.dont_write_bytecode = True
REPO_ROOT: Path | None = None


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_repository(value: str | Path | None = None) -> Path:
    if value:
        candidates = [Path(value).resolve()]
    elif os.environ.get("CAADAPFL_REPO_ROOT"):
        candidates = [Path(os.environ["CAADAPFL_REPO_ROOT"]).resolve()]
    else:
        candidates = []
        for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
            candidates += [parent, parent / "outputs/13_github_benchmark/CAAdapFL"]
    for candidate in candidates:
        if (candidate / "benchmark.py").is_file() and (
            candidate / "metadata/frozen_source_inventory.json"
        ).is_file():
            return candidate
    raise RuntimeError("Cannot locate benchmark.py; pass --repo-root PATH")


def repository() -> Path:
    global REPO_ROOT
    if REPO_ROOT is None:
        REPO_ROOT = resolve_repository()
    return REPO_ROOT


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def import_file(path: Path):
    name = "_packaging_check_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_process(args, cwd: Path, timeout=120):
    return subprocess.run(
        [sys.executable, *map(str, args)], cwd=cwd,
        env=dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1"),
        text=True, capture_output=True, timeout=timeout,
    )


class FixtureTests(unittest.TestCase):
    """Every mutation in these tests is confined to one disposable fixture."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="caadapfl_packaging_")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        shutil.copyfile(repository() / "benchmark.py", self.root / "benchmark.py")
        write_json(self.root / "metadata/frozen_source_inventory.json", {
            "files": [], "omissions": [],
        })
        write_json(self.root / "metadata/release_assets.json", {"assets": []})
        self.cli = import_file(self.root / "benchmark.py")
        self.assets = self.base / "assets"
        self.assets.mkdir()

    def install(self, private=None):
        args = types.SimpleNamespace(assets=self.assets, private_inputs=private)
        with contextlib.redirect_stdout(io.StringIO()):
            self.cli.install(args)

    def asset(self, name, members, *, target="benchmarks/accepted", prefix=""):
        path = self.assets / name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for member, content in members:
                    archive.writestr(member, content)
        entry = dict(file=name, bytes=path.stat().st_size,
                     sha256=digest(path.read_bytes()), target=target,
                     strip_prefix=prefix, members=len(members), omitted_inputs=[])
        manifest_path = self.root / "metadata/release_assets.json"
        manifest = read_json(manifest_path)
        manifest["assets"].append(entry)
        write_json(manifest_path, manifest)
        return entry

    def frozen(self, relative, content=b"original source\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest_path = self.root / "metadata/frozen_source_inventory.json"
        manifest = read_json(manifest_path)
        manifest["files"].append(dict(path=relative, bytes=len(content),
                                      sha256=digest(content)))
        write_json(manifest_path, manifest)
        return path

    def train(self, suite="primary", *, execute=True, workdir=None):
        self.cli.train(types.SimpleNamespace(
            suite=suite, execute=execute, workers=1,
            workdir=workdir or self.base / "fresh_training",
        ))


class SafeExtractionTests(FixtureTests):
    def test_portable_relative_member_is_admitted(self):
        self.assertEqual(self.cli.safe(self.root, "one/two.txt"),
                         self.root / "one/two.txt")

    def test_unsafe_members_are_refused(self):
        for member in ["../escape.txt", "a/../../escape.txt", "/escape.txt",
                       "C:escape.txt", "C:/escape.txt", "a\\escape.txt", ""]:
            with self.subTest(member=member):
                with self.assertRaises(ValueError):
                    self.cli.safe(self.root, member)
        self.assertFalse((self.base / "escape.txt").exists())

    def test_zip_traversal_is_refused_without_escape(self):
        self.asset("traversal.zip", [("../../../escape.txt", b"escape")])
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse((self.base / "escape.txt").exists())
        self.assertFalse((self.root / "escape.txt").exists())

    def test_archive_target_must_remain_inside_repository(self):
        self.asset("bad_target.zip", [("escape.txt", b"escape")],
                   target="../outside")
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse((self.base / "outside").exists())

    def test_asset_path_must_remain_inside_asset_directory(self):
        entry = self.asset("source.zip", [("safe.txt", b"safe")])
        shutil.copyfile(self.assets / "source.zip", self.base / "source.zip")
        entry["file"] = "../source.zip"
        write_json(self.root / "metadata/release_assets.json", {"assets": [entry]})
        with self.assertRaises(ValueError):
            self.install()
        self.assertFalse((self.root / "benchmarks").exists())

    def test_all_asset_hashes_checked_before_any_extraction(self):
        self.asset("first.zip", [("first.txt", b"authenticated")])
        self.asset("second.zip", [("second.txt", b"authenticated")])
        with (self.assets / "second.zip").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "authentication"):
            self.install()
        self.assertFalse((self.root / "benchmarks").exists())

    def test_equal_size_asset_with_changed_hash_is_refused(self):
        self.asset("corrupted.zip", [("evidence.txt", b"accepted")])
        archive = self.assets / "corrupted.zip"
        content = bytearray(archive.read_bytes())
        content[len(content) // 2] ^= 1
        archive.write_bytes(content)
        with self.assertRaisesRegex(RuntimeError, "authentication"):
            self.install()
        self.assertFalse((self.root / "benchmarks").exists())

    def test_changed_existing_evidence_is_never_overwritten(self):
        self.asset("one.zip", [("runs/evidence.txt", b"accepted")])
        target = self.root / "benchmarks/accepted/runs/evidence.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"local changed evidence")
        before = (target.read_bytes(), target.stat().st_mtime_ns)
        with self.assertRaisesRegex(RuntimeError, "Refusing changed"):
            self.install()
        self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))

    def test_identical_install_is_idempotent(self):
        self.asset("one.zip", [("runs/evidence.txt", b"accepted")])
        self.install()
        target = self.root / "benchmarks/accepted/runs/evidence.txt"
        before = (target.read_bytes(), target.stat().st_mtime_ns)
        self.install()
        self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))

    def test_duplicate_zip_member_is_refused(self):
        self.asset("duplicate.zip", [("same.txt", b"same"), ("same.txt", b"same")])
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            self.install()

    def test_unexpected_archive_prefix_is_refused(self):
        self.asset("prefix.zip", [("wrong/file.txt", b"content")], prefix="expected/")
        with self.assertRaisesRegex(RuntimeError, "Unexpected member root"):
            self.install()
        self.assertFalse((self.root / "benchmarks").exists())

    def private_zip(self, entries):
        path = self.base / "private_inputs.zip"
        with zipfile.ZipFile(path, "w") as archive:
            manifest = {"files": [dict(path=name, bytes=len(content),
                                       sha256=digest(content)) for name, content in entries]}
            archive.writestr("PRIVATE_INPUTS_MANIFEST.json", json.dumps(manifest))
            for name, content in entries:
                archive.writestr(name, content)
        return path

    def declare_private_input(self, relative, content):
        manifest_path = self.root / "metadata/frozen_source_inventory.json"
        manifest = read_json(manifest_path)
        manifest["omissions"].append(dict(path=relative, bytes=len(content),
            sha256=digest(content), reason="IMD input: redistribution permission not established"))
        write_json(manifest_path, manifest)

    def test_private_sidecar_cannot_add_undeclared_code(self):
        relative = "scripts/unapproved.py"
        with self.assertRaisesRegex(RuntimeError, "Undeclared private input"):
            self.install(self.private_zip([(relative, b"unapproved code")]))
        self.assertFalse((self.root / relative).exists())

    def test_declared_private_input_can_be_restored_without_replacement(self):
        relative, content = "data/imd_supplement.csv", b"authorized local input"
        self.declare_private_input(relative, content)
        target = self.root / "benchmarks/accepted" / relative
        sidecar = self.private_zip([("benchmarks/accepted/" + relative, content)])
        self.install(sidecar)
        before = (target.read_bytes(), target.stat().st_mtime_ns)
        self.install(sidecar)
        self.assertEqual(before, (target.read_bytes(), target.stat().st_mtime_ns))
        target.write_bytes(b"changed local input")
        with self.assertRaises((AssertionError, RuntimeError)):
            self.install(sidecar)
        self.assertEqual(target.read_bytes(), b"changed local input")

    def test_corrupted_private_input_is_refused(self):
        relative = "data/imd_supplement.csv"
        self.declare_private_input(relative, b"expected")
        sidecar = self.base / "corrupt_private_input.zip"
        with zipfile.ZipFile(sidecar, "w") as archive:
            archive.writestr("PRIVATE_INPUTS_MANIFEST.json", json.dumps({"files": [dict(
                path="benchmarks/accepted/" + relative,
                bytes=len(b"expected"), sha256=digest(b"expected"))]}))
            archive.writestr("benchmarks/accepted/" + relative, b"tampered")
        with self.assertRaises((AssertionError, RuntimeError)):
            self.install(sidecar)
        self.assertFalse((self.root / "benchmarks/accepted" / relative).exists())

    def test_doctor_refuses_missing_or_changed_frozen_source(self):
        source = self.frozen("benchmarks/accepted/caadapfl/example.py")
        for change in ["modify", "remove"]:
            with self.subTest(change=change):
                if change == "modify":
                    source.write_bytes(b"changed")
                else:
                    source.unlink()
                with self.assertRaisesRegex(RuntimeError, "missing/changed"):
                    self.cli.doctor(types.SimpleNamespace())


class FreshTrainingTests(FixtureTests):
    def test_cli_help_documents_explicit_execution_gate(self):
        for command in [[], ["train"], ["install"]]:
            with self.subTest(command=command):
                result = run_process([self.root / "benchmark.py", *command, "--help"], self.root)
                self.assertEqual(result.returncode, 0, result.stderr)
                if command == ["train"]:
                    for flag in ["--execute", "--workdir", "--suite", "--workers"]:
                        self.assertIn(flag, result.stdout)

    def test_all_suite_plans_leave_work_directory_absent(self):
        for suite in ["primary", "sources", "capacity", "sharing"]:
            with self.subTest(suite=suite):
                workdir = self.base / ("plan_" + suite)
                result = run_process([self.root / "benchmark.py", "train", "--suite", suite,
                                      "--workdir", workdir], self.root)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Plan:", result.stdout)
                self.assertIn("--execute", result.stdout)
                self.assertFalse(workdir.exists())

    def test_existing_training_directory_is_refused_unchanged(self):
        workdir = self.base / "existing"
        workdir.mkdir()
        marker = workdir / "marker.txt"
        marker.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.train(workdir=workdir)
        self.assertEqual(marker.read_bytes(), b"preserve")

    def test_training_inside_evidence_root_is_refused_precreate(self):
        for suite in ["accepted", "sharing"]:
            with self.subTest(root=suite):
                workdir = self.root / "benchmarks" / suite / "new_training"
                with self.assertRaisesRegex(RuntimeError, "outside preserved"):
                    self.train(workdir=workdir)
                self.assertFalse(workdir.exists())

    def test_changed_training_source_is_refused_precreate(self):
        source = self.frozen("benchmarks/accepted/caadapfl/example.py")
        source.write_bytes(b"changed source")
        workdir = self.base / "fresh_training"
        with mock.patch.object(self.cli, "invoke_module") as invocation:
            with self.assertRaisesRegex(RuntimeError, "changed frozen training source"):
                self.train(workdir=workdir)
            invocation.assert_not_called()
        self.assertFalse(workdir.exists())

    def test_missing_training_source_is_refused_precreate(self):
        self.frozen("benchmarks/accepted/caadapfl/example.py").unlink()
        with self.assertRaisesRegex(RuntimeError, "frozen training source"):
            self.train()
        self.assertFalse((self.base / "fresh_training").exists())

    def test_missing_private_imd_source_is_refused_precreate(self):
        with mock.patch.object(self.cli, "invoke") as invocation:
            with self.assertRaisesRegex(FileNotFoundError, "local IMD"):
                self.train("sources")
            invocation.assert_not_called()
        self.assertFalse((self.base / "fresh_training").exists())

    def test_primary_dispatches_as_a_package_module(self):
        self.frozen("benchmarks/accepted/caadapfl/example.py")
        workdir = self.base / "fresh_training"
        with mock.patch.object(self.cli, "invoke_module") as module, \
                mock.patch.object(self.cli, "invoke") as script:
            self.train(workdir=workdir)
        script.assert_not_called()
        self.assertEqual(module.call_count, 2)
        prepare, train = module.call_args_list
        self.assertEqual(prepare.args[:2], ("caadapfl.runner", "prepare"))
        self.assertEqual(train.args[:3], ("caadapfl.runner", "run", "--execute"))
        self.assertEqual(prepare.kwargs["cwd"], workdir)
        self.assertEqual(train.kwargs["cwd"], workdir)
        self.assertIn("--pvalues", train.args)
        self.assertEqual((workdir / "caadapfl/example.py").read_bytes(), b"original source\n")

    def test_module_invocation_uses_current_interpreter_and_correct_cwd(self):
        with mock.patch.object(self.cli.subprocess, "run") as process:
            self.cli.invoke_module("caadapfl.runner", "prepare", "--output", "fresh", cwd=self.base)
        call = process.call_args
        self.assertEqual(call.args[0], [sys.executable, "-m", "caadapfl.runner", "prepare", "--output", "fresh"])
        self.assertEqual(call.kwargs["cwd"], self.base)
        self.assertEqual(call.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertTrue(call.kwargs["check"])

    def test_fresh_primary_prepares_real_inputs_and_stops_before_fitting(self):
        original = repository()
        inventory = {entry["path"]: entry for entry in read_json(
            original / "metadata/frozen_source_inventory.json")["files"]}
        relative_paths = sorted("benchmarks/accepted/" + p.relative_to(
            original / "benchmarks/accepted").as_posix() for p in (
                original / "benchmarks/accepted/caadapfl").glob("*.py"))
        relative_paths += ["benchmarks/accepted/data/" + name for name in
                           ["monthly.csv", "indices.csv", "manifest.json"]]
        snapshots = {}
        for relative in relative_paths:
            source = original / relative
            content = source.read_bytes()
            self.assertEqual(digest(content), inventory[relative]["sha256"], relative)
            snapshots[relative] = (digest(content), source.stat().st_mtime_ns)
            self.frozen(relative, content)
        workdir = self.base / "fresh_primary"
        calls = []
        actual_invoke = self.cli.invoke_module

        class StopBeforeFitting(Exception):
            pass

        def prepare_only(module, *arguments, cwd):
            calls.append((module, arguments, cwd))
            if arguments[0] != "prepare":
                raise StopBeforeFitting()
            actual_invoke(module, *arguments, cwd=cwd)

        with mock.patch.object(self.cli, "invoke_module", side_effect=prepare_only), \
                mock.patch.object(self.cli, "invoke") as script:
            with self.assertRaises(StopBeforeFitting):
                self.train(workdir=workdir)
        script.assert_not_called()
        self.assertEqual([call[1][0] for call in calls], ["prepare", "run"])
        receipt = read_json(workdir / "runs/revision_v2/prepared.json")
        self.assertEqual(receipt["model_fits"], 0)
        self.assertEqual(receipt["evaluation_predictions"], 0)
        for scenario, train_count in [("MAIN", 299), ("LATENCY_REFERENCE", 298), ("LATENCY_LAG1", 298)]:
            self.assertEqual(receipt["scenarios"][scenario]["role_counts"],
                             dict(train=train_count, monitor=35, validation=71, evaluation=96))
        run = workdir / "runs/revision_v2"
        for path in (run / "datasets").glob("*_evaluation_features.npz"):
            with zipfile.ZipFile(path) as arrays:
                self.assertFalse(any(name.endswith(("::y.npy", "::y_z.npy", "::heavy.npy"))
                                     for name in arrays.namelist()), path.name)
        for relative, expected in receipt["files"].items():
            self.assertEqual(digest((run / relative).read_bytes()), expected, relative)
        self.assertFalse((run / "fits").exists())
        self.assertFalse((run / "choices_frozen.json").exists())
        self.assertFalse((run / "evaluation").exists())
        before = {str(p.relative_to(run)): digest(p.read_bytes()) for p in run.rglob("*") if p.is_file()}
        result = run_process(["-m", "caadapfl.runner", "prepare", "--monthly", "data/monthly.csv",
                              "--indices", "data/indices.csv", "--manifest", "data/manifest.json",
                              "--output", "runs/revision_v2"], workdir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, {str(p.relative_to(run)): digest(p.read_bytes())
                                 for p in run.rglob("*") if p.is_file()})
        result = run_process(["-m", "caadapfl.runner", "plan", "--output", "runs/symbolic_plan"], workdir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in (workdir / "runs/symbolic_plan").iterdir()},
                         {"protocol.json", "schedule.json"})
        result = run_process(["-m", "caadapfl.runner", "--help"], workdir)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare", result.stdout)
        for relative, snapshot in snapshots.items():
            source = original / relative
            self.assertEqual(snapshot, (digest(source.read_bytes()), source.stat().st_mtime_ns), relative)


class IndependentFormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formulas = import_file(repository() / "benchmarks/accepted/tools/qa/metric_reference.py")

    def test_perfect_regression(self):
        result = self.formulas.regression_reference([1, 2, 3], [1, 2, 3])
        for metric in ["R2", "NSE", "Pearson_r", "KGE_2009"]:
            self.assertEqual(result[metric], 1)
        for metric in ["RMSE_mm", "MAE_mm", "PBIAS_signed_percent", "forecast_bias_signed_percent"]:
            self.assertEqual(result[metric], 0)

    def test_signed_rainfall_bias_convention(self):
        result = self.formulas.regression_reference([1, 2, 3], [2, 3, 4])
        self.assertEqual(result["RMSE_mm"], 1)
        self.assertEqual(result["MAE_mm"], 1)
        self.assertEqual(result["R2"], -0.5)
        self.assertEqual(result["PBIAS_signed_percent"], -50)
        self.assertEqual(result["forecast_bias_signed_percent"], 50)

    def test_degenerate_regression_is_undefined_without_floor(self):
        result = self.formulas.regression_reference([2, 2, 2], [1, 2, 3])
        for metric in ["R2", "NSE", "Pearson_r", "KGE_2009"]:
            self.assertTrue(math.isnan(result[metric]), metric)
        with self.assertRaises(ValueError):
            self.formulas.regression_reference([1, float("nan")], [1, 2])

    def test_strict_q80_event_and_inclusive_probability_threshold(self):
        result = self.formulas.heavy_reference([10, 11], [10, 11], 10)
        self.assertEqual([result[key] for key in ["actual_heavy_n", "heavy_TP", "heavy_TN"]], [1, 1, 1])
        result = self.formulas.heavy_reference([10, 11], [10, 11], 10, [0.5, 0.49])
        self.assertEqual([result[key] for key in ["heavy_TP", "heavy_FP", "heavy_FN", "heavy_TN"]], [0, 1, 1, 0])
        self.assertEqual(result["heavy_F1"], 0)
        self.assertEqual(result["tail80_RMSE_mm"], 0)
        with self.assertRaises(ValueError):
            self.formulas.heavy_reference([10, 11], [10, 11], 10, [0.5, float("nan")])


if __name__ == "__main__":
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--repo-root", type=Path)
    selected, unittest_args = selector.parse_known_args()
    REPO_ROOT = resolve_repository(selected.repo_root)
    print("Testing candidate:", REPO_ROOT, flush=True)
    unittest.main(argv=[sys.argv[0], *unittest_args])
