"""Scientific invariants on synthetic fixtures; no study files or study scores.

Run from the installed repository: python -m unittest discover -s tests -p test_invariants.py -v
Only two tiny two-round fits are made. Source-panel tests create synthetic CSVs.
"""
from dataclasses import replace
from pathlib import Path
import copy
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from caadapfl import features as f
from caadapfl import neural_benchmark as n


def monthly_fixture():
    """Complete synthetic keys required by the public source-calendar gate."""
    months = pd.period_range("1981-01", "2024-12", freq="M")
    records = []
    for client_no, client in enumerate(f.CLIENTS):
        for ordinal, month in enumerate(months):
            records.append({"client": client, "month": str(month),
                            "rainfall_mm": 20 + client_no * 10 + ordinal * .2 + month.month,
                            "weather_toy": client_no + ordinal * .03})
    panel = pd.DataFrame(records)
    indices = pd.DataFrame({"month": months.astype(str)})
    for col_no, name in enumerate(f.INDEX_NAMES):
        indices[name] = np.sin(np.arange(len(months)) / (3 + col_no)) + col_no * .1
    return panel, indices


def labeled(keys, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(keys), 4, 3)).astype(np.float32)
    climate = rng.normal(size=(len(keys), 2)).astype(np.float32)
    drivers = rng.uniform(-1, 1, size=(len(keys), 4, 2)).astype(np.float32)
    y = np.array([-.7, .4, 1.2], dtype=np.float32)[:len(keys)]
    return n.LabeledBatch(x, climate, drivers, tuple(keys), y, (y > .7).astype(np.float32))


def clients_fixture():
    clients = []
    for number, count in enumerate((2, 3)):
        train_keys = ("2007-09-01", "2007-10-01", "2007-11-01")[-count:]
        clients.append(n.ClientInput(
            f"toy_client_{number}", labeled(train_keys, 10 + number),
            labeled(("2008-01-01", "2008-02-01", "2008-03-01"), 20 + number),
            labeled(("2011-01-01", "2011-02-01", "2011-03-01"), 30 + number),
            (.2 + number * .1, -.3), 100. + number * 20, 20.))
    return tuple(clients)


def scaling_fixture():
    keys = np.array(["2007-10", "2007-11", "2007-12", "2008-01", "2011-01", "2017-01"])
    data = {}
    for number, (first, ys) in enumerate(((1., (10., 30.)), (5., (20., 40.)))):
        latest = np.array([[first, 2 * first], [first + 2, 2 * (first + 2)],
                           [90, 180], [100, 200], [200, 400], [300, 600]], dtype=float)
        x = np.repeat(latest[:, None, :], 12, axis=1)
        data[f"toy_client_{number}"] = {"x": x, "climate": latest[:, 1:2],
              "drivers": np.zeros((6, 12, 2)), "y": np.array([*ys, 90, 100, 200, 300], dtype=float),
              "target_month": keys.copy(), "origin_month": (pd.PeriodIndex(keys, freq="M") - 1).astype(str).to_numpy()}
    schema = f.FeatureSet(("known", "climate"), ("climate",), {})
    return data, schema


class ScientificInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.clients = clients_fixture()
        cls.config = n.StrictConfig("fedavg", seed=42, rounds=2, local_epochs=1,
            batch_size=3, hidden=4, embedding=3, attention_hidden=4, layers=2,
            recurrent_dropout=.2, head_dropout=.4, lr=.01, optimizer="sgd",
            prox_mu=0., cpu_threads=1, state_checkpoint_interval=1)
        cls.avg = n.fit(cls.clients, cls.config)
        cls.prox_zero = n.fit(cls.clients, replace(cls.config, method="fedprox"))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_source_checksum_is_required(self):
        panel, _ = monthly_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy_monthly.csv"
            panel.to_csv(path, index=False)
            expected = f.sha256(path)
            self.assertEqual(len(f.checked_panel(path, expected, ["rainfall_mm", "weather_toy"])), 2640)
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                f.checked_panel(path, expected, ["rainfall_mm", "weather_toy"])

    def test_source_calendar_duplicates_and_missing_values_fail(self):
        panel, _ = monthly_fixture()
        cases = {"calendar": panel.iloc[1:].copy(),
                 "duplicate": pd.concat([panel, panel.iloc[:1]], ignore_index=True),
                 "nonfinite": panel.copy()}
        cases["nonfinite"].loc[0, "weather_toy"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            for name, bad in cases.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.csv"
                    bad.to_csv(path, index=False)
                    # A fresh authentic hash cannot make invalid contents admissible.
                    with self.assertRaises(ValueError):
                        f.checked_panel(path, f.sha256(path), ["rainfall_mm", "weather_toy"])

    def test_manual_availability_support_uses_available_months(self):
        panel, indices = monthly_fixture()
        panel["month"] = pd.PeriodIndex(panel["month"], freq="M")
        monthly = panel.set_index(["client", "month"])
        rows, schema = f.build_rows(monthly, indices, ("weather_toy",))
        output = f.samples(monthly, rows, schema, target_months=("2017-01",),
                           expected_evaluation_months=("2017-01",))
        client = f.CLIENTS[0]
        sample = output[client]
        last = dict(zip(schema.names, sample["x"][0, -1]))
        self.assertEqual(sample["origin_month"][0], "2016-12")
        # January target: November rainfall/index and September archived weather.
        self.assertEqual(last["rain_origin_lag1"], monthly.loc[(client, pd.Period("2016-11", freq="M")), "rainfall_mm"])
        self.assertEqual(last["weather_toy"], monthly.loc[(client, pd.Period("2016-09", freq="M")), "weather_toy"])
        self.assertEqual(last[f.INDEX_NAMES[0]], indices.set_index("month").loc["2016-11", f.INDEX_NAMES[0]])
        expected_mean = monthly.xs(client).loc[pd.period_range("2015-12", "2016-11", freq="M"), "rainfall_mm"].mean()
        self.assertAlmostEqual(last["rain_roll12_mean"], expected_mean)
        self.assertAlmostEqual(last["season_cos1"], np.cos(2 * np.pi / 12))
        # Mutating target and next-month observations cannot alter this forecast input.
        changed = monthly.copy()
        for date in ("2017-01", "2017-02"):
            changed.loc[(client, pd.Period(date, freq="M")), ["rainfall_mm", "weather_toy"]] = 999999.
        changed_rows, changed_schema = f.build_rows(changed, indices, ("weather_toy",))
        changed_output = f.samples(changed, changed_rows, changed_schema,
            target_months=("2017-01",), expected_evaluation_months=("2017-01",))
        np.testing.assert_array_equal(sample["x"], changed_output[client]["x"])

    def test_december_embargo_and_chronology_are_enforced(self):
        keys = np.array(["2007-11", "2007-12", "2008-01", "2010-11", "2010-12", "2011-01",
                         "2016-11", "2016-12", "2017-01"])
        roles = f.split_roles(keys)
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in roles.values()), [1, 0, 1, 1, 0, 1, 1, 0, 1])
        for role, last in (("train", "2007-12-01"), ("monitor", "2010-12-01"), ("validation", "2016-12-01")):
            with self.subTest(role=role):
                batch = getattr(self.clients[0], role)
                changed = replace(batch, keys=batch.keys[:-1] + (last,))
                clients = (replace(self.clients[0], **{role: changed}), self.clients[1])
                with self.assertRaises(ValueError):
                    n.fit(clients, self.config)
        with self.assertRaisesRegex(ValueError, "chronological"):
            batch = self.clients[0].train
            replace(batch, keys=tuple(reversed(batch.keys)))

    def test_future_observations_cannot_change_training_statistics(self):
        data, schema = scaling_fixture()
        transformed, stats = f.train_transform(data, schema)
        # Four gradient samples: first feature [1,3,5,7], second twice that.
        np.testing.assert_allclose(stats["global_input_mean"], [4., 8.], rtol=0, atol=1e-12)
        np.testing.assert_allclose(stats["global_input_std"], [np.sqrt(5), np.sqrt(20)], rtol=0, atol=1e-12)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["clients"]["toy_client_0"]["heavy80_threshold_mm"], 26.)
        changed = copy.deepcopy(data)
        for d in changed.values():
            future = ~f.split_roles(d["target_month"])["train"]
            d["x"][future] = 1e6
            d["climate"][future] = -1e6
            d["y"][future] = 1e6
        new_transformed, new_stats = f.train_transform(changed, schema)
        self.assertEqual(stats, new_stats)
        for client in data:
            train = f.split_roles(data[client]["target_month"])["train"]
            for name in ("x", "climate", "y_z", "heavy"):
                np.testing.assert_array_equal(transformed[client][name][train], new_transformed[client][name][train])

    def test_fedprox_zero_is_fedavg_with_real_parameter_updates(self):
        counts = np.array([len(c.train.keys) for c in self.clients], dtype=float)
        counts /= counts.sum()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            initial = n.clone_state(n.ALCE(3, 2, self.config))
        self.assertTrue(any(not torch.equal(initial[k], self.avg.round_states[-1]["global_state"][k]) for k in initial))
        for avg_round, prox_round in zip(self.avg.round_states, self.prox_zero.round_states):
            for key, tensor in avg_round["global_state"].items():
                with self.subTest(round=avg_round["round"], parameter=key):
                    torch.testing.assert_close(tensor, prox_round["global_state"][key], rtol=0, atol=0)
                    personal = avg_round["personal_states"]
                    expected = counts[0] * personal[self.clients[0].client_id][key] + counts[1] * personal[self.clients[1].client_id][key]
                    torch.testing.assert_close(tensor, expected, rtol=0, atol=1e-7)
        self.assertTrue(any(m["kind"] == "model_update" and m["payload_bytes"] > 8 for m in self.avg.messages))

    def test_inference_has_no_label_argument(self):
        labeled_eval = labeled(("2017-01-01", "2017-02-01"), 91)
        with self.assertRaisesRegex(TypeError, "label-free"):
            self.avg.predict(self.clients[0].client_id, labeled_eval, "global")
        batch = n.FeatureBatch(labeled_eval.x, labeled_eval.climate, labeled_eval.drivers, labeled_eval.keys)
        prediction = self.avg.predict(self.clients[0].client_id, batch, "global")
        self.assertEqual(prediction.keys, batch.keys)
        self.assertTrue(np.isfinite(prediction.rainfall_mm).all())
        self.assertFalse(hasattr(batch, "y_z"))
        with self.assertRaises(TypeError):
            n.FeatureBatch(batch.x, batch.climate, batch.drivers, batch.keys, y_z=np.zeros(2))
        for client in self.avg.data_manifest.values():
            self.assertTrue(all(split["last"] < "2017-01-01" for split in client.values()))

    def test_checkpoint_reload_and_dropout_evaluation_are_deterministic(self):
        source = labeled(("2017-01-01", "2017-02-01"), 92)
        batch = n.FeatureBatch(source.x, source.climate, source.drivers, source.keys)
        client = self.clients[0].client_id
        before_rng = torch.random.get_rng_state().clone()
        before = self.avg.predict(client, batch, "global")
        repeated = self.avg.predict(client, batch, "global")
        np.testing.assert_array_equal(before.z, repeated.z)
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy_checkpoint"
            self.avg.save(path)
            restored = n.load_fit(path)
            after = restored.predict(client, batch, "global")
            np.testing.assert_array_equal(before.rainfall_mm, after.rainfall_mm)
            np.testing.assert_array_equal(before.heavy_probability, after.heavy_probability)
            self.assertEqual(before.checkpoint_round, after.checkpoint_round)
            self.assertEqual(restored.selected_rounds, self.avg.selected_rounds)
            selected_only = n.load_fit(path, load_round_history=False)
            self.assertEqual(selected_only.round_states, [])
            self.assertGreater(len(restored.round_states), 0)
            for identity in ('global', 'personal'):
                full = restored.predict(client, batch, identity)
                small = selected_only.predict(client, batch, identity)
                np.testing.assert_array_equal(full.rainfall_mm, small.rainfall_mm)
                np.testing.assert_array_equal(full.heavy_probability, small.heavy_probability)
                self.assertEqual(full.checkpoint_round, small.checkpoint_round)

    def test_attention_changes_when_climate_context_is_intervened(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(173)
            model = n.ALCE(3, 2, self.config).eval()
            hidden = torch.randn(2, 4, 8)
        with torch.no_grad():
            calm = model.attention_weights(hidden, torch.zeros(2, 2))
            shifted = model.attention_weights(hidden, torch.tensor([[2., -2.], [-2., 2.]]))
        # A constant linear climate offset cancels under temporal softmax;
        # this intervention must change normalized weights, not just a head output.
        self.assertGreater(float((calm - shifted).abs().max()), 1e-6)
        torch.testing.assert_close(calm.sum(1), torch.ones(2), rtol=0, atol=1e-7)
        self.assertTrue(bool((calm > 0).all()))

    def test_disabled_embedding_makes_context_irrelevant(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(173)
            model = n.ALCE(3, 2, replace(self.config, use_embedding=False)).eval()
            hidden = torch.randn(2, 4, 8)
            x = torch.randn(2, 4, 3)
        contexts = (torch.zeros(2, 2), torch.tensor([[100., -100.], [-100., 100.]]))
        with torch.no_grad():
            weights = [model.attention_weights(hidden, value) for value in contexts]
            predictions = [model(x, value) for value in contexts]
        torch.testing.assert_close(weights[0], weights[1], rtol=0, atol=0)
        for first, second in zip(*predictions):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_anchor_removals_are_effective_and_invariant_to_removed_input(self):
        cfg = replace(self.config, method="caaa", alpha=.3, beta=.3, gamma=.4)
        args = (np.array([2., 6.]), np.array([0., 2.]), np.array([[1., 0.], [0., .5]]),
                np.array([[.9, .2], [.1, .8]]))
        base = n.caaa_weights(*args, cfg)
        no_performance = replace(cfg, use_performance=False)
        no_climate = replace(cfg, use_climate=False)
        perf_removed = n.caaa_weights(*args, no_performance)
        climate_removed = n.caaa_weights(*args, no_climate)
        self.assertFalse(np.allclose(base, perf_removed))
        self.assertFalse(np.allclose(base, climate_removed))
        np.testing.assert_array_equal(perf_removed, n.caaa_weights(args[0], np.array([90., .1]), args[2], args[3], no_performance))
        np.testing.assert_array_equal(climate_removed, n.caaa_weights(args[0], args[1], 9 * args[2], args[3], no_climate))
        self.assertTrue((base > 0).all())
        self.assertAlmostEqual(float(base.sum()), 1.)
        # A selected boundary candidate is admissible, but removing its sole
        # active term must fail. Fixed nonzero-anchor parents avoid this ambiguity.
        with self.assertRaisesRegex(ValueError, "every active"):
            n.caaa_weights(*args, replace(no_performance, alpha=0., beta=1., gamma=0.))


if __name__ == "__main__":
    unittest.main(verbosity=2)
