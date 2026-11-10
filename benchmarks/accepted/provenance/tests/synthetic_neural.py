"""Synthetic-only numerical checks; never load supplied rainfall inputs."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from neural_benchmark import (ALCE, ARCHITECTURE_ID, ClientInput, FeatureBatch, GraphInput,
                              LabeledBatch, StrictConfig, _PrivateClient, fit, load_fit, state_bytes)


def months(start, number):
    year, month = map(int, start.split("-")[:2])
    result = []
    for offset in range(number):
        total = year * 12 + month - 1 + offset
        result.append(f"{total//12:04d}-{total%12+1:02d}-01")
    return tuple(result)


def synthetic_clients():
    rng = np.random.default_rng(98234)
    clients = []
    for cid in range(3):
        parts = []
        for first, number in (("2000-01", 32), ("2008-01", 12), ("2011-01", 12)):
            x = rng.normal(size=(number, 4, 5)).astype("float32")
            climate = rng.normal(size=(number, 3)).astype("float32")
            drivers = rng.normal(loc=(cid*.8, -cid*.3), scale=1.1, size=(number, 4, 2)).astype("float32")
            y = (.5*x[:, -1, 0] - .3*x[:, -2, 1] + .2*climate[:, 0] + cid*.3).astype("float32")
            parts.append(LabeledBatch(x, climate, drivers, months(first, number), y, (y>.6).astype("float32")))
        clients.append(ClientInput(f"SyntheticClient{cid+1}", *parts,
                                   sensitivity=(.2+.3*cid, -.1-.1*cid), target_mean=100+20*cid, target_std=20))
    return clients


def state_difference(a, b):
    return max(float((a[k]-b[k]).abs().max()) for k in a)


def main():
    clients = synthetic_clients()
    cfg = StrictConfig("fedavg", rounds=2, local_epochs=1, batch_size=8,
                       optimizer="sgd", lr=.03, hidden=4, embedding=3, layers=1,
                       head_dropout=.2, recurrent_dropout=0, cpu_threads=1, state_checkpoint_interval=1)
    report = {"label": "SYNTHETIC VERIFICATION ONLY — NOT RAINFALL RESULTS", "architecture": ARCHITECTURE_ID, "checks": {}}
    fa = fit(clients, cfg)
    fp0 = fit(clients, replace(cfg, method="fedprox", prox_mu=0))
    delta = state_difference(fa.global_state, fp0.global_state)
    assert delta == 0, delta
    report["checks"]["fedprox_mu0_bitwise_fedavg"] = delta
    assert fa.selected_rounds == fp0.selected_rounds
    assert cfg.primary == "global" and replace(cfg, method="fedprox").primary == "global"
    assert replace(cfg, primary_identity="personal").primary == "personal"
    assert replace(cfg, method="fedprox", primary_identity="personal").primary == "personal"
    for cid in fa.client_ids:
        assert state_difference(fa.personal_states[cid], fp0.personal_states[cid]) == 0
        chosen = next(r for r in fa.round_states if r["round"] == fa.selected_rounds["personal"])
        assert state_difference(fa.personal_states[cid], chosen["personal_states"][cid]) == 0
    assert any(state_difference(state, fa.global_state) > 1e-9 for state in fa.personal_states.values())
    assert set(fa.selected_rounds) == {"global", "personal"}
    report["checks"]["fedavg_fedprox_personal_post_update_diagnostic_mu0_identity"] = True
    personal_curves = [r for r in fa.curves if r["kind"] == "evaluation" and r["identity"] == "personal"]
    assert fa.selected_rounds["personal"] == max(personal_curves, key=lambda r: (r["validation_macro_r2"], -r["round"]))["round"]
    report["checks"]["personal_checkpoint_macro_validation_rule"] = True
    sample = clients[0].validation
    features = FeatureBatch(sample.x, sample.climate, sample.drivers, sample.keys)
    a = fa.predict(clients[0].client_id, features)
    b = fa.predict(clients[0].client_id, features)
    assert np.array_equal(a.z, b.z)
    report["checks"]["repeat_inference_bitwise_equal"] = True
    try:
        fa.predict(clients[0].client_id, sample)
    except TypeError:
        report["checks"]["inference_rejects_labels"] = True
    else:
        raise AssertionError("Label-bearing inference was accepted")
    try:
        StrictConfig("fedavg", ignored_component_flag=False)
    except TypeError:
        report["checks"]["unknown_config_rejected"] = True
    else:
        raise AssertionError("Unused config flag was accepted")
    post_validation = LabeledBatch(sample.x, sample.climate, sample.drivers,
                                  months("2017-01", len(sample.keys)), sample.y_z, sample.heavy)
    try:
        fit([replace(clients[0], validation=post_validation), *clients[1:]], cfg)
    except ValueError:
        report["checks"]["post2016_labels_rejected"] = True
    else:
        raise AssertionError("Evaluation-period labels reached fit")
    embargoed = []
    for role, last_key in (("train", "2007-12-01"), ("monitor", "2010-12-01"),
                           ("validation", "2016-12-01")):
        original = getattr(clients[0], role)
        late_batch = replace(original, keys=(*original.keys[:-1], last_key))
        try:
            fit([replace(clients[0], **{role: late_batch}), *clients[1:]], cfg)
        except ValueError:
            embargoed.append(role)
        else:
            raise AssertionError(f"Unreleased role-end December {role} label reached fit")
    report["checks"]["unreleased_role_end_december_labels_rejected"] = embargoed
    total_examples = sum(row["examples"] for row in fa.local_steps)
    assert total_examples == cfg.rounds*cfg.local_epochs*sum(len(c.train.keys) for c in clients)
    report["checks"]["example_ledger_exact"] = total_examples
    model_size = state_bytes(fa.global_state)
    expected_down = len(clients)*model_size*(1+cfg.rounds)
    expected_up = len(clients)*(model_size*cfg.rounds+8+64*cfg.rounds)
    actual_down = sum(m["payload_bytes"] for m in fa.messages if m["direction"] == "downlink")
    actual_up = sum(m["payload_bytes"] for m in fa.messages if m["direction"] == "uplink")
    assert (actual_down, actual_up) == (expected_down, expected_up)
    report["checks"]["payload_byte_ledger_exact"] = {"model_state_bytes": model_size, "downlink": actual_down, "uplink": actual_up}
    assert sum(m["kind"] == "model_update" for m in fa.messages) == len(clients) * cfg.rounds
    assert sum(m["kind"] == "global_reference" for m in fa.messages) == len(clients) * cfg.rounds
    report["checks"]["personal_diagnostics_add_no_training_or_model_message_stream"] = True

    ca_cfg = replace(cfg, method="caaa", prox_mu=.25)
    full = fit(clients, ca_cfg)
    variants = dict(no_climate=dict(use_climate=False), no_performance=dict(use_performance=False),
                    no_prox=dict(use_prox=False), no_aux=dict(use_aux=False), no_embedding=dict(use_embedding=False))
    effects = {}
    for name, flags in variants.items():
        variant = fit(clients, replace(ca_cfg, **flags))
        effects[name] = state_difference(full.global_state, variant.global_state)
        assert effects[name] > 1e-9, (name, effects[name])
        if name == "no_prox":
            assert all(row["lambda_max"] == 0 and row["prox_mean"] == 0 for row in variant.local_steps)
        if name == "no_aux":
            assert all(row["bce_mean"] == 0 for row in variant.local_steps)
        if name == "no_performance":
            assert not any(row["kind"] == "inner_monitor_loss" for row in variant.messages)
        if name == "no_climate":
            assert not any(row["kind"] == "round_climate_strength" for row in variant.messages)
    report["checks"]["all_component_ablations_change_parameters"] = effects
    assert all(.3*ca_cfg.prox_mu-1e-7 <= s["lambda_min"] <= s["lambda_max"] <= ca_cfg.prox_mu+1e-7 for s in full.local_steps)
    report["checks"]["adaptive_prox_bounds"] = True
    assert all(abs(sum(r["weights"].values())-1) < 1e-12 and min(r["weights"].values()) >= 0
               for r in full.curves if r["kind"] == "aggregation")
    report["checks"]["caaa_convex_weights"] = True

    changed_validation = replace(clients[0], validation=LabeledBatch(sample.x, sample.climate, sample.drivers,
                                                                   sample.keys, -sample.y_z+2, sample.heavy))
    altered = fit([changed_validation, *clients[1:]], ca_cfg)
    assert all(state_difference(a["global_state"], b["global_state"]) == 0
               for a, b in zip(full.round_states, altered.round_states))
    report["checks"]["outer_labels_only_select_checkpoint_not_training_path"] = True
    monitor = clients[0].monitor
    changed_monitor = replace(clients[0], monitor=LabeledBatch(monitor.x, monitor.climate, monitor.drivers,
                                                             monitor.keys, monitor.y_z+3, monitor.heavy))
    altered = fit([changed_monitor, *clients[1:]], ca_cfg)
    assert state_difference(full.round_states[-1]["global_state"], altered.round_states[-1]["global_state"]) > 1e-9
    no_perf = fit(clients, replace(ca_cfg, use_performance=False))
    no_perf_altered = fit([changed_monitor, *clients[1:]], replace(ca_cfg, use_performance=False))
    assert state_difference(no_perf.round_states[-1]["global_state"], no_perf_altered.round_states[-1]["global_state"]) == 0
    report["checks"]["inner_monitor_feedback_and_removal_verified"] = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = ALCE(5, 3, replace(ca_cfg, use_embedding=False))
    model.eval()
    with torch.no_grad():
        p1 = model(torch.tensor(features.x), torch.tensor(features.climate))[0]
        p2 = model(torch.tensor(features.x), torch.tensor(features.climate+10))[0]
    assert torch.equal(p1, p2)
    report["checks"]["disabled_embedding_ignores_context"] = True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(117)
        climate_model = ALCE(5, 3, ca_cfg).eval()
        fixed_hidden = torch.randn(6, 4, 2 * ca_cfg.hidden)
    held_context = torch.tensor(features.climate[:6])
    intervened_context = held_context + torch.tensor([2.0, -3.0, 1.5])
    with torch.no_grad():
        att_original = climate_model.attention_weights(fixed_hidden, held_context)
        att_changed = climate_model.attention_weights(fixed_hidden, intervened_context)
        zero_original = model.attention_weights(fixed_hidden, held_context)
        zero_changed = model.attention_weights(fixed_hidden, intervened_context)
    attention_delta = float((att_original - att_changed).abs().max())
    assert attention_delta > 1e-7, attention_delta
    assert torch.equal(zero_original, zero_changed)
    assert torch.allclose(att_original.sum(1), torch.ones(6), atol=1e-7)
    report["checks"]["nonlinear_joint_attention_climate_intervention_fixed_hidden_max_delta"] = attention_delta
    report["checks"]["zero_embedding_removes_climate_attention_pathway"] = True
    # Force an artificial negative regression output; RAW mm inference must preserve it.
    raw_state = {k: v.clone() for k, v in fa.global_state.items()}
    raw_state["regression.5.weight"].zero_()
    raw_state["regression.5.bias"].fill_(-10.0)
    raw_result = replace(fa, global_state=raw_state)
    raw_prediction = raw_result.predict(clients[0].client_id, features, identity="global")
    assert np.all(raw_prediction.rainfall_mm < 0)
    report["checks"]["raw_mm_inference_is_unclipped"] = True
    raw_evaluator = _PrivateClient(clients[0], ALCE(5, 3, cfg))
    raw_score = raw_evaluator.evaluate(raw_state, "validation")["r2"]
    truth = clients[0].target_mean + clients[0].target_std * sample.y_z.astype("float64")
    forced_mm = clients[0].target_mean - 10 * clients[0].target_std
    expected_raw_score = 1 - np.sum((truth - forced_mm)**2) / np.sum((truth-truth.mean())**2)
    assert abs(raw_score - expected_raw_score) < 1e-12
    report["checks"]["raw_mm_checkpoint_score_is_unclipped"] = True
    core_clients = []
    for client in clients:
        roles = {name: replace(getattr(client, name), climate=np.empty((len(getattr(client, name).keys), 0)))
                 for name in ("train", "monitor", "validation")}
        core_clients.append(replace(client, **roles))
    core = fit(core_clients, replace(cfg, use_embedding=False, rounds=1))
    core_val = core_clients[0].validation
    core_features = FeatureBatch(core_val.x, core_val.climate, core_val.drivers, core_val.keys)
    assert np.isfinite(core.predict(clients[0].client_id, core_features).rainfall_mm).all()
    try:
        fit(core_clients, replace(cfg, rounds=1))
    except ValueError:
        report["checks"]["zero_context_core_supported_only_with_disabled_embedding"] = True
    else:
        raise AssertionError("A zero-column active climate embedding was accepted")

    ids = tuple(c.client_id for c in clients)
    graphs = {
        "uniform": GraphInput(ids, np.ones((3, 3))),
        "original": GraphInput(ids, [[1,.95,.8],[.95,1,.85],[.8,.85,1]]),
        "enriched": GraphInput(ids, [[1,.2,.7],[.2,1,.3],[.7,.3,1]])}
    methods = {}
    for method in ("central", "local", "fedprox", "ditto"):
        result = fit(clients, replace(cfg, method=method))
        assert np.isfinite(result.predict(ids[0], features).rainfall_mm).all()
        methods[method] = {"selected_rounds": result.selected_rounds,
                           "optimizer_steps": sum(r["optimizer_steps"] for r in result.local_steps)}
    graph_results = {}
    for name in graphs:
        result = fit(clients, replace(cfg, method="climate_graph", graph_name=name, prox_mu=.25), graphs)
        assert result.global_state is None and set(result.personal_states) == set(ids)
        assert np.isfinite(result.predict(ids[0], features).rainfall_mm).all()
        graph_results[name] = result
    assert state_difference(graph_results["uniform"].personal_states[ids[0]],
                            graph_results["enriched"].personal_states[ids[0]]) > 1e-9
    report["checks"]["all_methods_and_three_graphs_smoke"] = methods
    # Export to a unique directory; a rerun cannot silently reuse stale artifacts.
    root = Path(__file__).parent/"synthetic_smoke"
    root.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="synthetic_", dir=root))
    full.save(run/"fit")
    restored = load_fit(run/"fit")
    assert np.array_equal(full.predict(ids[0], features).z, restored.predict(ids[0], features).z)
    report["checks"]["saved_state_roundtrip_bitwise_predictions"] = True
    report["artifact_directory"] = str(run)
    report["module_sha256"] = hashlib.sha256((Path(__file__).parent/"neural_benchmark.py").read_bytes()).hexdigest()
    (run/"verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    (Path(__file__).parent/"verification_receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
