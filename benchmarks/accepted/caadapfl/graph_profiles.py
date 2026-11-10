"""Training-only climate graph with an explicit privileged preprocessing ledger.

This implements the manuscript's original and enriched graph concepts with
declared revisions where the supplied equations are incomplete. It does not
reproduce the historical table, infer release dates, or implement secure
aggregation. In particular, pairwise rainfall covariance requires privileged
offline access to jointly aligned training anomalies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:  # standalone scratch module or copied into a package
    from .neural_benchmark import GraphInput
except ImportError:
    from neural_benchmark import GraphInput


INDEX_NAMES = (
    "nino34_anomaly",
    "dmi_hadisst11_anomaly",
    "soi_cpc_standardized",
    "oni_anomaly_3mo",
    "meiv2_bimonthly",
    "amv_ersstv6_global_removed_1951_1980",
)
RAIN_STATS_NAMES = ("variance_mm2", "coefficient_of_variation", "jjas_rainfall_share")


def _month(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Target/support months must be strings")
    if len(value) == 7:
        value += "-01"
    if len(value) != 10:
        raise ValueError("Use monthly YYYY-MM or YYYY-MM-01 keys")
    parsed = date.fromisoformat(value)
    if parsed.day != 1:
        raise ValueError("A key must identify the first day of a calendar month")
    return parsed.isoformat()[:7]


def _month_number(value: str) -> int:
    return int(value[:4]) * 12 + int(value[5:7]) - 1


def _vector(value, n: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != (n,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {n}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ClientGraphData:
    target_month: tuple[str, ...]
    y: np.ndarray                  # raw rainfall mm, target-aligned
    indices: Mapping[str, np.ndarray]  # six eligible raw index vectors, target-aligned
    index_support_month: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self):
        keys = tuple(_month(k) for k in self.target_month)
        if not keys or keys != tuple(sorted(set(keys))):
            raise ValueError("Training target months must be nonempty, unique and chronological")
        object.__setattr__(self, "target_month", keys)
        object.__setattr__(self, "y", _vector(self.y, len(keys), "rainfall target"))
        if (self.y < 0).any():
            raise ValueError("Raw rainfall targets cannot be negative")
        if set(self.indices) != set(INDEX_NAMES):
            raise ValueError(f"Indices must have exactly these six names: {INDEX_NAMES}")
        vectors = {name: _vector(self.indices[name], len(keys), name) for name in INDEX_NAMES}
        object.__setattr__(self, "indices", vectors)
        if self.index_support_month is not None:
            if set(self.index_support_month) != set(INDEX_NAMES):
                raise ValueError("Support month schema must match the six indices")
            support = {name: tuple(_month(k) for k in self.index_support_month[name]) for name in INDEX_NAMES}
            for name, values in support.items():
                if len(values) != len(keys):
                    raise ValueError(f"{name}: support and target lengths differ")
                if any(_month_number(s) > _month_number(t) - 2 for s, t in zip(values, keys)):
                    raise ValueError(f"{name}: index support violates target-minus-two conservative lag")
            object.__setattr__(self, "index_support_month", support)


@dataclass
class GraphBuild:
    client_ids: tuple[str, ...]
    graphs: dict[str, GraphInput]
    components: dict[str, np.ndarray]
    profiles: dict[str, dict]
    metadata: dict
    messages: list[dict]

    @property
    def sensitivities(self) -> dict[str, tuple[float, float]]:
        return {cid: tuple(self.profiles[cid]["original_sensitivity_enso_iod"]) for cid in self.client_ids}

    def to_dict(self) -> dict:
        return {
            "client_ids": list(self.client_ids),
            "graphs": {key: value.matrix.tolist() for key, value in self.graphs.items()},
            "components": {key: value.tolist() for key, value in self.components.items()},
            "profiles": self.profiles,
            "metadata": self.metadata,
            "messages": self.messages,
        }

    def save(self, path: str | Path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8")


def _corr(a: np.ndarray, b: np.ndarray, *, label: str, flags: list[dict], min_pairs: int) -> float:
    if len(a) < min_pairs:
        raise ValueError(f"{label}: {len(a)} pairs below min_pairs={min_pairs}")
    da, db = a - a.mean(), b - b.mean()
    norm_a, norm_b = float(np.linalg.norm(da)), float(np.linalg.norm(db))
    # A relative floor catches exactly constant and machine-precision near-constant series.
    scale_a = max(float(np.linalg.norm(a)), 1.0)
    scale_b = max(float(np.linalg.norm(b)), 1.0)
    if norm_a <= 1e-12 * scale_a or norm_b <= 1e-12 * scale_b:
        flags.append({"operation": label, "reason": "constant_or_near_constant_series", "policy": "correlation_zero"})
        return 0.0
    return float(np.clip(np.dot(da / norm_a, db / norm_b), -1.0, 1.0))


def _shifted_cosine(a: np.ndarray, b: np.ndarray, *, label: str, flags: list[dict]) -> float:
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        flags.append({"operation": label, "reason": "zero_norm_descriptor", "policy": "cosine_zero_then_shift_to_half"})
        return 0.5
    return 0.5 * (1.0 + float(np.clip(np.dot(a / norm_a, b / norm_b), -1.0, 1.0)))


def _standardize(values: np.ndarray, *, label: str, flags: list[dict], axis: int = 0) -> np.ndarray:
    mean = values.mean(axis=axis, keepdims=True)
    std = values.std(axis=axis, keepdims=True)
    constant = std <= 1e-12 * np.maximum(np.abs(mean), 1.0)
    if constant.any():
        flags.append({"operation": label, "reason": "constant_descriptor", "count": int(constant.sum()),
                      "policy": "standardized_value_zero"})
    return np.where(constant, 0.0, (values - mean) / np.where(constant, 1.0, std))


def build_graphs(
    clients: Mapping[str, ClientGraphData | Mapping],
    *,
    client_order: Sequence[str] | None = None,
    cutoff: str = "2007-11",
    allow_training_envelope: bool = False,
    min_pairs: int = 24,
    min_calendar_samples: int = 2,
) -> GraphBuild:
    """Fit original/enriched/uniform graphs from explicitly supplied fit-role arrays.

    Default data must end in December 2007 or earlier. A separately labelled
    train-envelope diagnostic can use dates through December 2010 only when
    ``allow_training_envelope=True`` AND a later explicit ``cutoff`` are supplied.
    No monitor, validation or retrospective evaluation arrays are accepted.
    The caller must pre-align raw indices to eligible origin-time support. Optional
    support-month vectors enforce the conservative target-minus-two month rule;
    they do not establish historical publication availability by themselves.
    """
    if type(allow_training_envelope) is not bool:
        raise ValueError("allow_training_envelope must be boolean")
    if not isinstance(min_pairs, int) or isinstance(min_pairs, bool) or min_pairs < 2:
        raise ValueError("min_pairs must be an integer >=2")
    if not isinstance(min_calendar_samples, int) or isinstance(min_calendar_samples, bool) or min_calendar_samples < 1:
        raise ValueError("min_calendar_samples must be a positive integer")
    cutoff = _month(cutoff)
    if cutoff > "2007-11" and not allow_training_envelope:
        raise ValueError("Default graph fitting is gradient_train only, ending no later than 2007-11")
    if cutoff > "2010-11":
        raise ValueError("Even the explicitly labelled training envelope must end by 2010-11")
    order = tuple(clients) if client_order is None else tuple(client_order)
    if not order or len(set(order)) != len(order) or set(order) != set(clients):
        raise ValueError("client_order must contain each supplied client exactly once")
    if any(not isinstance(cid, str) or not cid for cid in order):
        raise ValueError("Client IDs must be nonempty strings")
    data: dict[str, ClientGraphData] = {}
    flags: list[dict] = []
    profiles: dict[str, dict] = {}
    anomalies: dict[str, np.ndarray] = {}
    for cid in order:
        value = clients[cid]
        if not isinstance(value, ClientGraphData):
            value = ClientGraphData(**value)  # strict: unknown or missing keys fail
        if value.target_month[-1] > cutoff:
            raise ValueError(f"{cid}: target {value.target_month[-1]} exceeds declared fitting cutoff {cutoff}")
        if len(value.y) < min_pairs:
            raise ValueError(f"{cid}: too few gradient-training examples")
        data[cid] = value
        calendar = np.array([int(key[5:7]) for key in value.target_month])
        means, month_counts = [], []
        for month in range(1, 13):
            month_y = value.y[calendar == month]
            if len(month_y) < min_calendar_samples:
                raise ValueError(f"{cid}: month {month} has fewer than {min_calendar_samples} fit observations")
            means.append(float(month_y.mean()))
            month_counts.append(len(month_y))
        seasonal = np.array(means)
        anomaly = value.y - seasonal[calendar - 1]
        anomalies[cid] = anomaly
        correlations = [_corr(anomaly, value.indices[name], label=f"{cid}:sensitivity:{name}",
                              flags=flags, min_pairs=min_pairs) for name in INDEX_NAMES]
        mean_y = float(value.y.mean())
        variance = float(value.y.var())
        if mean_y <= 1e-12:
            cv, concentration = 0.0, 0.0
            flags.append({"operation": f"{cid}:rain_stats", "reason": "zero_mean_rainfall",
                          "policy": "cv_and_monsoon_share_zero"})
        else:
            cv = float(value.y.std() / mean_y)
            # Equal calendar weighting avoids bias from uneven sample counts.
            concentration = float(seasonal[5:9].sum() / seasonal.sum())
        season_z = _standardize(seasonal, label=f"{cid}:seasonal_climatology", flags=flags)
        profiles[cid] = {
            "target_start": value.target_month[0], "target_end": value.target_month[-1], "n": len(value.y),
            "calendar_counts": month_counts,
            "seasonal_climatology_mm": means,
            "standardized_seasonal_climatology": season_z.tolist(),
            "index_sensitivity_order": list(INDEX_NAMES),
            "multi_index_sensitivity": correlations,
            "original_sensitivity_enso_iod": correlations[:2],
            "rain_stats_order": list(RAIN_STATS_NAMES),
            "rain_stats_raw": [variance, cv, concentration],
            "index_support_month_supplied": value.index_support_month is not None,
        }
    stats = np.array([profiles[cid]["rain_stats_raw"] for cid in order])
    if not np.isfinite(stats).all():
        raise ValueError("Training rainfall descriptors overflowed; check raw units/values")
    stats_z = _standardize(stats, label="cross_client_rain_statistics", flags=flags)
    for row, cid in enumerate(order):
        profiles[cid]["rain_stats_standardized_across_clients"] = stats_z[row].tolist()
    k = len(order)
    components = {name: np.eye(k, dtype=np.float64) for name in (
        "rain_anomaly_correlation_shifted", "seasonal_cosine_shifted", "sensitivity_cosine_shifted", "rain_stats_exp_squared_distance")}
    original = np.eye(k, dtype=np.float64)
    pair_counts: list[dict] = []
    for i, cid_i in enumerate(order):
        for j in range(i):
            cid_j = order[j]
            pi, pj = profiles[cid_i], profiles[cid_j]
            original[i, j] = original[j, i] = _shifted_cosine(
                np.array(pi["original_sensitivity_enso_iod"]), np.array(pj["original_sensitivity_enso_iod"]),
                label=f"original:{cid_i}:{cid_j}", flags=flags)
            keys_i = {key: at for at, key in enumerate(data[cid_i].target_month)}
            keys_j = {key: at for at, key in enumerate(data[cid_j].target_month)}
            common = sorted(set(keys_i) & set(keys_j))
            pair_counts.append({"client_i": cid_i, "client_j": cid_j, "aligned_target_count": len(common)})
            rainfall_corr = _corr(anomalies[cid_i][[keys_i[key] for key in common]],
                                  anomalies[cid_j][[keys_j[key] for key in common]],
                                  label=f"rain_covariance:{cid_i}:{cid_j}", flags=flags, min_pairs=min_pairs)
            values = (
                0.5 * (rainfall_corr + 1.0),
                _shifted_cosine(np.array(pi["standardized_seasonal_climatology"]),
                                np.array(pj["standardized_seasonal_climatology"]),
                                label=f"seasonal:{cid_i}:{cid_j}", flags=flags),
                _shifted_cosine(np.array(pi["multi_index_sensitivity"]), np.array(pj["multi_index_sensitivity"]),
                                label=f"sensitivity:{cid_i}:{cid_j}", flags=flags),
                float(np.exp(-np.mean((stats_z[i] - stats_z[j]) ** 2))),
            )
            for name, scalar in zip(components, values):
                components[name][i, j] = components[name][j, i] = scalar
    enriched = sum(components.values()) / 4.0
    graphs = {"original": GraphInput(order, original), "enriched": GraphInput(order, enriched),
              "uniform": GraphInput(order, np.ones((k, k), dtype=np.float64))}
    # Counts describe a concrete cleartext setup protocol. They are separated from
    # model-training traffic and do not imply these are secure/private summaries.
    # Twelve climatology + six sensitivities + three stats doubles, one int64 count.
    profile_bytes = (12 + 6 + 3) * 8 + 8
    messages: list[dict] = []
    for cid in order:
        n = len(data[cid].y)
        messages.extend([
            {"phase": "privileged_offline_graph_setup", "sender": cid, "receiver": "graph_preprocessor",
             "payload": "cleartext_profile_12_climatology_6_sensitivity_3_stats_plus_count",
             "bytes": profile_bytes, "privacy": "shared_training_summaries; no secure aggregation"},
            {"phase": "privileged_offline_graph_setup", "sender": cid, "receiver": "graph_preprocessor",
             "payload": "cleartext_deseasonalized_rainfall_float64_and_month_number_int64",
             "bytes": n * 16, "privacy": "privileged aligned training series required for pairwise covariance"},
        ])
    for name in graphs:
        messages.append({"phase": "setup_artifact", "sender": "graph_preprocessor", "receiver": "training_server",
                         "payload": f"{name}_graph_matrix_float32", "bytes": k * k * 4,
                         "privacy": "released graph; input series are not exposed to the training server API"})
    role = "training_envelope_explicit_diagnostic" if allow_training_envelope else "gradient_train"
    metadata = {
        "fit_role": role, "fit_cutoff": cutoff, "client_order": list(order),
        "index_names": list(INDEX_NAMES), "min_pairs": min_pairs, "min_calendar_samples": min_calendar_samples,
        "source": "manuscript P0258-P0260 / Eq3 graph concepts; explicit revised implementation",
        "definitions": {
            "rainfall_anomaly": "raw target rainfall minus own fit-only calendar-month mean",
            "sensitivity": "Pearson correlation of fit rainfall target anomaly with lag-eligible raw index",
            "original": "shifted cosine of ENSO/IOD sensitivity; diagonal one",
            "enriched": "equal 0.25 mean of four reported component matrices; diagonal one",
            "seasonal_revision": "within-client standardized 12-month climatology; cosine shifted into [0,1]",
            "rain_stats_revision": "population rainfall variance, population std/mean, JJAS/annual calendar climatology share",
            "distance_revision": "rain-stat columns standardized across fit clients; d² is mean squared descriptor difference; exp(-d²)",
            "uniform": "all entries one; row normalization is performed by the neural graph trainer",
        },
        "source_availability": "Caller supplies eligible raw vectors; optional support months check target-minus-two only. Historical release latency/revisions are not proven.",
        "degeneracy_policy": "constant correlations ->0; zero-norm cosine ->0 then shifted .5; constant standardized descriptors ->0; diagonal self-similarity1",
        "degeneracies": flags, "pairwise_alignment_counts": pair_counts,
        "privacy": "Privileged offline graph preprocessor has aligned client training rainfall. Shared profiles/covariance and model updates are not secure aggregation, differential privacy, or a privacy guarantee.",
        "ledger": "Concrete cleartext setup payload sizes only; excludes serialization, encryption, transport headers, public source ingestion and feature assembly. Logged separately from training messages.",
    }
    return GraphBuild(order, graphs, components, profiles, metadata, messages)
