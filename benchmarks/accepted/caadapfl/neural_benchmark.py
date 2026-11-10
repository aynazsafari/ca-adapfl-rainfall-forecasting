"""Auditable ALCE parameter federation. Historical rainfall results are not imported.

This is a one-process algorithm simulation, not secure aggregation or differential
privacy. Fit has no evaluation/test-label argument. Each private client returns
only model states, scalar losses/scores, counts and declared climate summaries.
The centralized comparator deliberately pools its training tensors.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

METHODS = {"central", "local", "fedavg", "fedprox", "ditto", "climate_graph", "caaa"}
PAIRED_SEEDS = (42, 123, 2024, 7, 999)
ARCHITECTURE_ID = "alce_joint_tanh_attention_v2"


def _float_array(value, dimensions: int, name: str) -> np.ndarray:
    value = np.array(value, dtype=np.float32, copy=True)
    if value.ndim != dimensions or not np.isfinite(value).all():
        raise ValueError(f"{name}: require a finite {dimensions}-dimensional array")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class FeatureBatch:
    x: np.ndarray                 # [N, historical window, features]
    climate: np.ndarray           # [N, origin climate context]
    drivers: np.ndarray           # [N, historical window, ENSO/IOD anomaly]
    keys: tuple[str, ...]          # target month ISO YYYY-MM-01, strictly increasing

    def __post_init__(self):
        for name, ndim in (("x", 3), ("climate", 2), ("drivers", 3)):
            object.__setattr__(self, name, _float_array(getattr(self, name), ndim, name))
        object.__setattr__(self, "keys", tuple(self.keys))
        n, window, features = self.x.shape
        if n < 1 or window < 1 or features < 1:
            raise ValueError("Empty sequences/features are not a benchmark sample")
        if self.climate.shape[0] != n or self.drivers.shape != (n, window, 2) or len(self.keys) != n:
            raise ValueError("Feature arrays and keys must have identical sample axes")
        dates = []
        for key in self.keys:
            if not isinstance(key, str) or len(key) != 10:
                raise ValueError("keys must be target-month ISO strings YYYY-MM-01")
            parsed = date.fromisoformat(key)
            if parsed.day != 1:
                raise ValueError("Keys must identify monthly targets")
            dates.append(parsed)
        if dates != sorted(set(dates)):
            raise ValueError("Target keys must be unique and strictly chronological")


@dataclass(frozen=True)
class LabeledBatch(FeatureBatch):
    y_z: np.ndarray
    heavy: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "y_z", _float_array(self.y_z, 1, "y_z"))
        object.__setattr__(self, "heavy", _float_array(self.heavy, 1, "heavy"))
        if self.y_z.shape != (len(self.keys),) or self.heavy.shape != self.y_z.shape:
            raise ValueError("Labels must have the sample axis of their feature batch")
        if not np.isin(self.heavy, (0, 1)).all():
            raise ValueError("heavy labels must be training-threshold binary events")


@dataclass(frozen=True)
class ClientInput:
    client_id: str
    train: LabeledBatch
    monitor: LabeledBatch
    validation: LabeledBatch
    sensitivity: tuple[float, float]  # train-derived ENSO/IOD correlation
    target_mean: float
    target_std: float

    def __post_init__(self):
        if not self.client_id or not isinstance(self.client_id, str):
            raise ValueError("client_id is required")
        if not np.isfinite((self.target_mean, self.target_std)).all() or self.target_std <= 0:
            raise ValueError("Target inverse transform must be finite with positive std")
        sensitivity = tuple(float(v) for v in self.sensitivity)
        if len(sensitivity) != 2 or not np.isfinite(sensitivity).all() or max(map(abs, sensitivity)) > 1:
            raise ValueError("Sensitivity must contain two finite correlations in [-1,1]")
        object.__setattr__(self, "sensitivity", sensitivity)
        shapes = {(s.x.shape[1:], s.climate.shape[1]) for s in (self.train, self.monitor, self.validation)}
        if len(shapes) != 1:
            raise ValueError("All client split feature schemas must match")
        if not self.train.keys[-1] < self.monitor.keys[0] or not self.monitor.keys[-1] < self.validation.keys[0]:
            raise ValueError("Gradient train, inner monitor and outer validation must be disjoint in chronological order")


@dataclass(frozen=True)
class GraphInput:
    client_ids: tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "client_ids", tuple(self.client_ids))
        object.__setattr__(self, "matrix", _float_array(self.matrix, 2, "graph"))
        k = len(self.client_ids)
        if len(set(self.client_ids)) != k or self.matrix.shape != (k, k):
            raise ValueError("Graph needs unique client IDs and a K by K matrix")
        if (self.matrix < 0).any() or (self.matrix.sum(axis=1) <= 0).any():
            raise ValueError("Graph weights must be nonnegative with positive row sums")


@dataclass(frozen=True)
class StrictConfig:
    method: str
    seed: int = 42
    rounds: int = 60
    local_epochs: int = 3
    batch_size: int = 32
    lr: float = 0.003
    optimizer: str = "adam"       # disclosed local-solver adaptation; SGD is supported
    momentum: float = 0.0
    prox_mu: float = 0.01
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.4
    use_climate: bool = True
    use_performance: bool = True
    use_prox: bool = True
    use_aux: bool = True
    use_embedding: bool = True
    adaptive_prox: bool = True
    aux_weight: float = 0.3
    graph_name: str | None = None
    hidden: int = 48
    embedding: int = 16
    attention_hidden: int = 32
    layers: int = 2
    recurrent_dropout: float = 0.2
    head_dropout: float = 0.3
    grad_clip: float = 5.0
    cpu_threads: int = 1
    state_checkpoint_interval: int = 10
    train_end: str = "2007-11-01"
    monitor_start: str = "2008-01-01"
    monitor_end: str = "2010-11-01"
    validation_start: str = "2011-01-01"
    validation_end: str = "2016-11-01"
    primary_identity: str = "auto"

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected {sorted(METHODS)}")
        for key in ("rounds", "local_epochs", "batch_size", "hidden", "embedding", "attention_hidden", "layers", "cpu_threads", "state_checkpoint_interval"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0,2**31)")
        for key in ("lr", "prox_mu", "alpha", "beta", "gamma", "aux_weight", "grad_clip", "momentum"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.lr <= 0 or self.grad_clip <= 0 or self.alpha + self.beta + self.gamma <= 0:
            raise ValueError("Learning rate/clip must be positive; CAAA coefficients cannot all be zero")
        if self.optimizer not in {"adam", "sgd"} or self.momentum >= 1:
            raise ValueError("optimizer must be adam or sgd; momentum must be below one")
        for key in ("recurrent_dropout", "head_dropout"):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must be in [0,1)")
        for key in ("use_climate", "use_performance", "use_prox", "use_aux", "use_embedding", "adaptive_prox"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.method == "climate_graph" and self.graph_name not in {"original", "enriched", "uniform"}:
            raise ValueError("ClimateGraph-PFL requires an explicitly named original/enriched/uniform graph")
        if self.method != "climate_graph" and self.graph_name is not None:
            raise ValueError("graph_name only applies to ClimateGraph-PFL")
        if self.primary_identity not in {"auto", "global", "personal"}:
            raise ValueError("primary_identity must be auto, global or personal")
        if self.primary_identity == "global" and self.method in {"local", "climate_graph"}:
            raise ValueError("Local and ClimateGraph-PFL have no single global prediction model")
        if self.primary_identity == "personal" and self.method == "central":
            raise ValueError("The central comparator has no personal model")
        for key in ("train_end", "monitor_start", "monitor_end", "validation_start", "validation_end"):
            value = getattr(self, key)
            if len(value) != 10 or date.fromisoformat(value).day != 1:
                raise ValueError(f"{key} must be a monthly ISO date")
        if not self.train_end < self.monitor_start <= self.monitor_end < self.validation_start <= self.validation_end:
            raise ValueError("Invalid monitoring/outer-validation cutoff order")

    @property
    def primary(self) -> str:
        if self.primary_identity != "auto":
            return self.primary_identity
        return "personal" if self.method in {"local", "ditto", "climate_graph", "caaa"} else "global"


class ALCE(nn.Module):
    def __init__(self, features: int, context: int, config: StrictConfig):
        super().__init__()
        if context < 0 or (context == 0 and config.use_embedding):
            raise ValueError("A zero-column climate context requires use_embedding=False")
        self.use_embedding = config.use_embedding
        self.embedding_dim = config.embedding
        self.lstm = nn.LSTM(features, config.hidden, num_layers=config.layers,
                            batch_first=True, bidirectional=True,
                            dropout=config.recurrent_dropout if config.layers > 1 else 0)
        # CORE inputs have no climate context; the disabled branch is never called.
        self.climate = nn.Sequential(nn.Linear(max(context, 1), 32), nn.Tanh(),
                                     nn.Linear(32, config.embedding), nn.Tanh())
        width = 2 * config.hidden + config.embedding
        # A linear score over [h_t, constant embedding] adds the same climate
        # offset at every t, which softmax cancels. Joint nonlinearity repairs
        # this pathway; it is a declared architecture revision shared by ALL methods.
        self.attention = nn.Sequential(nn.Linear(width, config.attention_hidden),
                                       nn.Tanh(), nn.Linear(config.attention_hidden, 1, bias=False))
        self.regression = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(config.head_dropout),
                                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.classification = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))
        if not self.use_embedding:
            for p in self.climate.parameters():
                p.requires_grad_(False)
        if not config.use_aux:
            for p in self.classification.parameters():
                p.requires_grad_(False)

    def _embedding(self, climate: torch.Tensor):
        return self.climate(climate) if self.use_embedding else climate.new_zeros((len(climate), self.embedding_dim))

    def _attention_weights(self, hidden: torch.Tensor, embedded: torch.Tensor):
        repeated = embedded[:, None, :].expand(-1, hidden.shape[1], -1)
        return torch.softmax(self.attention(torch.cat((hidden, repeated), -1)).squeeze(-1), 1)

    def attention_weights(self, hidden: torch.Tensor, climate: torch.Tensor):
        """Audit joint attention with an explicitly held-fixed historical hidden state."""
        return self._attention_weights(hidden, self._embedding(climate))

    def forward(self, x: torch.Tensor, climate: torch.Tensor):
        hidden, _ = self.lstm(x)
        embedded = self._embedding(climate)
        attention = self._attention_weights(hidden, embedded)
        context = torch.cat(((hidden * attention[:, :, None]).sum(1), embedded), -1)
        return self.regression(context).squeeze(-1), self.classification(context).squeeze(-1)


State = dict[str, torch.Tensor]


def clone_state(model_or_state) -> State:
    state = model_or_state.state_dict() if isinstance(model_or_state, nn.Module) else model_or_state
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_bytes(state: State) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


def aggregate_states(states: Sequence[State], weights: Sequence[float]) -> State:
    weights = np.asarray(weights, dtype=np.float64)
    if len(states) != len(weights) or len(states) < 1 or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid parameter aggregation states/weights")
    weights = weights / weights.sum()
    keys = list(states[0])
    if any(list(state) != keys for state in states):
        raise ValueError("Parameter state schemas differ")
    output = {}
    for key in keys:
        base = states[0][key]
        if any(s[key].shape != base.shape or s[key].dtype != base.dtype for s in states):
            raise ValueError("Parameter shape/dtype mismatch")
        if not base.is_floating_point():
            if any(not torch.equal(s[key], base) for s in states):
                raise ValueError("Nonfloating buffers require an explicit aggregation rule")
            output[key] = base.clone()
        else:
            output[key] = sum(float(w) * state[key] for w, state in zip(weights, states))
    return output


def _seed(seed: int, client: str, round_no: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}|{client}|{round_no}|{stream}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)


def caaa_weights(counts, monitor_losses, strengths, sensitivities, config: StrictConfig) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    losses = np.asarray(monitor_losses, dtype=np.float64)
    strengths = np.asarray(strengths, dtype=np.float64)
    sensitivities = np.asarray(sensitivities, dtype=np.float64)
    k = len(counts)
    if counts.shape != (k,) or losses.shape != (k,) or strengths.shape != (k, 2) or sensitivities.shape != (k, 2):
        raise ValueError("CAAA metadata axes mismatch")
    if not np.isfinite(np.r_[counts, losses, strengths.ravel(), sensitivities.ravel()]).all() or (counts <= 0).any() or (losses < 0).any():
        raise ValueError("CAAA needs finite positive sample counts and finite nonnegative losses")
    coefficients = np.array((config.alpha, config.beta if config.use_performance else 0,
                              config.gamma if config.use_climate else 0), dtype=float)
    if coefficients.sum() <= 0:
        raise ValueError("Ablation removes every active CAAA term")
    coefficients /= coefficients.sum()
    relevance = (np.abs(strengths) * np.abs(sensitivities)).sum(1)
    scores = coefficients[0] * counts / counts.sum() + coefficients[1] * np.exp(-losses) + coefficients[2] * relevance
    if not np.isfinite(scores).all() or scores.sum() <= 0:
        raise ValueError("CAAA weights degenerated; no silent size fallback")
    return scores / scores.sum()


def adaptive_lambda(drivers: torch.Tensor, sensitivity: tuple[float, float], mu: float) -> torch.Tensor:
    """Explicit Eq9 revision: origin anomaly, severity clipped before batch mean."""
    sensitivity_t = drivers.new_tensor(sensitivity).abs()
    severity = (drivers[:, -1, :].abs() * sensitivity_t).amax(1).clamp(0, 1).mean()
    return mu * (1 - 0.7 * severity)


@dataclass
class _Tensors:
    x: torch.Tensor
    climate: torch.Tensor
    drivers: torch.Tensor
    y: torch.Tensor
    heavy: torch.Tensor

    @classmethod
    def from_batch(cls, batch: LabeledBatch):
        return cls(*(torch.tensor(getattr(batch, name)) for name in ("x", "climate", "drivers", "y_z", "heavy")))


class _PrivateClient:
    def __init__(self, source: ClientInput, model_template: ALCE):
        self.client_id = source.client_id
        self._train = _Tensors.from_batch(source.train)
        self._monitor = _Tensors.from_batch(source.monitor)
        self._validation = _Tensors.from_batch(source.validation)
        self._mean, self._std = source.target_mean, source.target_std
        self.sensitivity = source.sensitivity
        self.count = len(source.train.keys)
        self.model = copy.deepcopy(model_template)

    def update(self, start: State, reference: State | None, cfg: StrictConfig,
               round_no: int, stream="main", *, adaptive=False, mu=0):
        self.model.load_state_dict(start)
        self.model.train()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(parameters, lr=cfg.lr)
        else:
            optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum)
        rng = np.random.default_rng(_seed(cfg.seed, self.client_id, round_no, stream + ":batches"))
        stats = dict(round=round_no, client_id=self.client_id, stream=stream,
                     optimizer=cfg.optimizer, optimizer_steps=0, sgd_steps=0, examples=0,
                     mse_sum=0.0, bce_sum=0.0, prox_sum=0.0, lambda_sum=0.0,
                     lambda_min=float("inf"), lambda_max=0.0)
        # Per-client dropout streams do not depend on other clients' call order.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(cfg.seed, self.client_id, round_no, stream + ":dropout"))
            for _ in range(cfg.local_epochs):
                for begin in range(0, self.count, cfg.batch_size):
                    # One epoch permutation, generated below at the first batch.
                    if begin == 0:
                        order = rng.permutation(self.count)
                    indices = torch.tensor(order[begin:begin + cfg.batch_size], dtype=torch.long)
                    batch = self._train
                    prediction, logits = self.model(batch.x[indices], batch.climate[indices])
                    mse = nn.functional.mse_loss(prediction, batch.y[indices])
                    bce = nn.functional.binary_cross_entropy_with_logits(logits, batch.heavy[indices]) if cfg.use_aux else prediction.new_zeros(())
                    coefficient = adaptive_lambda(batch.drivers[indices], self.sensitivity, mu) if adaptive and mu > 0 else prediction.new_tensor(mu)
                    prox = prediction.new_zeros(())
                    if reference is not None and mu > 0:
                        prox = sum((p - reference[name]).square().sum()
                                   for name, p in self.model.named_parameters() if p.requires_grad)
                    proximal_loss = coefficient / 2 * prox
                    loss = mse + (cfg.aux_weight * bce if cfg.use_aux else 0) + proximal_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite local objective")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, cfg.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    number = len(indices)
                    stats["optimizer_steps"] += 1
                    stats["sgd_steps"] += int(cfg.optimizer == "sgd")
                    stats["examples"] += number
                    stats["mse_sum"] += float(mse.detach()) * number
                    stats["bce_sum"] += float(bce.detach()) * number
                    stats["prox_sum"] += float(proximal_loss.detach()) * number
                    current_lambda = float(coefficient.detach())
                    stats["lambda_sum"] += current_lambda * number
                    stats["lambda_min"] = min(stats["lambda_min"], current_lambda)
                    stats["lambda_max"] = max(stats["lambda_max"], current_lambda)
        for name in ("mse", "bce", "prox", "lambda"):
            stats[name + "_mean"] = stats.pop(name + "_sum") / stats["examples"]
        # Absolute anomaly strength over exactly the examples consumed in a full epoch.
        strengths = tuple(float(v) for v in self._train.drivers[:, -1].abs().mean(0))
        self.model.eval()
        return clone_state(self.model), stats, strengths

    def evaluate(self, state: State, split: str) -> dict:
        self.model.load_state_dict(state)
        self.model.eval()
        tensors = self._monitor if split == "monitor" else self._validation
        with torch.no_grad():
            predictions = []
            for begin in range(0, len(tensors.y), 128):
                predictions.append(self.model(tensors.x[begin:begin+128], tensors.climate[begin:begin+128])[0])
            z = torch.cat(predictions).numpy().astype(np.float64)
        truth = tensors.y.numpy().astype(np.float64)
        mse_z = float(np.mean((truth - z)**2))
        y = self._mean + self._std * truth
        # The primary benchmark scores raw predictions for every learner.
        # Any nonnegative projection is a separately labelled common sensitivity.
        p = self._mean + self._std * z
        denominator = float(np.sum((y - y.mean())**2))
        if denominator <= 0:
            raise ValueError("Constant outer-validation target cannot identify R2")
        return {"n": len(y), "mse_z": mse_z, "r2": 1 - float(np.sum((y-p)**2)) / denominator}


class _Server:
    """Server methods accept parameter tensors and declared scalar metadata only."""
    def __init__(self, client_ids: Sequence[str]):
        self.client_ids = tuple(client_ids)
        self.messages: list[dict] = []

    def send(self, round_no, client, kind, payload, direction):
        size = state_bytes(payload) if isinstance(payload, dict) else np.asarray(payload).nbytes
        self.messages.append(dict(round=round_no, client_id=client, kind=kind,
                                  direction=direction, payload_bytes=int(size)))

    def aggregate(self, states, weights):
        return aggregate_states(states, weights)


@dataclass(frozen=True)
class Prediction:
    client_id: str
    identity: str
    keys: tuple[str, ...]
    z: np.ndarray
    rainfall_mm: np.ndarray
    heavy_probability: np.ndarray
    checkpoint_round: int


@dataclass
class FitResult:
    config: StrictConfig
    features: int
    context: int
    window: int
    client_ids: tuple[str, ...]
    target_stats: dict[str, tuple[float, float]]
    global_state: State | None
    personal_states: dict[str, State]
    selected_rounds: dict[str, int]
    curves: list[dict]
    local_steps: list[dict]
    messages: list[dict]
    data_manifest: dict
    elapsed_seconds: float
    round_states: list[dict] = field(repr=False, default_factory=list)

    def predict(self, client_id: str, batch: FeatureBatch, identity: str | None = None) -> Prediction:
        if isinstance(batch, LabeledBatch):
            raise TypeError("Inference accepts label-free FeatureBatch only")
        if client_id not in self.client_ids:
            raise KeyError(client_id)
        if batch.x.shape[1:] != (self.window, self.features) or batch.climate.shape[1] != self.context:
            raise ValueError("Inference feature schema does not match fitted model")
        identity = self.config.primary if identity is None else identity
        if identity not in {"global", "personal"}:
            raise ValueError("Unknown prediction identity")
        state = self.global_state if identity == "global" else self.personal_states.get(client_id)
        if state is None:
            raise ValueError(f"No fitted {identity} model for {client_id}")
        # Construction also preserves RNG for callers who are still fitting another model.
        with torch.random.fork_rng(devices=[]):
            model = ALCE(self.features, self.context, self.config)
        model.load_state_dict(state)
        model.eval()
        zs, probabilities = [], []
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self.config.cpu_threads)
        try:
            with torch.no_grad():
                for begin in range(0, len(batch.keys), self.config.batch_size):
                    z, logit = model(torch.tensor(batch.x[begin:begin+self.config.batch_size]),
                                     torch.tensor(batch.climate[begin:begin+self.config.batch_size]))
                    zs.append(z.numpy())
                    probabilities.append(torch.sigmoid(logit).numpy())
        finally:
            torch.set_num_threads(previous_threads)
        z = np.concatenate(zs)
        mean, sd = self.target_stats[client_id]
        heavy = np.concatenate(probabilities) if self.config.use_aux else np.full(len(z), np.nan, dtype=np.float32)
        return Prediction(client_id, identity, batch.keys, z, mean + sd*z,
                          heavy, self.selected_rounds[identity])

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        torch.save({"global": self.global_state, "personal": self.personal_states}, directory/"selected_states.pt")
        torch.save(self.round_states, directory/"round_states.pt")
        metadata = dict(config=asdict(self.config), features=self.features, context=self.context, window=self.window,
                        client_ids=self.client_ids, target_stats=self.target_stats, selected_rounds=self.selected_rounds,
                        data_manifest=self.data_manifest, elapsed_seconds=self.elapsed_seconds,
                        torch_version=torch.__version__, numpy_version=np.__version__,
                        architecture=ARCHITECTURE_ID,
                        rainfall_prediction_policy="raw inverse-transformed mm; no clipping in checkpoint scores or inference",
                        simulation="one-process CPU; no secure aggregation/differential privacy",
                        optimizer_state_policy="reset at each local round for every method/stream")
        (directory/"manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        for name, rows in (("curves", self.curves), ("local_steps", self.local_steps), ("messages", self.messages)):
            (directory/(name+".json")).write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
            if rows:
                columns = sorted(set().union(*(row.keys() for row in rows)))
                with (directory/(name+".csv")).open("w", newline="", encoding="utf-8") as fp:
                    writer = csv.DictWriter(fp, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)


def fit(clients: Sequence[ClientInput], config: StrictConfig, graphs: Mapping[str, GraphInput] | None = None) -> FitResult:
    """Fit using gradient-training, inner-monitor and outer-validation labels only.

    Returns separately selected global/personal states. Outer validation only
    chooses checkpoints; CAAA performance feedback uses inner-monitor labels.
    Every method shares the same gradient sample opportunity. CPU-only execution
    is intentional for the portable, auditable reference implementation.
    """
    clients = tuple(clients)
    if len(clients) < 2 or len({c.client_id for c in clients}) != len(clients):
        raise ValueError("At least two uniquely named clients are required")
    ids = tuple(c.client_id for c in clients)
    features, context, window = clients[0].train.x.shape[2], clients[0].train.climate.shape[1], clients[0].train.x.shape[1]
    schemas = {(c.train.x.shape[1:], c.train.climate.shape[1]) for c in clients}
    if len(schemas) != 1:
        raise ValueError("All clients require a common feature/window schema")
    for c in clients:
        if c.monitor.keys[0] < config.monitor_start or c.monitor.keys[-1] > config.monitor_end or c.train.keys[-1] > config.train_end:
            raise ValueError("Training/monitoring extends beyond its allowed cutoff")
        if c.validation.keys[0] < config.validation_start or c.validation.keys[-1] > config.validation_end:
            raise ValueError("Outer-validation keys are outside the allowed window")
    if len({c.validation.keys for c in clients}) != 1:
        raise ValueError("Outer-validation target-month keys must match across clients")
    graph = None
    if config.method == "climate_graph":
        if graphs is None or config.graph_name not in graphs:
            raise ValueError("Named training-derived graph is required")
        source = graphs[config.graph_name]
        if source.client_ids != ids:
            raise ValueError("Graph client order must exactly match input client order")
        graph = source.matrix.astype(np.float64)
        graph /= graph.sum(1, keepdims=True)
        if config.graph_name == "uniform" and not np.allclose(graph, np.full(graph.shape, 1/len(clients))):
            raise ValueError("Graph labelled uniform is not uniform")
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            template = ALCE(features, context, config)
        initial = clone_state(template)
        private = [_PrivateClient(c, template) for c in clients]
        server = _Server(ids)
        count_weights = np.array([c.count for c in private], dtype=float)
        count_weights /= count_weights.sum()
        global_state = clone_state(initial)
        personal = {cid: clone_state(initial) for cid in ids}
        has_global = config.method not in {"local", "climate_graph"}
        # FedAvg/FedProx personal identity is the already-computed post-update
        # local diagnostic. It adds no optimizer/model-update stream and does
        # not change the next round's global broadcast initialization.
        has_personal = config.method in {"local", "climate_graph", "ditto", "caaa", "fedavg", "fedprox"}
        for c in private:
            if config.method not in {"central", "local"}:
                server.send(0, c.client_id, "initial_model", initial, "downlink")
                server.send(0, c.client_id, "train_count", np.array(c.count, dtype=np.int64), "uplink")
            if config.method == "caaa":
                server.send(0, c.client_id, "sensitivity", np.array(c.sensitivity, dtype=np.float64), "uplink")
        if graph is not None:
            server.send(0, "server", "public_training_graph", graphs[config.graph_name].matrix, "setup_artifact")
        curves, steps, round_states = [], [], []
        best_scores = {"global": -float("inf"), "personal": -float("inf")}
        best_global, best_personal, selected = None, {}, {}
        # Explicit central comparator: pool only gradient training data, never monitors/validation.
        central = None
        if config.method == "central":
            batches = [c.train for c in clients]
            n = sum(len(b.keys) for b in batches)
            # Pooling is internal; chronology is not used as a pseudo-key for the central tensor.
            central = copy.deepcopy(private[0])
            central.client_id = "__central_pooled__"
            central.count = n
            central._train = _Tensors(*(torch.cat([getattr(c._train, name) for c in private])
                                       for name in ("x", "climate", "drivers", "y", "heavy")))
        for rd in range(1, config.rounds + 1):
            round_started = time.perf_counter()
            updates, monitoring, strengths = [], [], []
            if config.method == "central":
                global_state, log, _ = central.update(global_state, None, config, rd)
                steps.append(log)
            else:
                previous_personal = {c: clone_state(s) for c, s in personal.items()}
                for i, client in enumerate(private):
                    cid = client.client_id
                    if config.method == "local":
                        start, reference, mu, adaptive = previous_personal[cid], None, 0, False
                    elif config.method == "climate_graph":
                        reference = server.aggregate([previous_personal[j] for j in ids], graph[i])
                        start = previous_personal[cid]
                        mu = config.prox_mu if config.use_prox else 0
                        adaptive = False
                        server.send(rd, cid, "personal_graph_reference", reference, "downlink")
                    else:
                        start, reference = global_state, global_state
                        mu = config.prox_mu if config.method in {"fedprox", "caaa"} and config.use_prox else 0
                        adaptive = config.method == "caaa" and config.adaptive_prox and mu > 0
                        server.send(rd, cid, "global_reference", global_state, "downlink")
                    update, log, strength = client.update(start, reference, config, rd, adaptive=adaptive, mu=mu)
                    updates.append(update)
                    steps.append(log)
                    strengths.append(strength)
                    if config.method != "local":
                        server.send(rd, cid, "model_update", update, "uplink")
                    if config.method == "caaa":
                        loss = client.evaluate(update, "monitor")["mse_z"]
                        monitoring.append(loss)
                        if config.use_performance:
                            server.send(rd, cid, "inner_monitor_loss", np.array(loss, dtype=np.float64), "uplink")
                        if config.use_climate:
                            server.send(rd, cid, "round_climate_strength", np.array(strength, dtype=np.float64), "uplink")
                    if config.method == "ditto":
                        personal[cid], personal_log, _ = client.update(previous_personal[cid], global_state, config, rd,
                                                                      stream="ditto_personal", mu=config.prox_mu if config.use_prox else 0)
                        steps.append(personal_log)
                    elif has_personal:
                        personal[cid] = update
                if config.method not in {"local", "climate_graph"}:
                    weights = caaa_weights([c.count for c in private], monitoring, strengths,
                                           [c.sensitivity for c in private], config) if config.method == "caaa" else count_weights
                    global_state = server.aggregate(updates, weights)
                    curves.append(dict(round=rd, kind="aggregation", weights={c: float(w) for c, w in zip(ids, weights)},
                                       strengths={c: list(s) for c, s in zip(ids, strengths)},
                                       inner_monitor_losses={c: float(v) for c, v in zip(ids, monitoring)}))
            for identity, available in (("global", has_global), ("personal", has_personal)):
                if not available:
                    continue
                scores = {}
                monitor_scores = {}
                for client in private:
                    state = global_state if identity == "global" else personal[client.client_id]
                    scores[client.client_id] = client.evaluate(state, "validation")
                    monitor_scores[client.client_id] = client.evaluate(state, "monitor")
                    if config.method not in {"central", "local"}:
                        # Two scalar outer metrics/checkpoint metadata; no labels or predictions are uploaded.
                        payload = np.array((scores[client.client_id]["mse_z"], scores[client.client_id]["r2"]), dtype=np.float64)
                        server.send(rd, client.client_id, "outer_validation_metrics_"+identity, payload, "uplink")
                        monitor_payload = np.array((monitor_scores[client.client_id]["mse_z"],
                                                    monitor_scores[client.client_id]["r2"]), dtype=np.float64)
                        server.send(rd, client.client_id, "inner_monitor_diagnostic_metrics_"+identity, monitor_payload, "uplink")
                macro = float(np.mean([v["r2"] for v in scores.values()]))
                curves.append(dict(round=rd, kind="evaluation", identity=identity, validation_macro_r2=macro,
                                   validation=scores, inner_monitor=monitor_scores,
                                   round_elapsed_seconds=time.perf_counter()-round_started))
                if macro > best_scores[identity] + 1e-12:
                    best_scores[identity] = macro
                    selected[identity] = rd
                    if identity == "global":
                        best_global = clone_state(global_state)
                    else:
                        best_personal = {c: clone_state(s) for c, s in personal.items()}
            if rd == 1 or rd == config.rounds or rd % config.state_checkpoint_interval == 0:
                round_states.append(dict(round=rd, global_state=clone_state(global_state) if has_global else None,
                                         personal_states={c: clone_state(s) for c, s in personal.items()} if has_personal else {}))
        manifest = {c.client_id: {s: {"count": len(getattr(c, s).keys), "first": getattr(c, s).keys[0],
                                     "last": getattr(c, s).keys[-1], "keys": getattr(c, s).keys}
                                  for s in ("train", "monitor", "validation")} for c in clients}
        return FitResult(config, features, context, window, ids,
                         {c.client_id: (c.target_mean, c.target_std) for c in clients},
                         best_global, best_personal, selected, curves, steps, server.messages,
                         manifest, time.perf_counter()-started, round_states)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def load_fit(directory: str | Path, *, load_round_history: bool = True) -> FitResult:
    """Load saved selected states; periodic histories are optional diagnostics.

    Inference does not use round histories. The default retains the historical
    inspection API; production evaluation explicitly skips those large tensors.
    """
    if type(load_round_history) is not bool:
        raise TypeError('load_round_history must be a boolean')
    directory = Path(directory)
    metadata = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if metadata.get("architecture") != ARCHITECTURE_ID:
        raise ValueError("Saved fit uses a different ALCE architecture; historical linear-attention states cannot be silently converted")
    config = StrictConfig(**metadata["config"])
    selected = torch.load(directory/"selected_states.pt", map_location="cpu", weights_only=True)
    rounds = torch.load(directory/"round_states.pt", map_location="cpu", weights_only=True) if load_round_history else []
    logs = {name: json.loads((directory/(name+".json")).read_text(encoding="utf-8"))
            for name in ("curves", "local_steps", "messages")}
    return FitResult(config, metadata["features"], metadata["context"], metadata["window"],
                     tuple(metadata["client_ids"]), {c: tuple(v) for c, v in metadata["target_stats"].items()},
                     selected["global"], selected["personal"], metadata["selected_rounds"],
                     logs["curves"], logs["local_steps"], logs["messages"],
                     metadata["data_manifest"], metadata["elapsed_seconds"], rounds)
