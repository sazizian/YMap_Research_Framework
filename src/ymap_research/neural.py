"""Optional, genuine frozen-text + trainable-graph/symbolic ensemble.

Importing this module does not import torch or download a model. Missing weights
or dependencies are errors, never a request to substitute synthetic embeddings.
See docs/NEURAL_BACKEND.md for execution status and exact model semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .representations import GRAPH_NODE_FEATURES, GRAPH_RELATIONS, REPRESENTATION_VERSION, canonical_ddl, representation_key, schema_graph, symbolic_feature_names, symbolic_matrix


@dataclass(frozen=True)
class EncoderConfig:
    family: str
    model_name_or_path: str
    revision: str | None = None
    max_length: int = 512
    local_files_only: bool = True
    batch_size: int = 4

    def __post_init__(self):
        if self.family not in ("graphcodebert", "codet5", "llama"):
            raise ValueError("encoder family must be graphcodebert, codet5, or llama")
        if not isinstance(self.model_name_or_path, str) or not self.model_name_or_path.strip():
            raise ValueError("an actual model checkpoint path or identifier is required")
        if type(self.local_files_only) is not bool:
            raise ValueError("local_files_only must be a boolean")
        if type(self.max_length) is not int or type(self.batch_size) is not int or self.max_length < 2 or self.batch_size < 1:
            raise ValueError("max_length must be >=2 and batch_size must be positive")


@dataclass(frozen=True)
class NeuralConfig:
    # Explicitly empty by default: this is a graph+symbolic ablation, not the full model.
    text_encoders: tuple[EncoderConfig, ...] = ()
    use_graph: bool = True
    use_symbolic: bool = True
    ensemble_size: int = 5
    seed: int = 42
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    graph_layers: int = 2
    dropout: float = 0.1
    weight_decay: float = 1e-4
    patience: int = 15
    device: str = "cpu"
    cache_dir: str | None = None
    deterministic: bool = True
    target_names: tuple[str, ...] = ("insert_ms", "query_ms", "storage_mb")

    def __post_init__(self):
        if any(type(value) is not bool for value in (self.use_graph, self.use_symbolic, self.deterministic)):
            raise ValueError("use_graph, use_symbolic and deterministic must be booleans")
        if any(not isinstance(encoder, EncoderConfig) for encoder in self.text_encoders):
            raise ValueError("text_encoders must contain EncoderConfig instances")
        if not self.use_graph and not self.use_symbolic and not self.text_encoders:
            raise ValueError("at least one view must be enabled")
        counts = (self.ensemble_size, self.epochs, self.batch_size, self.hidden_dim, self.graph_layers, self.patience)
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("ensemble_size, epochs, batch_size, hidden_dim, graph_layers and patience must be positive")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - self.ensemble_size:
            raise ValueError("seed and per-member seeds must be integers in [0, 2**32)")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (self.dropout, self.learning_rate, self.weight_decay)) or not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid dropout, learning rate, or weight decay")
        if not self.target_names or any(not isinstance(name, str) or not name for name in self.target_names) or len(set(self.target_names)) != len(self.target_names):
            raise ValueError("target_names must be unique and nonempty")
        if len({e.family for e in self.text_encoders}) != len(self.text_encoders):
            raise ValueError("configure at most one checkpoint for each encoder family")


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("The neural backend requires the optional neural dependencies. Install with: pip install -e '.[neural]'. No hash/fake fallback is available.") from exc
    return torch


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "transformers", "tokenizers", "safetensors", "sentencepiece"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_checkpoint_manifest(path: Path) -> dict[str, str] | None:
    if not path.is_dir():
        return None
    # Include real weights, tokenizer, and configuration. This may read many GB;
    # once per encoder initialization it protects cache/reload from stale weights.
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in suffixes and ".git" not in p.parts)
    return {str(p.relative_to(path)): _file_sha256(p) for p in files}


def _encoder_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Stable checkpoint/preprocessing identity, excluding diagnostic ordering.

    Requested moving tags, batching and load-time diagnostics do not identify
    weights. The resolved commit or content manifest does. Execution information
    is added separately to cache keys, so caches do not cross devices/versions.
    """
    config = metadata["config"]
    fields = ("resolved_commit", "local_files_sha256", "model_type", "hidden_dimension", "pooling", "canonicalization", "tokenizer_class", "usable_token_limit")
    return {"family": config["family"], "source": config["model_name_or_path"], "max_length": config["max_length"], **{name: metadata.get(name) for name in fields}}


def _usable_token_limit(model_config, tokenizer, family: str) -> int | None:
    """Return a real token capacity, ignoring HF's huge 'unknown' sentinel."""
    limits = []
    configured = getattr(model_config, "max_position_embeddings", None)
    if isinstance(configured, int) and configured > 0:
        if family == "graphcodebert":
            # RoBERTa starts nonpadding positions at padding_idx + 1.
            padding_index = getattr(model_config, "pad_token_id", None)
            if padding_index is None:
                padding_index = tokenizer.pad_token_id
            configured -= int(padding_index or 0) + 1
        limits.append(configured)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000_000:
        limits.append(tokenizer_limit)
    return min(limits) if limits else None


class FrozenTextEncoder:
    """Load real HF weights, freeze all parameters, and mask-mean-pool tokens.

    GraphCodeBERT receives canonical SQL tokens through its standard text API;
    this does not implement its original code data-flow graph attention.
    CodeT5 encoder-decoder checkpoints use only ``get_encoder()``.
    """

    def __init__(self, config: EncoderConfig, device: str = "cpu", cache_dir: str | None = None, *, resolved_revision: str | None = None):
        torch = _torch()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Frozen text encoders require transformers and actual checkpoint files. Install the neural extra and supply checkpoints; no fallback is available.") from exc
        self.config, self.torch, self.device = config, torch, device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"local_files_only": config.local_files_only, "trust_remote_code": False}
        if config.revision is not None:
            kwargs["revision"] = config.revision
        is_local = Path(config.model_name_or_path).is_dir()
        if resolved_revision is not None and not is_local:
            kwargs["revision"] = resolved_revision
        try:
            full_model, loading_info = AutoModel.from_pretrained(config.model_name_or_path, output_loading_info=True, **kwargs)
            # Resolve once through the weights, then pin the tokenizer to the
            # same commit; a moving Hub branch cannot change between two loads.
            resolved_commit = getattr(full_model.config, "_commit_hash", None)
            tokenizer_kwargs = dict(kwargs)
            if resolved_commit is not None and not is_local:
                tokenizer_kwargs["revision"] = resolved_commit
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True, **tokenizer_kwargs)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not load genuine {config.family} checkpoint {config.model_name_or_path!r}. local_files_only={config.local_files_only}. Provide compatible local weights/tokenizer or intentionally configure online access and a pinned revision.") from exc
        model_config = full_model.config
        model_type = getattr(model_config, "model_type", None)
        expected_types = {"graphcodebert": {"roberta"}, "codet5": {"t5", "codet5p"}, "llama": {"llama"}}
        if model_type not in expected_types[config.family]:
            raise ValueError(f"checkpoint model_type={model_type!r} is incompatible with declared encoder family {config.family!r}")
        if config.family == "codet5" and getattr(model_config, "is_encoder_decoder", False):
            if not hasattr(full_model, "get_encoder"):
                raise ValueError("CodeT5 encoder-decoder checkpoint does not expose get_encoder()")
            self.model = full_model.get_encoder()
        else:
            if getattr(model_config, "is_encoder_decoder", False):
                raise ValueError("encoder-decoder checkpoints are supported only in the codet5 family")
            self.model = full_model
        def unused_parameter(name):
            # A Roberta pooler is not used by last_hidden_state mean pooling.
            # CodeT5 decoding parameters are not used by its encoder-only view.
            return ((config.family == "graphcodebert" and (name.startswith("pooler.") or name.startswith("roberta.pooler."))) or (config.family == "codet5" and getattr(model_config, "is_encoder_decoder", False) and name.startswith("decoder.")))
        missing_used = [name for name in loading_info.get("missing_keys", []) if not unused_parameter(name)]
        mismatched_used = [entry for entry in loading_info.get("mismatched_keys", []) if not unused_parameter(entry[0])]
        if missing_used or mismatched_used or loading_info.get("error_msgs"):
            raise ValueError(f"checkpoint leaves required frozen encoder weights missing/mismatched: missing={missing_used}, mismatched={mismatched_used}, errors={loading_info.get('error_msgs', [])}")
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                raise ValueError("checkpoint tokenizer needs a pad, eos, or unk token")
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        # A declared limit is enforced, never silently raised to the checkpoint limit.
        checkpoint_limit = _usable_token_limit(model_config, self.tokenizer, config.family)
        if checkpoint_limit is not None and config.max_length > checkpoint_limit:
            raise ValueError(f"max_length={config.max_length} exceeds checkpoint positional limit={checkpoint_limit}")
        self.dimension = int(getattr(model_config, "hidden_size", None) or getattr(model_config, "d_model", 0))
        if self.dimension < 1:
            raise ValueError("cannot determine the checkpoint hidden dimension")
        manifest = _local_checkpoint_manifest(Path(config.model_name_or_path))
        if manifest is None and resolved_commit is None:
            raise ValueError("checkpoint has neither a local content manifest nor a resolved Hub commit; cache provenance cannot be established")
        diagnostics = json.loads(json.dumps(loading_info, default=str))
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if key in diagnostics:
                diagnostics[key] = sorted(diagnostics[key], key=lambda value: json.dumps(value, sort_keys=True))
        execution = {"device": str(device), "dtype": str(next(self.model.parameters()).dtype), "packages": _environment()["packages"]}
        self.metadata = {"config": asdict(config), "resolved_commit": resolved_commit, "local_files_sha256": manifest, "model_type": model_type, "hidden_dimension": self.dimension, "pooling": "attention_mask_mean_float32", "canonicalization": REPRESENTATION_VERSION, "tokenizer_class": type(self.tokenizer).__name__, "usable_token_limit": checkpoint_limit, "model_config": json.loads(model_config.to_json_string(use_diff=False)), "loading_diagnostics": diagnostics, "execution": execution}
        self.cache_parameters = {"encoder": _encoder_identity(self.metadata), "execution": execution, "representation_version": REPRESENTATION_VERSION}
        self.truncation_count = 0

    def encode(self, ddls: Sequence[str]) -> np.ndarray:
        if not ddls:
            return np.empty((0, self.dimension), dtype=np.float32)
        rows: list[np.ndarray | None] = [None] * len(ddls)
        missing: dict[str, tuple[str, list[int]]] = {}
        for i, ddl in enumerate(ddls):
            canonical = canonical_ddl(ddl)
            key = representation_key(ddl, self.cache_parameters)
            cache_file = self.cache_dir / f"{key}.npy" if self.cache_dir else None
            if cache_file is not None and cache_file.exists():
                row = np.load(cache_file, allow_pickle=False)
                if row.shape != (self.dimension,) or not np.isfinite(row).all():
                    raise ValueError(f"invalid encoder cache file: {cache_file}")
                with np.errstate(over="ignore", invalid="ignore"):
                    converted = row.astype(np.float32)
                if not np.isfinite(converted).all():
                    raise ValueError(f"encoder cache file exceeds finite float32 range: {cache_file}")
                rows[i] = converted
            elif key in missing:
                missing[key][1].append(i)
            else:
                missing[key] = (canonical, [i])
        pending = list(missing.items())
        for start in range(0, len(pending), self.config.batch_size):
            subset = pending[start:start + self.config.batch_size]
            texts = [value[0] for _, value in subset]
            lengths = self.tokenizer(texts, truncation=False, padding=False, return_length=True, verbose=False)["length"]
            self.truncation_count += sum(int(length > self.config.max_length) for length in lengths)
            batch = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=self.config.max_length)
            # Decoder-only LLaMA and T5 encoder implementations do not accept token_type_ids.
            inputs = {key: value.to(self.device) for key, value in batch.items() if key in ("input_ids", "attention_mask")}
            with self.torch.inference_mode():
                output = self.model(**inputs, return_dict=True)
                hidden = output.last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                vectors = pooled.cpu().numpy().astype(np.float32)
            if vectors.shape != (len(subset), self.dimension) or not np.isfinite(vectors).all():
                raise ValueError("checkpoint produced nonfinite or incorrectly shaped embeddings")
            for (key, (_, indices)), row in zip(subset, vectors):
                if self.cache_dir is not None:
                    with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npy", delete=False) as stream:
                        temporary = Path(stream.name)
                        np.save(stream, row, allow_pickle=False)
                    temporary.replace(self.cache_dir / f"{key}.npy")
                for i in indices:
                    rows[i] = row
        return np.vstack(rows).astype(np.float32)


def _network_class(torch):
    nn = torch.nn

    class TypedGraphEncoder(nn.Module):
        def __init__(self, hidden_dim: int, layers: int, dropout: float):
            super().__init__()
            self.input = nn.Linear(len(GRAPH_NODE_FEATURES), hidden_dim)
            self.self_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
            self.relation_layers = nn.ModuleList(nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in GRAPH_RELATIONS) for _ in range(layers))
            self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(hidden_dim * 2, hidden_dim)

        def forward(self, graph):
            x, sources, targets, relations = graph
            h = torch.relu(self.input(x))
            for self_layer, relation_layers, norm in zip(self.self_layers, self.relation_layers, self.norms):
                aggregate = torch.zeros_like(h)
                degree = h.new_zeros((h.shape[0], 1))
                for relation, layer in enumerate(relation_layers):
                    selection = relations == relation
                    if selection.any():
                        src, dst = sources[selection], targets[selection]
                        aggregate.index_add_(0, dst, layer(h[src]))
                        degree.index_add_(0, dst, h.new_ones((len(dst), 1)))
                h = self.dropout(torch.relu(norm(self_layer(h) + aggregate / degree.clamp_min(1))))
            pooled = []
            for column in (0, 1):
                chosen = x[:, column] > 0.5
                pooled.append(h[chosen].mean(dim=0) if chosen.any() else h.new_zeros(h.shape[1]))
            return self.output(torch.cat(pooled))

    class MultiViewNetwork(nn.Module):
        def __init__(self, config, symbolic_dim, text_dims):
            super().__init__()
            dim = config.hidden_dim
            self.graph = TypedGraphEncoder(dim, config.graph_layers, config.dropout) if config.use_graph else None
            self.symbolic = nn.Sequential(nn.Linear(symbolic_dim, dim), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(dim, dim)) if config.use_symbolic else None
            self.text_projections = nn.ModuleList(nn.Sequential(nn.Linear(size, dim), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(dim, dim)) for size in text_dims)
            view_count = int(config.use_graph) + int(config.use_symbolic) + len(text_dims)
            self.gates = nn.ModuleList(nn.Linear(dim, 1) for _ in range(view_count))
            self.regressor = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(dim, len(config.target_names)))

        def forward(self, symbolic, texts, graphs, indices):
            views = []
            if self.graph is not None:
                views.append(torch.stack([self.graph(graphs[i]) for i in indices]))
            if self.symbolic is not None:
                views.append(self.symbolic(symbolic[indices]))
            for projection, text in zip(self.text_projections, texts):
                views.append(projection(text[indices]))
            scores = torch.cat([gate(view) for gate, view in zip(self.gates, views)], dim=1)
            weights = torch.softmax(scores, dim=1)
            fused = sum(view * weights[:, i:i + 1] for i, view in enumerate(views))
            return self.regressor(fused)

    return MultiViewNetwork


class NeuralMultiViewRegressor:
    """Independent trainable fusion/graph ensemble sharing frozen text encoders.

    ``fit`` never creates a schema-random validation split. Supply distinct
    object-model groups for validation, or training runs for fixed ``epochs``.
    Mean/scale of features and targets are estimated from training records only.
    """

    def __init__(self, config: NeuralConfig | None = None):
        self.config = config or NeuralConfig()
        self.members_: list[Any] = []
        self.encoders_: list[FrozenTextEncoder] = []
        self.fitted_ = False

    def _records(self, records):
        records = list(records)
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("schema_id"), str) or not isinstance(record.get("object_model"), str):
                raise ValueError("each candidate requires string schema_id and object_model")
            canonical_ddl(record.get("ddl"))
        return records

    def _seed(self, seed):
        torch = self.torch_
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(self.config.deterministic)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = self.config.deterministic

    def _initialize_encoders(self, expected_metadata=None):
        self.encoders_ = [FrozenTextEncoder(config, self.config.device, self.config.cache_dir, resolved_revision=expected_metadata[index].get("resolved_commit") if expected_metadata is not None else None) for index, config in enumerate(self.config.text_encoders)]

    def _prepare(self, records):
        torch = self.torch_
        symbolic = None
        if self.config.use_symbolic:
            array = symbolic_matrix(records, self.feature_names_)
            array = (array - self.feature_mean_) / self.feature_scale_
            if not np.isfinite(array).all() or np.any(np.abs(array) > np.finfo(np.float32).max):
                raise ValueError("normalized symbolic features exceed finite float32 range")
            symbolic = torch.as_tensor(array, dtype=torch.float32, device=self.config.device)
        texts = [torch.as_tensor(encoder.encode([record["ddl"] for record in records]), dtype=torch.float32, device=self.config.device) for encoder in self.encoders_]
        graphs = []
        if self.config.use_graph:
            for record in records:
                graph = schema_graph(record.get("graph"))
                graphs.append((torch.as_tensor(graph.node_features, dtype=torch.float32, device=self.config.device), *(torch.as_tensor(array, dtype=torch.long, device=self.config.device) for array in (graph.edge_sources, graph.edge_targets, graph.edge_types))))
        return symbolic, texts, graphs

    def _new_member(self):
        return _network_class(self.torch_)(self.config, len(self.feature_names_), [encoder.dimension for encoder in self.encoders_]).to(device=self.config.device, dtype=self.torch_.float32)

    def fit(self, schema_records, y, validation_records=None, validation_y=None):
        records = self._records(schema_records)
        targets = np.asarray(y, dtype=np.float64)
        expected = (len(records), len(self.config.target_names))
        if len(records) < 2 or targets.shape != expected or not np.isfinite(targets).all():
            raise ValueError(f"fit requires >=2 candidates and finite targets of shape {expected}")
        if (validation_records is None) != (validation_y is None):
            raise ValueError("supply both validation_records and validation_y or neither")
        validation = self._records(validation_records) if validation_records is not None else None
        validation_targets = None
        if validation is not None:
            validation_targets = np.asarray(validation_y, dtype=np.float64)
            if not validation or validation_targets.shape != (len(validation), targets.shape[1]) or not np.isfinite(validation_targets).all():
                raise ValueError("invalid validation targets")
            overlap = {r["object_model"] for r in records} & {r["object_model"] for r in validation}
            if overlap:
                raise ValueError(f"validation object_model groups overlap training: {sorted(overlap)}")
            train_ddls = {canonical_ddl(r["ddl"]) for r in records}
            if train_ddls & {canonical_ddl(r["ddl"]) for r in validation}:
                raise ValueError("identical canonical DDL appears in training and validation")
        self.fitted_ = False
        self.torch_ = _torch()
        self._seed(self.config.seed)
        self.feature_names_ = symbolic_feature_names(records) if self.config.use_symbolic else ()
        raw_features = symbolic_matrix(records, self.feature_names_) if self.config.use_symbolic else np.empty((len(records), 0))
        self.feature_mean_ = raw_features.mean(axis=0, dtype=np.float64)
        self.feature_scale_ = raw_features.std(axis=0, dtype=np.float64)
        self.feature_scale_[self.feature_scale_ < 1e-8] = 1.0
        self.target_mean_ = targets.mean(axis=0)
        self.target_scale_ = targets.std(axis=0)
        if not all(np.isfinite(array).all() for array in (self.feature_mean_, self.feature_scale_, self.target_mean_, self.target_scale_)):
            raise ValueError("feature or target normalization moments are not finite")
        self.target_scale_[self.target_scale_ < 1e-8] = 1.0
        self._initialize_encoders()
        prepared = self._prepare(records)
        normalized = self.torch_.as_tensor((targets - self.target_mean_) / self.target_scale_, dtype=self.torch_.float32, device=self.config.device)
        validation_prepared = self._prepare(validation) if validation is not None else None
        validation_normalized = self.torch_.as_tensor((validation_targets - self.target_mean_) / self.target_scale_, dtype=self.torch_.float32, device=self.config.device) if validation is not None else None
        self.members_, self.history_ = [], []
        for member_id in range(self.config.ensemble_size):
            seed = self.config.seed + member_id
            self._seed(seed)
            rng = np.random.default_rng(seed)
            # Bootstrap only training candidates; each ensemble member has independent weights.
            sampled = rng.integers(0, len(records), size=len(records))
            member = self._new_member()
            optimizer = self.torch_.optim.AdamW(member.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
            best_loss, best_state, stale, history = float("inf"), None, 0, []
            for epoch in range(self.config.epochs):
                member.train()
                epoch_losses = []
                shuffled = rng.permutation(sampled)
                for start in range(0, len(records), self.config.batch_size):
                    indices = shuffled[start:start + self.config.batch_size].tolist()
                    optimizer.zero_grad(set_to_none=True)
                    prediction = member(*prepared, indices)
                    loss = self.torch_.nn.functional.mse_loss(prediction, normalized[indices])
                    if not self.torch_.isfinite(loss):
                        raise FloatingPointError("nonfinite neural training loss")
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append((float(loss.detach().cpu()), len(indices)))
                row = {"epoch": epoch + 1, "train_normalized_mse": sum(value * size for value, size in epoch_losses) / len(records)}
                if validation_prepared is not None:
                    member.eval()
                    with self.torch_.inference_mode():
                        validation_prediction = self._forward_batches(member, validation_prepared, len(validation))
                        val_loss = float(self.torch_.nn.functional.mse_loss(validation_prediction, validation_normalized).cpu())
                    if not np.isfinite(val_loss):
                        raise FloatingPointError("nonfinite validation loss")
                    row["validation_normalized_mse"] = val_loss
                    if val_loss < best_loss:
                        best_loss, stale = val_loss, 0
                        best_state = {key: value.detach().cpu().clone() for key, value in member.state_dict().items()}
                    else:
                        stale += 1
                history.append(row)
                if validation_prepared is not None and stale >= self.config.patience:
                    break
            if best_state is not None:
                member.load_state_dict(best_state)
            member.eval()
            self.members_.append(member)
            self.history_.append({"member": member_id, "seed": seed, "bootstrap_indices": sampled.tolist(), "epochs": history})
        self.training_groups_ = sorted({record["object_model"] for record in records})
        self.validation_groups_ = sorted({record["object_model"] for record in validation}) if validation is not None else []
        self.training_candidate_ids_ = [record["schema_id"] for record in records]
        self.environment_ = _environment()
        self.environment_["torch_cuda_version"] = self.torch_.version.cuda
        self.environment_["device"] = self.config.device
        self.fitted_ = True
        return self

    def _forward_batches(self, member, prepared, count):
        return self.torch_.cat([member(*prepared, list(range(start, min(start + self.config.batch_size, count)))) for start in range(0, count, self.config.batch_size)], dim=0)

    def predict_members(self, schema_records):
        if not self.fitted_:
            raise RuntimeError("fit or load the neural model before prediction")
        records = self._records(schema_records)
        if not records:
            return np.empty((len(self.members_), 0, len(self.config.target_names)), dtype=np.float64)
        prepared = self._prepare(records)
        with self.torch_.inference_mode():
            predictions = np.stack([self._forward_batches(member.eval(), prepared, len(records)).cpu().numpy() for member in self.members_])
        return predictions * self.target_scale_[None, None, :] + self.target_mean_[None, None, :]

    def predict(self, schema_records):
        return self.predict_members(schema_records).mean(axis=0)

    def metadata(self):
        if not self.fitted_:
            raise RuntimeError("model has not been fit")
        return {"format_version": 1, "backend": "neural_frozen_text_typed_graph_symbolic_gated_ensemble", "representation_version": REPRESENTATION_VERSION, "config": asdict(self.config), "feature_names": list(self.feature_names_), "feature_mean": self.feature_mean_.tolist(), "feature_scale": self.feature_scale_.tolist(), "target_mean": self.target_mean_.tolist(), "target_scale": self.target_scale_.tolist(), "training_groups": self.training_groups_, "validation_groups": self.validation_groups_, "training_candidate_ids": self.training_candidate_ids_, "encoders": [encoder.metadata for encoder in self.encoders_], "truncation_counts_this_process": [encoder.truncation_count for encoder in self.encoders_], "environment": self.environment_, "history": self.history_, "uncertainty": "ensemble disagreement only; not a calibrated predictive interval"}

    def save(self, path):
        """Write JSON metadata and tensor-only state files to a local directory."""
        metadata = self.metadata()
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for index, member in enumerate(self.members_):
            filename = f"member_{index:03d}.pt"
            self.torch_.save({key: value.detach().cpu() for key, value in member.state_dict().items()}, destination / filename)
            manifest[filename] = _file_sha256(destination / filename)
        metadata["member_files_sha256"] = manifest
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path, *, trusted: bool = False, device: str | None = None):
        """Load only artifacts you trust, with weights_only=True and SHA256 checks.

        Hashes detect corruption, not authenticity. Never use a model directory
        supplied by an unknown source, even though arbitrary pickled objects are
        disallowed by the weights-only loader.
        """
        if not trusted:
            raise ValueError("load requires trusted=True for a trusted local artifact directory")
        source = Path(path)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or metadata.get("representation_version") != REPRESENTATION_VERSION:
            raise ValueError("unsupported neural artifact/representation version")
        raw_config = dict(metadata["config"])
        raw_config["text_encoders"] = tuple(EncoderConfig(**entry) for entry in raw_config["text_encoders"])
        raw_config["target_names"] = tuple(raw_config["target_names"])
        if device is not None:
            raw_config["device"] = device
        result = cls(NeuralConfig(**raw_config))
        result.torch_ = _torch()
        result.feature_names_ = tuple(metadata["feature_names"])
        for name in ("feature_mean", "feature_scale", "target_mean", "target_scale"):
            setattr(result, name + "_", np.asarray(metadata[name], dtype=np.float64))
        if len(metadata["encoders"]) != len(result.config.text_encoders):
            raise ValueError("encoder metadata does not match configured text views")
        result._initialize_encoders(metadata["encoders"])
        if [_encoder_identity(encoder.metadata) for encoder in result.encoders_] != [_encoder_identity(entry) for entry in metadata["encoders"]]:
            raise ValueError("frozen checkpoint provenance differs from the trained artifact")
        filenames = [f"member_{index:03d}.pt" for index in range(result.config.ensemble_size)]
        if set(filenames) != set(metadata["member_files_sha256"]):
            raise ValueError("member checkpoint manifest does not match ensemble_size")
        for filename in filenames:
            if _file_sha256(source / filename) != metadata["member_files_sha256"][filename]:
                raise ValueError(f"member checkpoint checksum mismatch: {filename}")
            state = result.torch_.load(source / filename, map_location=result.config.device, weights_only=True)
            member = result._new_member()
            member.load_state_dict(state, strict=True)
            result.members_.append(member.eval())
        result.training_groups_ = metadata["training_groups"]
        result.validation_groups_ = metadata["validation_groups"]
        result.training_candidate_ids_ = metadata["training_candidate_ids"]
        result.environment_ = metadata["environment"]
        result.history_ = metadata["history"]
        result.fitted_ = True
        return result
