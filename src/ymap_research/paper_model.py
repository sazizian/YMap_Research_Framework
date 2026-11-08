"""The manuscript's optional five-view, three-objective neural regressor.

The legacy graph/descriptor backend is a separate experiment. This module never
downloads checkpoints or substitutes synthetic embeddings when weights are absent.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .neural import _environment, _local_checkpoint_manifest, _torch, _usable_token_limit


TEXT_VIEWS = ("graphcodebert", "codet5", "llama")
PAPER_VIEWS = (*TEXT_VIEWS, "graph", "symbolic")
TEXT_SPECS = {
    "graphcodebert": ("microsoft/graphcodebert-base", "roberta", 768, 512),
    "codet5": ("Salesforce/codet5p-220m", "t5", 768, 512),
    "llama": ("meta-llama/Llama-2-7b-hf", "llama", 4096, 4096),
}
VIEW_ALIASES = {"gcb": "graphcodebert", "ct5": "codet5", "llm": "llama", "descriptors": "symbolic"}
MODEL_VERSION = "paper-model-v1"


@dataclass(frozen=True)
class PaperModelConfig:
    """Neural settings. Pass checkpoint paths as family -> local directory."""

    device: str = "cuda"
    cache_dir: str | None = None
    checkpoints: dict = field(default_factory=dict)
    epochs: int = 50
    batch_size: int = 64
    patience: int = 10
    fusion: str = "gated"
    symbolic_hidden: int = 128
    gate_hidden: int = 512

    def __post_init__(self):
        if self.fusion not in ("gated", "concatenate"):
            raise ValueError("paper neural fusion must be 'gated' or 'concatenate'")
        for key in ("epochs", "batch_size", "patience", "symbolic_hidden", "gate_hidden"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"paper neural {key} must be a positive integer")
        if self.epochs > 50:
            raise ValueError("the paper neural epoch cap is 50")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("paper neural device must be a PyTorch device string")
        if self.cache_dir is not None and not isinstance(self.cache_dir, (str, Path)):
            raise ValueError("paper neural cache_dir must be a local path")
        if not isinstance(self.checkpoints, Mapping):
            raise ValueError("paper neural checkpoints must be a family-to-directory mapping")
        unknown = set(self.checkpoints) - set(TEXT_VIEWS)
        if unknown:
            raise ValueError(f"unknown paper neural checkpoint families: {sorted(unknown)}")

    @classmethod
    def from_mapping(cls, config):
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("paper model config must be a mapping or PaperModelConfig")
        # Accept either the complete study config or its neural subsection.
        if "neural" not in config and set(config).isdisjoint(cls.__dataclass_fields__):
            return cls()
        values = config.get("neural", config)
        if not isinstance(values, Mapping):
            raise ValueError("paper study config 'neural' must be a mapping")
        # Other study configuration is read by the evaluator.
        if "neural" in config:
            unknown = set(values) - set(cls.__dataclass_fields__)
            if unknown:
                raise ValueError(f"unknown paper neural settings: {sorted(unknown)}")
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


def _paper_views(views):
    if views is None:
        return PAPER_VIEWS
    if isinstance(views, str):
        raise ValueError("pass paper neural views as a sequence of names")
    names = tuple(VIEW_ALIASES.get(view, view) for view in views)
    if not names or len(set(names)) != len(names) or set(names) - set(PAPER_VIEWS):
        raise ValueError(f"select distinct paper neural views from {PAPER_VIEWS}")
    return tuple(view for view in PAPER_VIEWS if view in names)


def _checkpoint_specs(config, views):
    specs = {}
    for family in TEXT_VIEWS:
        if family not in views:
            continue
        value = config.checkpoints.get(family)
        value = {"path": value} if isinstance(value, (str, Path)) else value
        default_batch = 1 if family == "llama" else 4
        label = TEXT_SPECS[family][0]
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), (str, Path)):
            raise FileNotFoundError(
                f"The {family} view needs a local {label} checkpoint. Set "
                f"neural.checkpoints.{family} to its directory, including tokenizer and weights."
            )
        if set(value) - {"path", "batch_size"}:
            raise ValueError(f"neural.checkpoints.{family} accepts only path and batch_size")
        path = Path(value["path"]).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"The {family} checkpoint directory {path} needs {label} config.json, "
                "weights, and tokenizer files."
            )
        batch = value.get("batch_size", default_batch)
        if type(batch) is not int or batch < 1:
            raise ValueError(f"neural.checkpoints.{family}.batch_size must be a positive integer")
        specs[family] = path, batch
    return specs


def _target_array(values, n, split):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (n, 3) or not n or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(
            f"{split} requires {n} aligned rows of three finite positive measurements "
            "(insert latency, query latency, storage) for the paper log transform."
        )
    return values


def _check_system_splits(train_records, val_records, test_records):
    groups = []
    for name, records in (("training", train_records), ("validation", val_records), ("test", test_records)):
        models = set()
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("object_model"), str) or not record["object_model"]:
                raise ValueError(f"each {name} paper record must identify its object_model")
            models.add(record["object_model"])
        if any(models.intersection(other) for other in groups):
            raise ValueError("paper neural training, validation and test object models must be disjoint")
        groups.append(models)


def _standardizer(values):
    mean, scale = values.mean(axis=0), values.std(axis=0)
    return mean, np.where(scale == 0, 1.0, scale)


def _invert_targets(predictions, mean, scale):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        values = np.exp(np.asarray(predictions, dtype=np.float64) * scale + mean)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("paper neural predictions overflowed or underflowed in original units")
    return values


def _ddl_chunks(ddl, encode, limit):
    """Greedily pack the canonical one-statement-per-line representation.

    Delayed cyclic constraints and generated trigger statements remain intact.
    A single overlong statement is an explicit error: the manuscript specifies
    splitting between tables and does not describe truncation or splitting a table.
    """
    if not isinstance(ddl, str) or not ddl.strip():
        raise ValueError("the text views require nonempty record['paper_ddl'] from Stage B")
    chunks, current, current_ids = [], "", None
    for line in ddl.splitlines():
        if not line.strip():
            continue
        proposed = f"{current}\n{line}" if current else line
        ids = encode(proposed)
        if len(ids) <= limit:
            current, current_ids = proposed, ids
            continue
        if current_ids is not None:
            chunks.append(current_ids)
        current, current_ids = line, encode(line)
        if len(current_ids) > limit:
            raise ValueError(
                f"A canonical DDL statement requires {len(current_ids)} tokens; this encoder "
                f"allows {limit}. Table-boundary chunking cannot split this statement."
            )
    if current_ids is not None:
        chunks.append(current_ids)
    return chunks


def _paper_environment():
    environment = _environment()
    for name in ("accelerate", "bitsandbytes"):
        try:
            environment["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment["packages"][name] = None
    return environment


@lru_cache(maxsize=12)
def _fingerprint_checkpoint(path, file_stats):
    # The signature invalidates a process-local memo when files are replaced;
    # stored encoder identities and artifacts still use full SHA-256 digests.
    return _local_checkpoint_manifest(path)


def _checkpoint_manifest(path):
    suffixes = {".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"}
    signature = []
    for file in sorted(path.rglob("*")):
        if file.suffix in suffixes and ".git" not in file.parts and file.is_file():
            details = file.stat()
            signature.append((str(file.relative_to(path)), details.st_size, details.st_mtime_ns, details.st_ctime_ns))
    return _fingerprint_checkpoint(path, tuple(signature))


class _FrozenText:
    def __init__(self, family, path, batch_size, device, cache_dir):
        self.family, self.path, self.batch_size, self.device = family, path, batch_size, device
        source, _, self.dimension, self.limit = TEXT_SPECS[family]
        local_files = _checkpoint_manifest(path)
        if not local_files or not any(name.endswith((".safetensors", ".bin")) for name in local_files):
            raise FileNotFoundError(f"{family} has no checkpoint weight files in {path}")
        self.metadata = {
            "family": family, "expected_checkpoint": source,
            "local_files_sha256": local_files, "hidden_size": self.dimension,
            "token_limit": self.limit, "weight_loading": "bitsandbytes_fp4" if family == "llama" else "float32",
            "pooling": "mean of nonpadding final hidden states including special tokens; equal weight per chunk",
            "chunking": "canonical statement boundaries, reject an individual oversized statement",
        }
        if family == "llama":
            self.metadata["quantization_note"] = (
                "bitsandbytes FP4 weights with float16 compute; the manuscript "
                "calls its weights INT4 and does not state the historical 4-bit format"
            )
        self.identity = json.dumps({
            **self.metadata, "version": MODEL_VERSION, "device": str(device),
            "batch_size": batch_size, "packages": _paper_environment()["packages"],
        }, sort_keys=True).encode()
        self.cache_dir = Path(cache_dir).expanduser() / family if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectors = {}

    def _load(self, torch):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the project's neural extra to use local paper checkpoints.") from exc

        model_args = {
            "local_files_only": True, "trust_remote_code": False, "output_loading_info": True,
            "torch_dtype": torch.float16 if self.family == "llama" else torch.float32,
        }
        if self.family == "llama":
            try:
                import accelerate  # noqa: F401  # Transformers' device-map support.
                import bitsandbytes  # noqa: F401  # Required for verified 4-bit loading.
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError("The Llama 2 4-bit view requires the paper-int4 extra (accelerate and bitsandbytes) on a supported CUDA device.") from exc
            if self.device.type != "cuda":
                raise ValueError("The Llama 2 4-bit view requires neural.device='cuda' or 'cuda:N'.")
            declared = json.loads((self.path / "config.json").read_text(encoding="utf-8")).get("quantization_config")
            if declared:
                raise ValueError(
                    "Supply the original local Llama 2 checkpoint for on-load FP4 quantization; "
                    "pre-quantized weights may use a different format."
                )
            model_args.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map={"": str(self.device)},
            )

        try:
            model, diagnostics = AutoModel.from_pretrained(str(self.path), **model_args)
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.path), local_files_only=True, trust_remote_code=False, use_fast=True,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"Cannot load the local {self.family} checkpoint from {self.path}: {exc}"
            ) from exc
        cfg = model.config
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        allowed_types = {TEXT_SPECS[self.family][1]}
        if self.family == "codet5":
            allowed_types.add("codet5p")
        if cfg.model_type not in allowed_types or hidden != self.dimension:
            raise ValueError(f"{self.family} checkpoint has unexpected model type or hidden size")
        if self.family == "llama":
            if (getattr(cfg, "num_hidden_layers", None), getattr(cfg, "num_attention_heads", None)) != (32, 32):
                raise ValueError("The paper's Llama view expects a 7B Llama 2 configuration.")
            if not getattr(model, "is_loaded_in_4bit", False):
                raise ValueError("Llama did not load with 4-bit weights; refusing a different encoder.")

        def required(key):
            if self.family == "graphcodebert" and "pooler." in key:
                return False
            if self.family == "codet5" and key.startswith(("decoder.", "lm_head.")):
                return False
            return True

        missing = [key for key in diagnostics.get("missing_keys", ()) if required(key)]
        mismatched = [entry[0] for entry in diagnostics.get("mismatched_keys", ()) if required(entry[0])]
        if missing or mismatched or diagnostics.get("error_msgs"):
            raise ValueError(
                f"{self.family} has incomplete frozen weights: missing={missing}, "
                f"mismatched={mismatched}, errors={diagnostics.get('error_msgs', [])}"
            )
        if self.family == "codet5" and getattr(cfg, "is_encoder_decoder", False):
            if not hasattr(model, "get_encoder"):
                raise ValueError("CodeT5+ checkpoint must expose its encoder")
            model = model.get_encoder()
        elif getattr(cfg, "is_encoder_decoder", False):
            raise ValueError("Only the CodeT5+ view may use an encoder-decoder checkpoint.")
        if self.family != "llama":
            model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token_id is None:
            raise ValueError(f"{self.family} tokenizer needs a pad, eos, or unk token")
        tokenizer.padding_side = "right"
        capacity = _usable_token_limit(cfg, tokenizer, self.family)
        if capacity is not None and capacity < self.limit:
            raise ValueError(f"{self.family} checkpoint supports {capacity} tokens, below the paper limit {self.limit}")
        return model, tokenizer

    def encode(self, ddls, torch):
        if not ddls:
            return np.empty((0, self.dimension), dtype=np.float32)
        keys = []
        missing = {}
        for ddl in ddls:
            if not isinstance(ddl, str) or not ddl.strip():
                raise ValueError("each paper text record requires nonempty 'paper_ddl' from Stage B")
            key = hashlib.sha256(self.identity + b"\x00" + ddl.encode()).hexdigest()
            keys.append(key)
            if key in self.vectors or key in missing:
                continue
            cache_file = self.cache_dir / f"{key}.npy" if self.cache_dir else None
            if cache_file and cache_file.is_file():
                row = np.load(cache_file, allow_pickle=False)
                if row.dtype != np.float32 or row.shape != (self.dimension,) or not np.isfinite(row).all():
                    raise ValueError(f"invalid paper encoder embedding cache: {cache_file}")
                self.vectors[key] = row
            else:
                missing[key] = ddl
        if missing:
            model, tokenizer = self._load(torch)
            chunks = []
            owner = []
            for index, ddl in enumerate(missing.values()):
                parts = _ddl_chunks(
                    ddl, lambda value: tokenizer.encode(value, add_special_tokens=True, truncation=False, verbose=False),
                    self.limit,
                )
                chunks.extend(parts)
                owner.extend([index] * len(parts))
            sums = np.zeros((len(missing), self.dimension), dtype=np.float64)
            counts = np.zeros(len(missing), dtype=np.int64)
            with torch.inference_mode():
                for start in range(0, len(chunks), self.batch_size):
                    stop = start + self.batch_size
                    batch = tokenizer.pad(
                        {"input_ids": chunks[start:stop]}, padding=True, return_attention_mask=True, return_tensors="pt",
                    )
                    inputs = {key: value.to(self.device) for key, value in batch.items() if key in ("input_ids", "attention_mask")}
                    output = model(**inputs, return_dict=True, **({"use_cache": False} if self.family == "llama" else {}))
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (output.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1)
                    rows = pooled.cpu().numpy()
                    if rows.shape != (len(chunks[start:stop]), self.dimension) or not np.isfinite(rows).all():
                        raise ValueError(f"{self.family} produced invalid pooled hidden states")
                    np.add.at(sums, owner[start:stop], rows)
                    np.add.at(counts, owner[start:stop], 1)
            del model
            for key, row in zip(missing, (sums / counts[:, None]).astype(np.float32)):
                self.vectors[key] = row
                if self.cache_dir is not None:
                    with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npy", delete=False) as stream:
                        temporary = Path(stream.name)
                        np.save(stream, row, allow_pickle=False)
                    temporary.replace(self.cache_dir / f"{key}.npy")
        return np.stack([self.vectors[key] for key in keys])


def _graph_arrays(graph, vocabulary, relations):
    nodes = graph["nodes"]
    if not nodes:
        raise ValueError("a paper schema graph must contain at least one node")
    index = {node["id"]: i for i, node in enumerate(nodes)}
    if len(index) != len(nodes):
        raise ValueError("paper schema graph node IDs must be distinct")
    features = np.zeros((len(nodes), 7 + len(vocabulary)), dtype=np.float32)
    for i, node in enumerate(nodes):
        if node["type"] == "table":
            features[i, 0] = node["column_count"]
        elif node["type"] == "column":
            features[i, 1:6] = [node[flag] for flag in ("primary_key", "foreign_key", "nullable", "unique", "discriminator")]
            features[i, 6 + vocabulary.get(node["data_type"], len(vocabulary))] = 1
        else:
            raise ValueError(f"unknown paper schema graph node type: {node['type']!r}")
    by_relation = [[] for _ in relations]
    relation_ids = {name: i for i, name in enumerate(relations)}
    for edge in graph["edges"]:
        by_relation[relation_ids[edge["type"]]].append((index[edge["source"]], index[edge["target"]]))
    return features, tuple(np.asarray(edges, dtype=np.int64).reshape(-1, 2) for edges in by_relation)


def _pack_graphs(graphs, torch, device):
    counts = np.asarray([len(features) for features, _ in graphs], dtype=np.int64)
    offsets = np.cumsum(counts) - counts
    joined_edges = [
        np.concatenate([edges[relation] + offset for (_, edges), offset in zip(graphs, offsets)])
        for relation in range(len(graphs[0][1]))
    ]
    return (
        torch.as_tensor(np.concatenate([features for features, _ in graphs]), device=device),
        tuple(torch.as_tensor(edge.T, device=device) for edge in joined_edges),
        torch.as_tensor(np.repeat(np.arange(len(graphs)), counts), device=device),
        torch.as_tensor(counts, device=device),
    )


def _network(torch, config, views, graph_dim, relation_count):
    nn = torch.nn
    width, graph_width, bases, dropout = 512, 256, 4, 0.2

    class RelationalLayer(nn.Module):
        def __init__(self, source_dim):
            super().__init__()
            self.bases = nn.Parameter(torch.empty(bases, source_dim, graph_width))
            self.coefficients = nn.Parameter(torch.empty(relation_count, bases))
            self.self_transform = nn.Linear(source_dim, graph_width)
            with torch.no_grad():
                for index in range(bases):
                    nn.init.xavier_uniform_(self.bases[index])
            nn.init.xavier_uniform_(self.coefficients)

        def forward(self, nodes, edges):
            weights = torch.einsum("rb,bio->rio", self.coefficients, self.bases)
            updated = self.self_transform(nodes)
            for relation, (source, target) in enumerate(edges):
                if not source.numel():
                    continue
                messages = nodes[source] @ weights[relation]
                aggregate = updated.new_zeros(updated.shape).index_add_(0, target, messages)
                degree = torch.bincount(target, minlength=nodes.shape[0]).clamp_min(1).unsqueeze(1)
                updated = updated + aggregate / degree
            return updated

    class GraphEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([RelationalLayer(graph_dim), RelationalLayer(graph_width)])
            self.dropout = nn.Dropout(dropout)

        def forward(self, graph):
            nodes, edges, groups, counts = graph
            for layer in self.layers:
                nodes = self.dropout(torch.relu(layer(nodes, edges)))
            shape = (len(counts), graph_width)
            mean = nodes.new_zeros(shape).index_add_(0, groups, nodes) / counts.unsqueeze(1)
            maximum = nodes.new_full(shape, -float("inf")).scatter_reduce_(
                0, groups.unsqueeze(1).expand_as(nodes), nodes, reduce="amax", include_self=False,
            )
            return torch.cat((mean, maximum), dim=1)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            dimensions = {name: TEXT_SPECS[name][2] for name in views if name in TEXT_VIEWS}
            if "graph" in views:
                self.graph = GraphEncoder()
                dimensions["graph"] = 2 * graph_width
            if "symbolic" in views:
                self.symbolic = nn.Sequential(
                    nn.Linear(12, config.symbolic_hidden), nn.ReLU(),
                    nn.Linear(config.symbolic_hidden, config.symbolic_hidden), nn.ReLU(),
                )
                dimensions["symbolic"] = config.symbolic_hidden
            self.projections = nn.ModuleDict({view: nn.Linear(dimensions[view], width) for view in views})
            if config.fusion == "gated":
                self.gate_transform = nn.Linear(width, config.gate_hidden, bias=False)
                self.gate_score = nn.Linear(config.gate_hidden, 1, bias=False)
            fused_width = width if config.fusion == "gated" else width * len(views)
            self.normalize = nn.LayerNorm(fused_width)
            self.trunk = nn.Sequential(
                nn.Linear(fused_width, width), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            )
            self.heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(3))

        def forward(self, inputs):
            projected = []
            for view in views:
                vector = inputs[view]
                if view == "graph":
                    vector = self.graph(vector)
                elif view == "symbolic":
                    vector = self.symbolic(vector)
                projected.append(torch.relu(self.projections[view](vector)))
            if config.fusion == "gated":
                stacked = torch.stack(projected, dim=1)
                weights = torch.softmax(self.gate_score(torch.tanh(self.gate_transform(stacked))), dim=1)
                fused = (weights * stacked).sum(dim=1)
            else:
                fused = torch.cat(projected, dim=1)
            trunk = self.trunk(self.normalize(fused))
            return torch.cat([head(trunk) for head in self.heads], dim=1)

    return Model()


def _device(torch, config, views):
    device = torch.device(config.device)
    if "llama" in views and device.type != "cuda":
        raise ValueError("The paper Llama 2 4-bit view requires neural.device='cuda' or 'cuda:N'.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device is available for the paper neural model. "
                "For a graph/descriptor-only smoke run, explicitly select those views and set neural.device='cpu'."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"requested CUDA device {device} is unavailable")
    return device


def _check_protocol(config, seeds, views):
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("paper neural seeds must be distinct integers in [0, 2**32)")
    if views is None:
        if (config.epochs, config.batch_size, config.patience, config.fusion) != (50, 64, 10, "gated"):
            raise ValueError(
                "the full paper model uses 50 maximum epochs, batch size 64, patience 10, and gated fusion. "
                "Pass explicit views for smoke or ablation settings."
            )
        if seeds[0] not in (0, 5, 10) or seeds != tuple(range(seeds[0], seeds[0] + 5)):
            raise ValueError("a full paper repetition uses seeds 0–4, 5–9, or 10–14")
    return seeds


class PaperEnsemble:
    """A fitted fold ensemble for later Stage D prediction in measurement units."""

    def __init__(self, config, views, seeds, device, torch, checkpoint_specs):
        from .paper_schema import PAPER_FEATURES, PAPER_RELATIONS
        self.config, self.views, self.seeds = config, views, seeds
        self.torch, self.device = torch, device
        self.relations, self.descriptor_names = tuple(PAPER_RELATIONS), tuple(PAPER_FEATURES)
        self.members = []
        self.encoders = {
            family: _FrozenText(family, path, batch, device, config.cache_dir)
            for family, (path, batch) in checkpoint_specs.items()
        }
        self.graph_vocabulary = {}
        self.descriptor_mean = self.descriptor_scale = None
        self.target_mean = self.target_scale = None
        self.metadata = {}

    def _inputs(self, records, *, train_size=None):
        from .paper_schema import paper_features, paper_graph

        inputs = {}
        if "symbolic" in self.views:
            descriptor_rows = [paper_features(record) for record in records]
            descriptors = np.asarray([
                [features[name] for name in self.descriptor_names] for features in descriptor_rows
            ], dtype=np.float64).reshape(len(records), len(self.descriptor_names))
            if descriptors.shape[1] != 12 or not np.isfinite(descriptors).all():
                raise ValueError("paper model requires exactly 12 finite symbolic descriptors")
            if train_size is not None:
                self.descriptor_mean, self.descriptor_scale = _standardizer(descriptors[:train_size])
            inputs["symbolic"] = ((descriptors - self.descriptor_mean) / self.descriptor_scale).astype(np.float32)
        if "graph" in self.views:
            graphs = [paper_graph(record) for record in records]
            if train_size is not None:
                types = {node["data_type"] for graph in graphs[:train_size] for node in graph["nodes"] if node["type"] == "column"}
                self.graph_vocabulary = {value: index for index, value in enumerate(sorted(types))}
            inputs["graph"] = [_graph_arrays(graph, self.graph_vocabulary, self.relations) for graph in graphs]
        for family, encoder in self.encoders.items():
            inputs[family] = encoder.encode([record.get("paper_ddl") for record in records], self.torch)
        return inputs

    def _batch(self, inputs, indices):
        return {
            view: (
                _pack_graphs([value[i] for i in indices], self.torch, self.device)
                if view == "graph" else self.torch.as_tensor(value[indices], device=self.device)
            )
            for view, value in inputs.items()
        }

    def _new_member(self):
        return _network(self.torch, self.config, self.views, 7 + len(self.graph_vocabulary), len(self.relations)).to(self.device)

    def _predict(self, member, inputs, indices):
        if not len(indices):
            return np.empty((0, 3), dtype=np.float32)
        member.eval()
        with self.torch.inference_mode():
            return np.concatenate([
                member(self._batch(inputs, indices[start:start + self.config.batch_size])).cpu().numpy()
                for start in range(0, len(indices), self.config.batch_size)
            ])

    def _predict_members(self, inputs, indices):
        standardized = np.stack([self._predict(member, inputs, indices) for member in self.members])
        return _invert_targets(standardized, self.target_mean, self.target_scale)

    def predict_members(self, records):
        """Return (members, candidates, 3), ordered as insert, query, storage."""
        if not self.members:
            raise RuntimeError("fit or load a paper ensemble before prediction")
        records = list(records)
        return self._predict_members(self._inputs(records), np.arange(len(records)))

    def save(self, directory):
        """Write a new ensemble artifact using JSON and non-pickle NumPy arrays."""
        if not self.members:
            raise RuntimeError("fit a paper ensemble before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for index, member in enumerate(self.members):
            state = {name: value.detach().cpu().numpy() for name, value in member.state_dict().items()}
            np.savez_compressed(directory / f"member-{index}.npz", **state)
        document = {
            "config": asdict(self.config),
            "metadata": self.metadata,
            "graph_vocabulary": self.graph_vocabulary,
            "descriptor_names": self.descriptor_names,
            "relations": self.relations,
        }
        (directory / "manifest.json").write_text(json.dumps(document, indent=2, default=str) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory, *, device=None, checkpoints=None, cache_dir=None):
        """Read a saved ensemble and verify that any supplied text weights match."""
        directory = Path(directory)
        document = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        metadata = document["metadata"]
        if metadata.get("model_version") != MODEL_VERSION:
            raise ValueError("the saved paper ensemble has an unsupported model version")
        raw_config = document["config"]
        for key, value in (("device", device), ("checkpoints", checkpoints), ("cache_dir", cache_dir)):
            if value is not None:
                raw_config[key] = value
        config = PaperModelConfig.from_mapping({"neural": raw_config})
        views = _paper_views(metadata["views"])
        seeds = _check_protocol(config, metadata["seeds"], views)
        checkpoints = _checkpoint_specs(config, views)
        torch = _torch()
        result = cls(config, views, seeds, _device(torch, config, views), torch, checkpoints)
        if tuple(document["relations"]) != result.relations or tuple(document["descriptor_names"]) != result.descriptor_names:
            raise ValueError("the saved paper ensemble uses a different Stage B representation")
        for name, encoder in result.encoders.items():
            if encoder.metadata != metadata["text_encoders"].get(name):
                raise ValueError(f"the local {name} checkpoint differs from the trained paper ensemble")
        result.graph_vocabulary = document["graph_vocabulary"]
        stats = metadata["training_fold_statistics"]
        result.target_mean = np.asarray(stats["log_target_mean"], dtype=np.float64)
        result.target_scale = np.asarray(stats["log_target_scale"], dtype=np.float64)
        if stats["descriptor_mean"] is not None:
            result.descriptor_mean = np.asarray(stats["descriptor_mean"], dtype=np.float64)
            result.descriptor_scale = np.asarray(stats["descriptor_scale"], dtype=np.float64)
        for index in range(len(seeds)):
            with np.load(directory / f"member-{index}.npz", allow_pickle=False) as arrays:
                member = result._new_member()
                member.load_state_dict({name: torch.from_numpy(arrays[name]) for name in arrays.files}, strict=True)
            result.members.append(member.eval())
        result.metadata = metadata
        return result


def _fit(train_records, train_y, val_records, val_y, test_records, config, seeds, views):
    config = PaperModelConfig.from_mapping(config)
    seeds = _check_protocol(config, seeds, views)
    selected = _paper_views(views)
    checkpoint_specs = _checkpoint_specs(config, selected)
    train_records, val_records, test_records = list(train_records), list(val_records), list(test_records)
    _check_system_splits(train_records, val_records, test_records)
    train_count = len(train_records)
    val_stop = train_count + len(val_records)
    train_y = _target_array(train_y, train_count, "training")
    val_y = _target_array(val_y, len(val_records), "validation")

    torch = _torch()
    device = _device(torch, config, selected)
    ensemble = PaperEnsemble(config, selected, seeds, device, torch, checkpoint_specs)
    ensemble.target_mean, ensemble.target_scale = _standardizer(np.log(train_y))
    inputs = ensemble._inputs([*train_records, *val_records, *test_records], train_size=train_count)
    labels = ((np.log(np.concatenate((train_y, val_y))) - ensemble.target_mean) / ensemble.target_scale).astype(np.float32)
    labels_torch = torch.as_tensor(labels, device=device)
    val_indices = np.arange(train_count, val_stop)
    test_indices = np.arange(val_stop, val_stop + len(test_records))
    # Sum over examples and objectives as in the manuscript loss. The validation
    # value is divided by row count for comparable reporting; the argmin is unchanged.
    criterion = torch.nn.HuberLoss(reduction="sum", delta=1.0)
    audits = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        member = ensemble._new_member()
        optimizer = torch.optim.Adam(member.parameters(), lr=1e-4, betas=(0.9, 0.999))
        best_state, best_loss, best_epoch = None, float("inf"), 0
        for epoch in range(1, config.epochs + 1):
            member.train()
            order = rng.permutation(train_count)
            for start in range(0, train_count, config.batch_size):
                indices = order[start:start + config.batch_size]
                optimizer.zero_grad(set_to_none=True)
                prediction = member(ensemble._batch(inputs, indices))
                loss = criterion(prediction, labels_torch[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite paper neural training loss for seed {seed} at epoch {epoch}")
                loss.backward()
                optimizer.step()
            predictions = ensemble._predict(member, inputs, val_indices)
            validation_loss = criterion(torch.from_numpy(predictions), torch.from_numpy(labels[val_indices])).item() / len(val_indices)
            if not np.isfinite(validation_loss):
                raise RuntimeError(f"nonfinite paper neural validation loss for seed {seed} at epoch {epoch}")
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in member.state_dict().items()}
            elif epoch - best_epoch >= config.patience:
                break
        member.load_state_dict(best_state)
        ensemble.members.append(member.eval())
        audits.append({"seed": seed, "epochs": epoch, "best_epoch": best_epoch, "validation_huber_sum_per_row": best_loss})
    ensemble.metadata = {
        "model_version": MODEL_VERSION, "full_paper_configuration": views is None,
        "views": list(selected), "seeds": list(seeds), "fusion": config.fusion,
        "architecture": {
            "projection_width": 512, "trunk": [512, 512], "dropout": 0.2, "heads": 3,
            "graph_width": 256, "graph_layers": 2, "graph_bases": 4,
            "graph_relations": list(ensemble.relations), "graph_readout": "mean_and_max",
            "graph_normalization": "mean over incoming edges by relation, learned self transform, no added reverse edges",
            "graph_data_types_train_vocabulary": ensemble.graph_vocabulary,
            "symbolic_mlp": [config.symbolic_hidden, config.symbolic_hidden],
            "gate_hidden": config.gate_hidden,
        },
        "training": {
            "optimizer": "Adam", "learning_rate": 1e-4, "betas": [0.9, 0.999],
            "batch_size": config.batch_size, "patience": config.patience, "max_epochs": config.epochs,
            "loss": "summed Huber, delta=1, standardized natural-log targets", "members": audits,
        },
        "text_encoders": {name: encoder.metadata for name, encoder in ensemble.encoders.items()},
        "training_fold_statistics": {
            "log_target_mean": ensemble.target_mean.tolist(), "log_target_scale": ensemble.target_scale.tolist(),
            "descriptor_mean": ensemble.descriptor_mean.tolist() if ensemble.descriptor_mean is not None else None,
            "descriptor_scale": ensemble.descriptor_scale.tolist() if ensemble.descriptor_scale is not None else None,
        },
        "device": str(device), "environment": _paper_environment(),
    }
    return ensemble, inputs, val_indices, test_indices


def fit_paper_ensemble(train_records, train_y, val_records, val_y, config, seeds, *, views=None):
    """Fit a fold on its training systems and select member epochs on validation."""
    ensemble, _, _, _ = _fit(train_records, train_y, val_records, val_y, (), config, seeds, views)
    return ensemble


def fit_predict_paper(train_records, train_y, val_records, val_y, test_records, config, seeds, *, views=None):
    """Return (test_members, validation_members, metadata) in original units.

    Inputs are aligned Stage B records and positive measurements. Passing
    `views=None` requires the complete manuscript architecture and five seeds;
    an explicit view tuple identifies a smoke run or ablation.
    """
    ensemble, inputs, val_indices, test_indices = _fit(
        train_records, train_y, val_records, val_y, test_records, config, seeds, views,
    )
    return ensemble._predict_members(inputs, test_indices), ensemble._predict_members(inputs, val_indices), copy.deepcopy(ensemble.metadata)
