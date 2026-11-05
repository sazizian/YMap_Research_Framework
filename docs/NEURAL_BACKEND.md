# Optional neural multi-view backend

This implementation is a new research implementation. It is not a reconstruction
of unavailable historical model weights, and it does not establish the submitted
paper's results. No manuscript performance number is produced or hard-coded here.

The original uploaded project includes a lightweight hash-token encoder. This
backend **never uses that encoder**. Checkpoint or dependency failures stop the
run. SHA256 is used only for cache/checkpoint identity and integrity.

## What is implemented

Each candidate supplies `schema_id`, its source `object_model` name, actual `ddl`,
numerical `features`, and the generated schema `graph`.

1. SQL is tokenized deterministically. Whitespace/comments and unquoted case are
   normalized; quoted identifiers and string contents are retained. This is
   lexical canonicalization, not proof of semantic equivalence or validation of
   arbitrary SQL. Unterminated strings/comments are rejected.
2. Each configured text checkpoint tokenizes canonical DDL with its real
   tokenizer and runs a frozen pretrained model in evaluation/inference mode.
   Attention-mask-weighted mean pooling includes nonpadding special tokens and
   excludes all padding. Pooling accumulates in float32.
3. A trainable graph encoder performs typed message passing on actual tables and
   columns with `contains`, `contained_by`, `foreign_key`, and `referenced_by`
   relations. It uses separate learned transforms per relation, incoming-degree
   normalization, self transforms, layer normalization, ReLU and dropout. Mean
   table and mean column states are concatenated and projected. Node features
   encode declared node/type/kind, column properties, SQL type categories and
   local edge counts; no names are assigned pseudo-random semantic vectors.
4. An MLP encodes the supplied numerical feature vector. Its vocabulary and
   standardization mean/standard deviation come from training candidates only.
5. Each enabled view has a trainable projection into `hidden_dim` dimensions.
   Scalar gates apply a per-candidate softmax across views. A regression MLP
   predicts all objectives from their weighted sum.
6. Independently initialized fusion/graph models are trained on independent
   bootstrap samples of **training** candidates. They share frozen text
   checkpoints. `predict_members` returns member predictions in raw target units;
   `predict` returns their mean. Ensemble disagreement is not automatically a
   calibrated uncertainty interval.

The default configuration enables only the graph and symbolic views. That is an
explicit **ablation**, not the full three-text-encoder model. To run the full
configuration, supply all three real checkpoints as below. `target_names` defaults
to `insert_ms`, `query_ms`, `storage_mb`, in that order.

## Checkpoints and optional dependencies

From the project root:

```bash
python -m pip install -e '.[neural]'
```

The neural extra needs PyTorch, Transformers, Safetensors and SentencePiece in
addition to the numerical core. Appropriate wheels depend on the machine and GPU.
The dependencies being installed does not supply pretrained weights.

Supply already authorized local checkpoints with their tokenizers. The loader
defaults to `local_files_only=True` and `trust_remote_code=False`. It never
requests a login or downloads weights implicitly. To intentionally allow Hub
resolution, set `local_files_only=False` and provide an immutable commit in
`revision`; the actual resolved commit is recorded. Local directories are
identified by SHA256 of model/tokenizer/configuration files. Reading these files
may take appreciable time for large checkpoints, but it prevents silently reusing
cached embeddings from changed local weights.

Compatible examples are a local copy of `microsoft/graphcodebert-base`, a
standard Transformers-compatible encoder-decoder CodeT5+ checkpoint such as
`Salesforce/codet5p-220m`, and a compatible LLaMA-family base-model checkpoint
obtained under its own access/license terms. A checkpoint requiring custom Hub
Python code is deliberately rejected; not every CodeT5+ distribution uses the
same architecture or loading API. The exact checkpoint identifier, local content
hashes or resolved commit, model configuration, tokenizer class and hidden
dimensions are recorded in the fitted artifact.

Loading diagnostics are checked: missing/mismatched parameters used by a frozen
encoder are rejected. Missing RoBERTa pooler weights or unused CodeT5 decoder
weights may be ignored because this implementation never uses them; their
diagnostics are still recorded. Model-family architecture checks also prevent
accidentally labeling, for example, a non-LLaMA architecture as the LLaMA view.

**Scope of the GraphCodeBERT view:** this code uses its standard pretrained
RoBERTa-style text interface on DDL. It does not construct the original
GraphCodeBERT code data-flow attention. The separate trainable schema graph view
represents relational structure. Any manuscript must describe that distinction.

Text is **right truncated** at `max_length` including special tokens. No unseen
tail is inferred and no chunk aggregation is silently performed. Each encoder
records the number of newly encoded, distinct cache misses exceeding the limit
in the current process. Audit schema lengths before choosing the limit; compare
limits only with validation groups. Positional limits declared by checkpoints
are enforced, including RoBERTa's nonpadding positional offset and meaningful
tokenizer limits. Hub tokenizers are pinned to the same resolved commit as their
model weights. Reloading a saved artifact uses its recorded resolved commit,
even if the originally requested Hub branch has subsequently changed. All text model weights currently load using their standard
Transformers floating-point loading behavior; large LLaMA checkpoints can require
substantial RAM/VRAM. Configure a feasible checkpoint and device for your hardware.

## Python interface

```python
from ymap_research.neural import (
    EncoderConfig, NeuralConfig, NeuralMultiViewRegressor,
)

configuration = NeuralConfig(
    text_encoders=(
        EncoderConfig("graphcodebert", "/models/graphcodebert-base", max_length=512),
        EncoderConfig("codet5", "/models/codet5p-220m", max_length=512),
        EncoderConfig("llama", "/models/approved-llama", max_length=1024),
    ),
    use_graph=True,
    use_symbolic=True,
    ensemble_size=5,
    seed=42,
    epochs=100,
    hidden_dim=128,
    graph_layers=2,
    device="cpu",
    cache_dir="runs/text-cache",
)

# Obtain train_candidates and y_train from measured data and an object-model
# group split. Do not take them from the held-out test object model.
model = NeuralMultiViewRegressor(configuration)
model.fit(train_candidates, y_train)
mean_predictions = model.predict(test_candidates)       # (candidates, 3)
member_predictions = model.predict_members(test_candidates)  # (5, candidates, 3)
model.save("runs/fitted-neural-model")

# Only for a model directory whose provenance you trust:
restored = NeuralMultiViewRegressor.load("runs/fitted-neural-model", trusted=True)
```

Frozen text embeddings are cached as nonpickled `.npy` vectors. Cache identity
includes canonical DDL, encoder family, checkpoint identity, tokenizer identity,
pooling, maximum length, representation version, actual execution device/dtype,
and relevant installed package versions. Unordered loading diagnostics do not
change checkpoint identity. The
graph is constructed from the candidate's JSON on each call; no historical graph
cache is reused. Numerical standardization and trainable projections belong to
the fitted fold and are not shared across held-out fits.

## Full evaluation configuration

Use the following fields in the evaluation JSON configuration accepted by the
evaluation CLI. Replace all checkpoint paths with real accessible directories.
The evaluator runs object-model-held-out folds; do not relabel these as a
schema-random split. `models:["neural"]` evaluates a single independently trained
model, while `ensemble_base:"neural"` selects the neural architecture for the
separate ensemble experiment. The evaluator supplies their member counts and
fold seeds explicitly.

```json
{
  "models": ["neural"],
  "ensemble_base": "neural",
  "ensemble_size": 5,
  "neural": {
    "text_encoders": [
      {
        "family": "graphcodebert",
        "model_name_or_path": "/models/graphcodebert-base",
        "local_files_only": true,
        "max_length": 512,
        "batch_size": 4
      },
      {
        "family": "codet5",
        "model_name_or_path": "/models/codet5p-220m",
        "local_files_only": true,
        "max_length": 512,
        "batch_size": 4
      },
      {
        "family": "llama",
        "model_name_or_path": "/models/approved-llama",
        "local_files_only": true,
        "max_length": 1024,
        "batch_size": 1
      }
    ],
    "use_graph": true,
    "use_symbolic": true,
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "hidden_dim": 128,
    "graph_layers": 2,
    "dropout": 0.1,
    "patience": 15,
    "weight_decay": 0.0001,
    "device": "cpu",
    "cache_dir": "runs/text-cache",
    "deterministic": true
  }
}
```

For a graph+symbolic ablation set `text_encoders: []`. For text-only or symbolic-
only ablations disable the other views explicitly. Freeze the ablation design
before evaluating the held-out outcomes. Never label a graph+symbolic result as
GraphCodeBERT/CodeT5+/LLaMA performance.

## Training, leakage and reproducibility

The loss is mean squared error across z-scored training targets. Target mean and
population standard deviation come only from the training rows; zero standard
deviations become one. Targets are not silently logged or clamped. Predictions
can be negative; any downstream objective-handling rule must be documented and
applied identically to compared methods.

`fit(..., validation_records=..., validation_y=...)` supports early stopping only
when validation object-model groups and canonical DDL do not overlap training.
When no validation set is supplied, it runs the fixed epoch budget. The evaluation
driver deliberately does not use its held-out calibration outcomes for fitting
or early stopping. Tune hyperparameters on separate development groups or
nested group folds, then freeze the configuration for the final evaluation.

Python, NumPy, CPU and CUDA seeds are set per member; deterministic PyTorch
algorithms are requested by default. Unsupported nondeterministic GPU operations
raise rather than silently weaken that request. Bitwise equality across devices,
library versions or operating systems is not promised. Artifacts contain actual
installed package versions, Python/platform details, device/CUDA version, training
and validation groups, candidate IDs, ensemble bootstrap indices and training
history. None of this substitutes for the raw measurement provenance.

`save` writes JSON plus tensor-only state files and checksums. `load` requires
`trusted=True`, verifies checksums and original frozen-checkpoint provenance,
and uses PyTorch's `weights_only=True` loader. Checksums detect changes, not the
authenticity of an untrusted supplier. Keep the original local checkpoints
available at their recorded paths; they are not duplicated into each ensemble
artifact. Loading never silently replaces a missing checkpoint with another one.

## Verification performed when this deliverable was created

The pure tokenizer, cache identity and schema graph tests ran successfully.
Configuration, explicit missing-dependency errors and group-leakage rejection
also ran successfully in the minimal environment. A separate verification
environment installed CPU PyTorch and Transformers, and executed actual neural
training/serialization and all three HF loading/encoding paths. The HF tests
create tiny **randomly initialized local** RoBERTa, T5 and LLaMA checkpoints with
real tokenizers; no Hub weights are downloaded. They check mask mean pooling,
truncation, cache hits, batched versus individual encoding, full five-view
ensemble training, missing-weight rejection, and saved-model prediction equality.
Additional tests check explicit tensor dtypes under a float64 global default,
finite scaling for large features, and repeatability under a fixed seed.

The verification environment used CPU PyTorch 2.14.1+cpu, Transformers 4.57.6,
Tokenizers 0.22.2 and Safetensors 0.8.0. These successful software integration
tests do **not** establish pretrained semantic quality, the manuscript's
predictive accuracy, or historical reproducibility. No full pretrained
GraphCodeBERT/CodeT5+/LLaMA experiment on measured research data was executed;
those experiments still require actual authorized checkpoints and measurements.

After installing dependencies, run:

```bash
python -m unittest discover -s tests -p 'test_neural.py' -v
python -m unittest discover -s tests -p 'test_representations.py' -v
```

These small tests check software behavior. They do not estimate predictive
accuracy, Pareto quality, calibration, speedup, or statistical significance.
