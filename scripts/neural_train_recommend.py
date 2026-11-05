"""Optional final neural model deployment, after held-out evaluation.

This script uses actual checkpoint weights. It never replaces them with hashes.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ymap_research.common import TARGETS, aligned_training_data, read_jsonl, validate_output_schema_ids, write_json
from ymap_research.neural import EncoderConfig, NeuralConfig, NeuralMultiViewRegressor
from ymap_research.recommendation import fit_normalization, recommend


def validate_output_candidates(records):
    """Reject ambiguous or unsafe DDL filenames before loading any model."""
    validate_output_schema_ids(records)
    for row in records:
        if not isinstance(row.get("object_model"), str) or not row["object_model"].strip():
            raise ValueError("Candidate object_model must be a nonempty string")
    if len({row["object_model"] for row in records}) != 1:
        raise ValueError("Provide one source model per recommendation")


def validate_target_order(target_names):
    if list(target_names) != TARGETS:
        raise ValueError(f"This wrapper requires target_names in measurement-column order: {TARGETS}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--candidates", required=True)
    train.add_argument("--measurements", required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--out", required=True)
    predict = sub.add_parser("recommend")
    predict.add_argument("--candidates", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--risk-lambda", type=float, default=0.)
    args = parser.parse_args()
    if args.command == "train":
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        mode = config.get("mode", "research")
        records, _, targets, source = aligned_training_data(args.candidates, args.measurements, mode)
        if mode != "research":
            raise ValueError("Final neural deployment script accepts research measurements only")
        values = dict(config.get("neural", {}))
        values["text_encoders"] = tuple(EncoderConfig(**value) for value in values.get("text_encoders", []))
        values["ensemble_size"] = int(config.get("ensemble_size", 5))
        values["seed"] = int(config.get("seed", 42))
        neural_config = NeuralConfig(**values)
        validate_target_order(neural_config.target_names)
        model = NeuralMultiViewRegressor(neural_config).fit(records, targets)
        model.save(args.out)
        write_json(Path(args.out) / "deployment_provenance.json", {"normalization": fit_normalization(targets), "training_provenance": source,
            "evaluation_note": "Final deployment fit on all supplied data; held-out evaluation is a separate command."})
        print(f"Saved actual neural model to {args.out}")
    else:
        records = read_jsonl(args.candidates)
        validate_output_candidates(records)
        # Load only trusted locally trained artifacts, never arbitrary downloaded pickle files.
        model = NeuralMultiViewRegressor.load(args.model, trusted=True)
        validate_target_order(model.config.target_names)
        provenance = json.loads((Path(args.model) / "deployment_provenance.json").read_text(encoding="utf-8"))
        members = model.predict_members(records)
        mean = members.mean(axis=0)
        std = members.std(axis=0, ddof=1) if len(members) > 1 else np.zeros_like(mean)
        result = recommend(mean, std, provenance["normalization"], risk_lambda=args.risk_lambda)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        data = {"schema_id": [row["schema_id"] for row in records]}
        for j, target in enumerate(TARGETS):
            data[f"pred_{target}"] = mean[:, j]
            data[f"std_{target}"] = std[:, j]
        selected = set(result["indices"])
        data["selected"] = [i in selected for i in range(len(records))]
        pd.DataFrame(data).to_csv(out / "neural_predictions.csv", index=False)
        write_json(out / "selection.json", result)
        for i in result["indices"]:
            (out / f"{records[i]['schema_id']}.sql").write_text(records[i]["ddl"], encoding="utf-8")
        print(f"Saved predictions without reading measurements to {out}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"Y-Map neural: {error}", file=sys.stderr)
        raise SystemExit(2)
