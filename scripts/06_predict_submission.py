#!/usr/bin/env python
"""Predict the official test set and create a validated Subtask 1 CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from suicide_risk_detection.anchors import load_risk_anchors
from suicide_risk_detection.config import load_config, resolve_path, seed_everything
from suicide_risk_detection.constants import ID_TO_RISK, RISK_LABELS
from suicide_risk_detection.data import load_workbook
from suicide_risk_detection.embedding import MPNetEncoder
from suicide_risk_detection.evidence import (
    EvidenceRetriever,
    select_scored_spans,
    validate_verbatim,
)
from suicide_risk_detection.features import concatenate_blocks, load_feature_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subtask1.yaml")
    parser.add_argument("--team-name", required=True)
    args = parser.parse_args()
    if Path(args.team_name).name != args.team_name or not args.team_name.strip():
        raise ValueError("--team-name must be a filename-safe team name, not a path")
    config = load_config(args.config)
    project, evidence_config = config["project"], config["evidence"]
    seed_everything(int(project["seed"]))
    output_dir = resolve_path(project["output_dir"])
    model_payload = joblib.load(resolve_path(project["model_dir"]) / "risk_model.joblib")
    bundle = load_feature_bundle(output_dir / "test_features.npz")
    test = load_workbook(resolve_path(project["test_path"]), labeled=False)
    if bundle.row_ids.tolist() != test["row_id"].tolist():
        raise ValueError("Test feature archive is not aligned to the current test workbook")
    X, feature_names = concatenate_blocks(bundle, model_payload["blocks"])
    if feature_names != model_payload["feature_names"]:
        raise ValueError("Train/test feature names differ")
    predicted_ids = model_payload["pipeline"].predict(X)
    risk_levels = [ID_TO_RISK[int(value)] for value in predicted_ids]
    if not set(risk_levels).issubset(RISK_LABELS):
        raise ValueError("Model emitted a non-canonical risk label")

    tuning = json.loads((output_dir / "evidence_tuning.json").read_text(encoding="utf-8"))
    best = tuning["best"]
    anchors = load_risk_anchors(
        resolve_path(project["anchors_dir"]), config["features"]["prompting_strategies"]
    )
    evidence_embedding = {
        **config["embedding"],
        "batch_size": int(
            evidence_config.get("embedding_batch_size", config["embedding"]["batch_size"])
        ),
    }
    retriever = EvidenceRetriever(
        MPNetEncoder(**evidence_embedding),
        anchors,
        window_sizes=evidence_config["window_sizes"],
        candidate_max_tokens=int(evidence_config["candidate_max_tokens"]),
    )
    scored_rows = retriever.score_many(test["post"].tolist(), risk_levels, show_progress=True)
    evidence_predictions: list[str] = []
    for post, risk_level, scored in zip(test["post"], risk_levels, scored_rows, strict=True):
        if risk_level == "Indicator":
            evidence_predictions.append(str(evidence_config.get("indicator_output", "")))
            continue
        spans = [
            span.text
            for span in select_scored_spans(
                scored,
                top_k=int(best["top_k"]),
                min_similarity=float(best["min_similarity"]),
                max_tokens=int(best["max_tokens"]),
            )
        ]
        validate_verbatim(post, spans)
        evidence_predictions.append(";".join(spans))

    submission = pd.DataFrame(
        {
            "row_id": test["row_id"],
            "risk_level": risk_levels,
            "evidence": evidence_predictions,
            # The full official schema is retained; Subtask 2 is intentionally not predicted.
            "factors": "[]",
        }
    )
    if submission["row_id"].duplicated().any() or len(submission) != len(test):
        raise ValueError("Submission row coverage is invalid")
    destination = output_dir / f"{args.team_name}.csv"
    submission.to_csv(destination, index=False)
    print(f"Validated submission created: {destination}")


if __name__ == "__main__":
    main()
