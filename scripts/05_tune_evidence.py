#!/usr/bin/env python
"""Tune anchor-guided evidence retrieval against OOF risk predictions."""

from __future__ import annotations

import argparse

import pandas as pd

from suicide_risk_detection.anchors import load_risk_anchors
from suicide_risk_detection.config import load_config, resolve_path, seed_everything, write_json
from suicide_risk_detection.data import load_workbook
from suicide_risk_detection.embedding import MPNetEncoder
from suicide_risk_detection.evidence import (
    EvidenceRetriever,
    select_scored_spans,
    tune_evidence_parameters,
    validate_verbatim,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subtask1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    project, evidence_config = config["project"], config["evidence"]
    seed_everything(int(project["seed"]))
    output_dir = resolve_path(project["output_dir"])
    train = load_workbook(resolve_path(project["train_path"]), labeled=True)
    oof = pd.read_csv(output_dir / "oof_risk_predictions.csv", dtype={"row_id": str})
    merged = train.merge(
        oof[["row_id", "risk_level_predicted"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["risk_level_predicted"].isna().any():
        raise ValueError("OOF risk predictions do not cover every training row")
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
    scored_rows = retriever.score_many(
        merged["post"].tolist(),
        merged["risk_level_predicted"].tolist(),
        show_progress=True,
    )
    best, search = tune_evidence_parameters(
        scored_rows,
        merged["gold_evidence"].tolist(),
        top_k_values=evidence_config["top_k_values"],
        min_similarity_values=evidence_config["min_similarity_values"],
        max_tokens_values=evidence_config["max_tokens_values"],
    )
    predictions = []
    for post, scored in zip(merged["post"], scored_rows, strict=True):
        spans = [
            item.text
            for item in select_scored_spans(
                scored,
                top_k=int(best["top_k"]),
                min_similarity=float(best["min_similarity"]),
                max_tokens=int(best["max_tokens"]),
            )
        ]
        validate_verbatim(post, spans)
        predictions.append(";".join(spans))
    pd.DataFrame({"row_id": merged["row_id"], "evidence": predictions}).to_csv(
        output_dir / "oof_evidence_predictions.csv", index=False
    )
    write_json(
        output_dir / "evidence_tuning.json",
        {
            "status": "exploratory tuning on repeated-grouped OOF risk predictions",
            "best": best,
            "grid": search,
        },
    )
    print(f"Best OOF Phrase F1: {float(best['phrase_f1']):.4f}")


if __name__ == "__main__":
    main()
