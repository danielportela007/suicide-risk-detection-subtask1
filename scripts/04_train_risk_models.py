#!/usr/bin/env python
"""Compare prompting/fusion configurations and fit the final Subtask 1 risk model."""

from __future__ import annotations

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from suicide_risk_detection.config import load_config, resolve_path, seed_everything, write_json
from suicide_risk_detection.constants import ID_TO_RISK, RISK_LABELS
from suicide_risk_detection.features import concatenate_blocks, load_feature_bundle
from suicide_risk_detection.modeling import (
    build_pipeline,
    configurations,
    majority_vote,
    repeated_stratified_group_splits,
    risk_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subtask1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    project, validation = config["project"], config["validation"]
    seed = int(project["seed"])
    seed_everything(seed)
    output_dir = resolve_path(project["output_dir"])
    model_dir = resolve_path(project["model_dir"])
    bundle = load_feature_bundle(output_dir / "train_features.npz")
    if bundle.labels is None:
        raise ValueError("Training feature archive has no labels")
    y = bundle.labels
    splits = list(
        repeated_stratified_group_splits(
            y,
            bundle.user_ids,
            n_splits=int(validation["n_splits"]),
            n_repeats=int(validation["n_repeats"]),
            random_state=int(validation["random_state"]),
        )
    )
    candidate_configs = configurations(
        config,
        list(bundle.blocks),
        {name: matrix.shape[1] for name, matrix in bundle.blocks.items()},
    )
    matrices: dict[tuple[str, ...], tuple[np.ndarray, list[str]]] = {}
    rows: list[dict] = []
    for number, candidate in enumerate(candidate_configs, start=1):
        if candidate.blocks not in matrices:
            matrices[candidate.blocks] = concatenate_blocks(bundle, list(candidate.blocks))
        X, _ = matrices[candidate.blocks]
        for repeat, fold, train_indices, validation_indices in splits:
            pipeline = build_pipeline(candidate, X.shape[1], seed + repeat)
            pipeline.fit(X[train_indices], y[train_indices])
            predicted = pipeline.predict(X[validation_indices])
            rows.append(
                {
                    "configuration_id": candidate.identifier,
                    "experiment": candidate.experiment,
                    "blocks": ";".join(candidate.blocks),
                    "classifier": candidate.classifier,
                    "C": candidate.c,
                    "k": candidate.k,
                    "repeat": repeat,
                    "fold": fold,
                    **risk_metrics(y[validation_indices], predicted),
                }
            )
        if number % 20 == 0 or number == len(candidate_configs):
            print(f"Evaluated {number}/{len(candidate_configs)} configurations", flush=True)

    detailed = pd.DataFrame(rows)
    detailed.to_csv(output_dir / "risk_cv_fold_metrics.csv", index=False)
    summary = detailed.groupby(
        ["configuration_id", "experiment", "blocks", "classifier", "C", "k"],
        dropna=False,
        as_index=False,
        sort=False,
    ).agg(
        risk_weighted_f1__mean=("risk_weighted_f1", "mean"),
        risk_weighted_f1__std=("risk_weighted_f1", "std"),
        risk_macro_f1__mean=("risk_macro_f1", "mean"),
        risk_macro_f1__std=("risk_macro_f1", "std"),
        accuracy__mean=("accuracy", "mean"),
        accuracy__std=("accuracy", "std"),
    )
    summary = summary.sort_values(
        ["risk_weighted_f1__mean", "risk_macro_f1__mean"], ascending=False
    )
    summary.to_csv(output_dir / "risk_cv_summary.csv", index=False)
    best_id = str(summary.iloc[0]["configuration_id"])
    lookup = {candidate.identifier: candidate for candidate in candidate_configs}
    best = lookup[best_id]
    X, feature_names = matrices[best.blocks]

    votes: list[list[int]] = [[] for _ in range(len(y))]
    for repeat, fold, train_indices, validation_indices in splits:
        pipeline = build_pipeline(best, X.shape[1], seed + repeat)
        pipeline.fit(X[train_indices], y[train_indices])
        predicted = pipeline.predict(X[validation_indices])
        for index, label in zip(validation_indices, predicted, strict=True):
            votes[int(index)].append(int(label))
    if any(len(row) != int(validation["n_repeats"]) for row in votes):
        raise RuntimeError("Each row must receive one OOF prediction per repeat")
    oof = majority_vote(votes)
    pd.DataFrame(
        {
            "row_id": bundle.row_ids,
            "risk_level_gold": [ID_TO_RISK[int(value)] for value in y],
            "risk_level_predicted": [ID_TO_RISK[int(value)] for value in oof],
        }
    ).to_csv(output_dir / "oof_risk_predictions.csv", index=False)

    final_pipeline = build_pipeline(best, X.shape[1], seed)
    final_pipeline.fit(X, y)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "blocks": list(best.blocks),
            "feature_names": feature_names,
            "configuration": best.__dict__,
            "risk_labels": RISK_LABELS,
        },
        model_dir / "risk_model.joblib",
    )
    report = {
        "selection_status": "exploratory repeated grouped cross-validation",
        "selection_metric": "risk_weighted_f1",
        "best_configuration": best.__dict__,
        "oof_aggregate_metrics": risk_metrics(y, oof),
        "per_class": classification_report(
            y,
            oof,
            labels=list(range(len(RISK_LABELS))),
            target_names=RISK_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, oof).tolist(),
    }
    write_json(output_dir / "risk_oof_report.json", report)
    print(f"Best configuration: {best.identifier}")
    print(f"OOF Weighted F1: {report['oof_aggregate_metrics']['risk_weighted_f1']:.4f}")
    print(f"Final model: {model_dir / 'risk_model.joblib'}")


if __name__ == "__main__":
    main()
