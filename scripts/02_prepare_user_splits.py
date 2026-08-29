#!/usr/bin/env python
"""Create deterministic repeated user-grouped validation manifests."""

from __future__ import annotations

import argparse

import numpy as np

from suicide_risk_detection.config import load_config, resolve_path, write_json
from suicide_risk_detection.constants import RISK_TO_ID
from suicide_risk_detection.data import load_workbook
from suicide_risk_detection.modeling import repeated_stratified_group_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subtask1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    validation = config["validation"]
    train = load_workbook(resolve_path(config["project"]["train_path"]), labeled=True)
    y = train["risk_level"].map(RISK_TO_ID).to_numpy(dtype=np.int64)
    groups = train["anon_user_id"].astype(str).to_numpy()
    row_ids = train["row_id"].astype(str).to_numpy()
    splits = []
    for repeat, fold, train_indices, validation_indices in repeated_stratified_group_splits(
        y,
        groups,
        n_splits=int(validation["n_splits"]),
        n_repeats=int(validation["n_repeats"]),
        random_state=int(validation["random_state"]),
    ):
        splits.append(
            {
                "repeat": repeat,
                "fold": fold,
                "train_row_ids": row_ids[train_indices].tolist(),
                "validation_row_ids": row_ids[validation_indices].tolist(),
                "train_users": int(len(set(groups[train_indices]))),
                "validation_users": int(len(set(groups[validation_indices]))),
            }
        )
    output = resolve_path(config["project"]["output_dir"]) / "split_manifest.json"
    write_json(output, {"validation": validation, "splits": splits})
    print(f"Created {len(splits)} leakage-checked splits: {output}")


if __name__ == "__main__":
    main()

