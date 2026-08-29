#!/usr/bin/env python
"""Validate restricted workbooks and write a text-free audit report."""

from __future__ import annotations

import argparse
from collections import Counter

from suicide_risk_detection.config import load_config, resolve_path, sha256_file, write_json
from suicide_risk_detection.constants import RISK_LABELS
from suicide_risk_detection.data import alignment_rate, load_workbook, validate_chronology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subtask1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    project = config["project"]
    train_path = resolve_path(project["train_path"])
    test_path = resolve_path(project["test_path"])
    output_dir = resolve_path(project["output_dir"])

    train = load_workbook(train_path, labeled=True)
    test = load_workbook(test_path, labeled=False)
    train_users = set(train["anon_user_id"].astype(str))
    test_users = set(test["anon_user_id"].astype(str))
    report = {
        "train_file": train_path.name,
        "train_sha256": sha256_file(train_path),
        "test_file": test_path.name,
        "test_sha256": sha256_file(test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_users": len(train_users),
        "test_users": len(test_users),
        "user_overlap": len(train_users & test_users),
        "risk_counts": {label: Counter(train["risk_level"])[label] for label in RISK_LABELS},
        "missing_gold_evidence": int(
            train["evidence for suicide risk level"].isna().sum()
        ),
        "train_chronology_issue_count": len(validate_chronology(train)),
        "test_chronology_issue_count": len(validate_chronology(test)),
    }
    for column in ("post_frases", "post_segmentos", "post_segmentosV1"):
        if column in train:
            aligned, total = alignment_rate(train["post"], train[column])
            report[f"{column}_units"] = total
            report[f"{column}_literal_alignment_rate"] = aligned / total if total else None

    if report["user_overlap"]:
        raise ValueError("Train/test users overlap; stop and audit the source files")
    if sum(report["risk_counts"].values()) != len(train):
        raise ValueError("Canonical labels do not cover all training rows")
    write_json(output_dir / "data_validation.json", report)
    print(f"Validation passed. Text-free report: {output_dir / 'data_validation.json'}")


if __name__ == "__main__":
    main()

