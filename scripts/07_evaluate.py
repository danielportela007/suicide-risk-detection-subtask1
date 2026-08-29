#!/usr/bin/env python
"""Report OOF risk and evidence components separately for Subtask 1."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from suicide_risk_detection.config import load_config, resolve_path, write_json
from suicide_risk_detection.constants import RISK_TO_ID
from suicide_risk_detection.data import load_workbook, parse_gold_evidence
from suicide_risk_detection.evidence import mean_phrase_f1, validate_verbatim
from suicide_risk_detection.modeling import risk_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/subtask1.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = resolve_path(config["project"]["output_dir"])
    train = load_workbook(resolve_path(config["project"]["train_path"]), labeled=True)
    risk = pd.read_csv(output_dir / "oof_risk_predictions.csv", dtype={"row_id": str})
    evidence = pd.read_csv(
        output_dir / "oof_evidence_predictions.csv", dtype={"row_id": str}, keep_default_na=False
    )
    merged = train.merge(risk, on="row_id", validate="one_to_one").merge(
        evidence, on="row_id", validate="one_to_one"
    )
    if len(merged) != len(train):
        raise ValueError("OOF artifacts do not cover the complete training workbook")
    predicted_evidence = [parse_gold_evidence(value) for value in merged["evidence"]]
    for post, spans in zip(merged["post"], predicted_evidence, strict=True):
        validate_verbatim(post, spans)
    y_true = merged["risk_level"].map(RISK_TO_ID).to_numpy(dtype=np.int64)
    y_pred = merged["risk_level_predicted"].map(RISK_TO_ID).to_numpy(dtype=np.int64)
    risk_result = risk_metrics(y_true, y_pred)
    evidence_f1 = mean_phrase_f1(predicted_evidence, merged["gold_evidence"].tolist())
    # Subtask 2 is deliberately outside this experiment. The submission emits
    # an empty factor set for every row, whose multilabel Macro F1 is zero for
    # the 24 supported factor labels. Keep this placeholder visible so the full
    # competition composite is never confused with the normalized Subtask 1 score.
    factor_macro_f1 = 0.0
    official_composite = (
        0.40 * risk_result["risk_weighted_f1"]
        + 0.30 * evidence_f1
        + 0.30 * factor_macro_f1
    )
    report = {
        "status": "exploratory OOF; not held-out test evidence",
        "configuration": config,
        "risk": risk_result,
        "evidence_phrase_f1": evidence_f1,
        "factor_macro_f1": factor_macro_f1,
        "factor_status": "empty-prediction placeholder; Subtask 2 is not modeled",
        "weighted_contributions": {
            "risk_0.40": 0.40 * risk_result["risk_weighted_f1"],
            "evidence_0.30": 0.30 * evidence_f1,
            "factors_0.30": 0.30 * factor_macro_f1,
        },
        "official_composite_0.40_0.30_0.30": official_composite,
        "subtask1_score_normalized_to_one": official_composite / 0.70,
        "metric_note": (
            "Local Phrase F1 treats empty prediction plus empty gold as 1.0; confirm the "
            "organizer implementation before final reporting."
        ),
    }
    write_json(output_dir / "subtask1_oof_report.json", report)
    print(f"Risk Weighted F1: {risk_result['risk_weighted_f1']:.4f}")
    print(f"Evidence Phrase F1: {evidence_f1:.4f}")
    print(f"Factor Macro F1 (empty placeholder): {factor_macro_f1:.4f}")
    print(f"Official 0.40/0.30/0.30 composite: {official_composite:.4f}")
    print(f"Normalized Subtask 1 score: {report['subtask1_score_normalized_to_one']:.4f}")


if __name__ == "__main__":
    main()
