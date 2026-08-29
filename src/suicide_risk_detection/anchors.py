"""Strict loading of canonical risk-level synthetic anchors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .constants import RISK_LABELS


@dataclass(frozen=True)
class AnchorSet:
    strategy: str
    prompt_version: str
    curation_version: str
    phrases: dict[str, list[str]]
    source_files: tuple[str, ...]


def load_risk_anchors(directory: Path, strategies: list[str]) -> dict[str, AnchorSet]:
    result: dict[str, AnchorSet] = {}
    for strategy in strategies:
        files = sorted(directory.glob(f"{strategy}__risk_levels.json"))
        if len(files) != 1:
            raise ValueError(f"Expected one risk anchor file for {strategy}, found {len(files)}")
        path = files[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        phrases: dict[str, list[str]] = {}
        for target in payload.get("targets", []):
            label = target.get("target_label")
            if label not in RISK_LABELS:
                raise ValueError(f"Unexpected risk target {label!r} in {path}")
            # Deliberately consume only the canonical phrase text field.
            phrases[label] = [item["text"] for item in target.get("phrases", [])]
        if tuple(phrases) != RISK_LABELS:
            raise ValueError(f"Risk targets in {path} must follow canonical order {RISK_LABELS}")
        if any(not items for items in phrases.values()):
            raise ValueError(f"Every risk target requires at least one phrase in {path}")
        result[strategy] = AnchorSet(
            strategy=strategy,
            prompt_version=str(payload.get("prompt_version", "unknown")),
            curation_version=str(payload.get("curation_version", "unknown")),
            phrases=phrases,
            source_files=(path.name,),
        )
    return result
