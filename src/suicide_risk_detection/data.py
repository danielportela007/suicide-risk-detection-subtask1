"""Restricted workbook loading and schema validation."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import (
    EMPTY_EVIDENCE_MARKERS,
    REQUIRED_TEST_COLUMNS,
    REQUIRED_TRAIN_COLUMNS,
    RISK_LABELS,
)


def canonicalize_risk_label(value: object) -> str:
    normalized = re.sub(r"\s+", " ", str(value).strip()).casefold()
    lookup = {label.casefold(): label for label in RISK_LABELS}
    if normalized not in lookup:
        raise ValueError(f"Unknown suicide-risk label: {value!r}")
    return lookup[normalized]


def parse_serialized_units(value: object) -> list[str]:
    """Parse an augmented sentence/segment cell without executing arbitrary code."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if str(item).strip()]
    return [str(parsed)]


def parse_gold_evidence(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    spans = []
    for item in str(value).split(";"):
        span = item.strip()
        if span.casefold() not in EMPTY_EVIDENCE_MARKERS:
            spans.append(span)
    return spans


def load_workbook(path: Path, *, labeled: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path, engine="openpyxl")
    required = REQUIRED_TRAIN_COLUMNS if labeled else REQUIRED_TEST_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    frame = frame.copy()
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError("row_id must be present and unique")
    if frame["anon_user_id"].isna().any():
        raise ValueError("anon_user_id may not be missing")
    if frame["post"].isna().any():
        raise ValueError("post may not be missing")
    frame["row_id"] = frame["row_id"].astype(str)
    frame["anon_user_id"] = frame["anon_user_id"].astype(str)
    frame["post"] = frame["post"].astype(str)
    if labeled:
        frame["risk_level"] = frame["suicide risk"].map(canonicalize_risk_label)
        frame["gold_evidence"] = frame["evidence for suicide risk level"].map(
            parse_gold_evidence
        )
    return frame


def validate_chronology(frame: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for user_id, group in frame.groupby("anon_user_id", sort=False):
        ids = pd.to_numeric(group["post_id"], errors="coerce")
        if ids.isna().any():
            issues.append(f"user={user_id}: non-numeric post_id")
            continue
        ordered = sorted(ids.astype(int).tolist())
        if ordered != list(range(len(ordered))):
            issues.append(f"user={user_id}: post_id is not contiguous from zero")
    return issues


def alignment_rate(posts: Iterable[str], serialized_units: Iterable[object]) -> tuple[int, int]:
    """Return aligned/total augmented units without retaining their text."""
    aligned = total = 0
    for post, cell in zip(posts, serialized_units, strict=True):
        folded = post.casefold()
        for unit in parse_serialized_units(cell):
            total += 1
            aligned += int(unit.strip().casefold() in folded)
    return aligned, total
