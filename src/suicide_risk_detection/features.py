"""MPNet post embeddings and macro/individual synthetic-anchor features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .anchors import AnchorSet
from .constants import RISK_LABELS, RISK_TO_ID
from .data import parse_serialized_units
from .embedding import MPNetEncoder, cosine_scores
from .spans import clause_spans, sentence_spans, token_window_spans


@dataclass
class FeatureBundle:
    row_ids: np.ndarray
    user_ids: np.ndarray
    blocks: dict[str, np.ndarray]
    feature_names: dict[str, list[str]]
    labels: np.ndarray | None = None


def _stats(values: np.ndarray, names: list[str]) -> list[float]:
    if values.size == 0:
        return [0.0] * len(names)
    functions = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std,
        "var": np.var,
        "max": np.max,
        "min": np.min,
        "p25": lambda x: np.percentile(x, 25),
        "p75": lambda x: np.percentile(x, 75),
    }
    unknown = sorted(set(names) - set(functions))
    if unknown:
        raise ValueError(f"Unknown statistics: {unknown}")
    return [float(functions[name](values)) for name in names]


def _unit_texts(post: str, view: str, segment_max_tokens: int) -> list[str]:
    if view == "sentence":
        return [span.text for span in sentence_spans(post)]
    if view == "segment":
        clauses = clause_spans(post)
        short = [span.text for span in clauses if span.token_count <= segment_max_tokens]
        windows = [
            window.text
            for span in clauses
            if span.token_count > segment_max_tokens
            for window in token_window_spans(span.text, [segment_max_tokens])
        ]
        units = short + windows
        return units or [post]
    raise ValueError(f"Unknown unit view: {view}")


def _flatten_units(
    frame: pd.DataFrame,
    view: str,
    max_tokens: int,
    source_columns: dict[str, str] | None = None,
) -> tuple[list[str], list[slice]]:
    flat: list[str] = []
    locations: list[slice] = []
    source_column = (source_columns or {}).get(view)
    if source_column and source_column not in frame.columns:
        raise ValueError(f"Configured unit source column is missing: {source_column}")
    for _, row in frame.iterrows():
        post = str(row["post"])
        units = (
            parse_serialized_units(row[source_column])
            if source_column
            else _unit_texts(post, view, max_tokens)
        )
        if not units:
            raise ValueError(f"No {view} units found for row_id={row['row_id']}")
        if source_column and any(unit.strip().casefold() not in post.casefold() for unit in units):
            raise ValueError(f"Non-verbatim unit in {source_column} for row_id={row['row_id']}")
        start = len(flat)
        flat.extend(units)
        locations.append(slice(start, len(flat)))
    return flat, locations


def _anchor_embeddings(
    encoder: MPNetEncoder, anchor_set: AnchorSet
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    macro_texts = [" ".join(anchor_set.phrases[label]) for label in RISK_LABELS]
    macro = encoder.encode(macro_texts)
    individual = {label: encoder.encode(anchor_set.phrases[label]) for label in RISK_LABELS}
    return macro, individual


def build_feature_bundle(
    frame: pd.DataFrame,
    *,
    encoder: MPNetEncoder,
    anchor_sets: dict[str, AnchorSet],
    views: list[str],
    statistics: list[str],
    segment_max_tokens: int,
    include_post_embedding: bool,
    unit_source_columns: dict[str, str] | None = None,
) -> FeatureBundle:
    posts = frame["post"].astype(str).tolist()
    blocks: dict[str, np.ndarray] = {}
    names: dict[str, list[str]] = {}
    if include_post_embedding:
        blocks["post_embedding"] = encoder.encode(posts, show_progress=True)
        names["post_embedding"] = [f"mpnet_{index:03d}" for index in range(encoder.dimension)]

    prepared_anchors = {
        strategy: _anchor_embeddings(encoder, anchor_set)
        for strategy, anchor_set in anchor_sets.items()
    }
    for view in views:
        flat_units, locations = _flatten_units(frame, view, segment_max_tokens, unit_source_columns)
        unit_embeddings = encoder.encode(flat_units, show_progress=True)
        for strategy, (macro_anchors, individual_anchors) in prepared_anchors.items():
            macro_similarities = cosine_scores(unit_embeddings, macro_anchors)
            macro_rows: list[list[float]] = []
            individual_rows: list[list[float]] = []
            for location in locations:
                macro_row: list[float] = []
                individual_row: list[float] = []
                unit_matrix = unit_embeddings[location]
                for label_index, label in enumerate(RISK_LABELS):
                    macro_row.extend(_stats(macro_similarities[location, label_index], statistics))
                    local = cosine_scores(unit_matrix, individual_anchors[label])
                    individual_row.extend(_stats(local.max(axis=1), statistics))
                macro_rows.append(macro_row)
                individual_rows.append(individual_row)

            for modality, rows in (("macro", macro_rows), ("individual", individual_rows)):
                key = f"{strategy}__{modality}__{view}"
                blocks[key] = np.asarray(rows, dtype=np.float32)
                names[key] = [
                    f"{key}__{label}__{statistic}"
                    for label in RISK_LABELS
                    for statistic in statistics
                ]

    labels = None
    if "risk_level" in frame:
        labels = frame["risk_level"].map(RISK_TO_ID).to_numpy(dtype=np.int64)
    return FeatureBundle(
        row_ids=frame["row_id"].astype(str).to_numpy(),
        user_ids=frame["anon_user_id"].astype(str).to_numpy(),
        labels=labels,
        blocks=blocks,
        feature_names=names,
    )


def save_feature_bundle(path: Path, bundle: FeatureBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        # Pandas commonly yields object arrays for string columns. Persist fixed-width
        # Unicode so archives remain loadable with the safer allow_pickle=False.
        "row_ids": np.asarray(bundle.row_ids, dtype=str),
        "user_ids": np.asarray(bundle.user_ids, dtype=str),
        "block_names": np.asarray(list(bundle.blocks)),
    }
    if bundle.labels is not None:
        arrays["labels"] = bundle.labels
    for index, (name, matrix) in enumerate(bundle.blocks.items()):
        arrays[f"block_{index}"] = matrix
        arrays[f"names_{index}"] = np.asarray(bundle.feature_names[name])
    np.savez_compressed(path, **arrays)


def load_feature_bundle(path: Path) -> FeatureBundle:
    with np.load(path, allow_pickle=False) as archive:
        block_names = archive["block_names"].astype(str).tolist()
        blocks = {name: archive[f"block_{index}"] for index, name in enumerate(block_names)}
        names = {
            name: archive[f"names_{index}"].astype(str).tolist()
            for index, name in enumerate(block_names)
        }
        labels = archive["labels"] if "labels" in archive.files else None
        return FeatureBundle(
            row_ids=archive["row_ids"].astype(str),
            user_ids=archive["user_ids"].astype(str),
            labels=labels,
            blocks=blocks,
            feature_names=names,
        )


def concatenate_blocks(
    bundle: FeatureBundle, block_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    missing = sorted(set(block_names) - set(bundle.blocks))
    if missing:
        raise KeyError(f"Feature blocks not found: {missing}")
    return (
        np.hstack([bundle.blocks[name] for name in block_names]),
        [feature for name in block_names for feature in bundle.feature_names[name]],
    )
