"""Leakage-safe grouped validation and early-fusion risk classifiers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class RiskConfiguration:
    experiment: str
    blocks: tuple[str, ...]
    classifier: str
    c: float
    k: int | str

    @property
    def identifier(self) -> str:
        return f"{self.experiment}__{self.classifier}__C{self.c:g}__k{self.k}"


def repeated_stratified_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        for fold, (train, validation) in enumerate(splitter.split(np.zeros_like(y), y, groups)):
            if set(groups[train]) & set(groups[validation]):
                raise RuntimeError("User leakage detected in generated split")
            yield repeat, fold, train, validation


def experiment_block_sets(block_names: list[str]) -> dict[str, tuple[str, ...]]:
    anchors = [name for name in block_names if name != "post_embedding"]
    experiments: dict[str, tuple[str, ...]] = {}
    if "post_embedding" in block_names:
        experiments["post_mpnet"] = ("post_embedding",)
    for name in anchors:
        experiments[f"single__{name}"] = (name,)
    zero = tuple(name for name in anchors if name.startswith("zero_shot__"))
    meta = tuple(name for name in anchors if name.startswith("meta_prompting__"))
    complementary = tuple(
        name
        for name in anchors
        if name.startswith("meta_prompting__macro__")
        or name.startswith("zero_shot__individual__")
    )
    if zero:
        experiments["anchors_zero_shot"] = zero
    if meta:
        experiments["anchors_meta_prompting"] = meta
    if complementary:
        experiments["anchors_complementary"] = complementary
    if anchors:
        experiments["anchors_all"] = tuple(anchors)
    if anchors and "post_embedding" in block_names:
        experiments["early_fusion_all"] = ("post_embedding", *anchors)
        if complementary:
            experiments["early_fusion_complementary"] = (
                "post_embedding",
                *complementary,
            )
    return experiments


def configurations(
    config: dict,
    block_names: list[str],
    block_dimensions: dict[str, int] | None = None,
) -> list[RiskConfiguration]:
    experiments = experiment_block_sets(block_names)
    candidates: list[RiskConfiguration] = []
    for (name, blocks), classifier, c in product(
        experiments.items(),
        config["models"]["classifiers"],
        config["models"]["c_values"],
    ):
        feature_count = (
            sum(block_dimensions[block] for block in blocks) if block_dimensions else None
        )
        k_values = []
        for raw_k in config["models"]["selection_k"]:
            if (
                raw_k == "all"
                and name.startswith("early_fusion")
                and not config["models"].get("allow_all_features_in_early_fusion", False)
            ):
                continue
            if raw_k != "all" and feature_count is not None and int(raw_k) >= feature_count:
                continue
            k_values.append(raw_k)
        if not k_values:
            k_values.append("all")
        for k in k_values:
            candidates.append(RiskConfiguration(name, blocks, classifier, float(c), k))
    return candidates


def build_pipeline(configuration: RiskConfiguration, feature_count: int, seed: int) -> Pipeline:
    k = configuration.k
    if k != "all":
        k = min(int(k), feature_count)
    if configuration.classifier == "logistic_regression":
        classifier = LogisticRegression(
            C=configuration.c,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=seed,
        )
    elif configuration.classifier == "linear_svm":
        classifier = LinearSVC(
            C=configuration.c,
            class_weight="balanced",
            random_state=seed,
            dual="auto",
            max_iter=10000,
        )
    else:
        raise ValueError(f"Unknown classifier: {configuration.classifier}")
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("classifier", classifier),
        ]
    )


def risk_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "risk_weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "risk_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def majority_vote(predictions: list[list[int]]) -> np.ndarray:
    """Stable vote; ties favor the less severe class to avoid silent over-escalation."""
    result = []
    for row_predictions in predictions:
        counts = Counter(row_predictions)
        result.append(min(counts, key=lambda label: (-counts[label], label)))
    return np.asarray(result, dtype=np.int64)
