"""Anchor-guided, verbatim evidence retrieval and official-style Phrase F1."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .anchors import AnchorSet
from .embedding import MPNetEncoder, cosine_scores
from .spans import Span, evidence_candidates


@dataclass(frozen=True)
class ScoredSpan:
    span: Span
    score: float


def _token_count(text: str) -> int:
    return len(text.split())


def phrases_match(predicted: str, gold: str) -> bool:
    predicted_normalized = " ".join(predicted.casefold().split())
    gold_normalized = " ".join(gold.casefold().split())
    if not predicted_normalized or not gold_normalized:
        return False
    contained = predicted_normalized in gold_normalized or gold_normalized in predicted_normalized
    return contained and _token_count(predicted) <= 3 * max(1, _token_count(gold))


def phrase_counts(predicted: Sequence[str], gold: Sequence[str]) -> tuple[int, int, int]:
    """Maximum one-to-one matching count, returned as TP, predicted count, gold count."""
    edges = [
        [gold_index for gold_index, target in enumerate(gold) if phrases_match(span, target)]
        for span in predicted
    ]
    matched_gold: dict[int, int] = {}

    def augment(predicted_index: int, seen: set[int]) -> bool:
        for gold_index in edges[predicted_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = predicted_index
                return True
        return False

    matches = sum(augment(index, set()) for index in range(len(predicted)))
    return matches, len(predicted), len(gold)


def phrase_f1_per_post(predicted: Sequence[str], gold: Sequence[str]) -> float:
    if not predicted and not gold:
        return 1.0
    true_positive, predicted_count, gold_count = phrase_counts(predicted, gold)
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mean_phrase_f1(predictions: Sequence[Sequence[str]], targets: Sequence[Sequence[str]]) -> float:
    return float(np.mean([phrase_f1_per_post(p, g) for p, g in zip(predictions, targets, strict=True)]))


def _overlap_ratio(left: Span, right: Span) -> float:
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union else 0.0


class EvidenceRetriever:
    def __init__(
        self,
        encoder: MPNetEncoder,
        anchor_sets: dict[str, AnchorSet],
        *,
        window_sizes: list[int],
        candidate_max_tokens: int,
    ) -> None:
        self.encoder = encoder
        self.window_sizes = window_sizes
        self.candidate_max_tokens = candidate_max_tokens
        self.anchor_embeddings: dict[str, list[np.ndarray]] = {}
        for label in ("Ideation", "Behavior", "Attempt"):
            self.anchor_embeddings[label] = [
                encoder.encode(anchor_set.phrases[label]) for anchor_set in anchor_sets.values()
            ]

    def score(self, post: str, risk_label: str) -> list[ScoredSpan]:
        return self.score_many([post], [risk_label])[0]

    def score_many(
        self, posts: Sequence[str], risk_labels: Sequence[str], *, show_progress: bool = False
    ) -> list[list[ScoredSpan]]:
        """Score candidates in one MPNet call to avoid per-post inference overhead."""
        if len(posts) != len(risk_labels):
            raise ValueError("posts and risk_labels must have identical lengths")
        candidates_by_row: list[list[Span]] = []
        flat_candidates: list[Span] = []
        locations: list[slice] = []
        for post, risk_label in zip(posts, risk_labels, strict=True):
            candidates = (
                []
                if risk_label == "Indicator"
                else evidence_candidates(
                    post,
                    window_sizes=self.window_sizes,
                    max_tokens=self.candidate_max_tokens,
                )
            )
            candidates_by_row.append(candidates)
            start = len(flat_candidates)
            flat_candidates.extend(candidates)
            locations.append(slice(start, len(flat_candidates)))
        embeddings = self.encoder.encode(
            [candidate.text for candidate in flat_candidates], show_progress=show_progress
        )
        rows: list[list[ScoredSpan]] = []
        for risk_label, candidates, location in zip(
            risk_labels, candidates_by_row, locations, strict=True
        ):
            if not candidates:
                rows.append([])
                continue
            row_embeddings = embeddings[location]
            strategy_scores = [
                cosine_scores(row_embeddings, anchors).max(axis=1)
                for anchors in self.anchor_embeddings[risk_label]
            ]
            scores = np.mean(np.vstack(strategy_scores), axis=0)
            rows.append(
                sorted(
                    [
                        ScoredSpan(span, float(score))
                        for span, score in zip(candidates, scores, strict=True)
                    ],
                    key=lambda item: (-item.score, item.span.token_count, item.span.start),
                )
            )
        return rows


def select_scored_spans(
    scored: Sequence[ScoredSpan],
    *,
    top_k: int,
    min_similarity: float,
    max_tokens: int,
    max_overlap: float = 0.8,
) -> list[Span]:
    selected: list[Span] = []
    for item in scored:
        if item.score < min_similarity or item.span.token_count > max_tokens:
            continue
        if any(_overlap_ratio(item.span, previous) > max_overlap for previous in selected):
            continue
        selected.append(item.span)
        if len(selected) == top_k:
            break
    return sorted(selected, key=lambda span: span.start)


def tune_evidence_parameters(
    scored_rows: Sequence[Sequence[ScoredSpan]],
    gold_rows: Sequence[Sequence[str]],
    *,
    top_k_values: list[int],
    min_similarity_values: list[float],
    max_tokens_values: list[int],
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    results: list[dict[str, int | float]] = []
    for top_k, threshold, max_tokens in product(
        top_k_values, min_similarity_values, max_tokens_values
    ):
        predictions = [
            [span.text for span in select_scored_spans(
                row,
                top_k=top_k,
                min_similarity=threshold,
                max_tokens=max_tokens,
            )]
            for row in scored_rows
        ]
        results.append(
            {
                "top_k": top_k,
                "min_similarity": threshold,
                "max_tokens": max_tokens,
                "phrase_f1": mean_phrase_f1(predictions, gold_rows),
            }
        )
    results.sort(
        key=lambda row: (
            -float(row["phrase_f1"]),
            int(row["top_k"]),
            int(row["max_tokens"]),
            -float(row["min_similarity"]),
        )
    )
    return results[0], results


def validate_verbatim(post: str, spans: Sequence[str]) -> None:
    for span in spans:
        if not span or span not in post:
            raise ValueError("Every evidence prediction must be a non-empty verbatim post substring")
