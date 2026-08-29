"""A small, version-recorded wrapper around SentenceTransformer MPNet."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np


class MPNetEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None,
        batch_size: int,
        max_seq_length: int,
        normalize_embeddings: bool,
        device: str | None = None,
    ) -> None:
        # Keep the downloaded research model inside the ignored workspace cache unless
        # the caller explicitly chooses a different Hugging Face home.
        workspace_cache = Path(__file__).resolve().parents[2] / ".hf-cache"
        os.environ.setdefault("HF_HOME", str(workspace_cache))
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device} if device else {}
        if revision:
            kwargs["revision"] = revision
        self.model = SentenceTransformer(model_name, **kwargs)
        self.model.max_seq_length = max_seq_length
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def resolved_revision(self) -> str | None:
        try:
            return self.model[0].auto_model.config._commit_hash
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            ),
            dtype=np.float32,
        )


def cosine_scores(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity for normalized embeddings, with a safe fallback."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True).clip(min=1e-12)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).clip(min=1e-12)
    return (left / left_norm) @ (right / right_norm).T
