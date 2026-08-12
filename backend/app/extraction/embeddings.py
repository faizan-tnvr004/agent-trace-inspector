"""Sentence embeddings for the extraction engine.

Not in the specification's file tree; separated out because both `scoring` and
`claims` need the same model and the same similarity function, and a second
copy of the loading and caching logic would be a place for the two to diverge.

The extraction engine must be deterministic (FR-16, NFR-6) and must contain no
LLM calls. Three things here serve that:

* the model is loaded once, in inference mode, with gradients disabled
* embeddings are cached by exact text, so a repeated call cannot drift
* similarities are rounded to a fixed precision

Rounding is the load-bearing one. Float arithmetic on identical CPU input is
already reproducible, but scores flow into ranking comparisons and into JSON
served over the API, and rounding makes equality between two runs of the same
input exact rather than nearly exact.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

__all__ = [
    "PRECISION",
    "cosine_similarity",
    "embed",
    "model_name",
    "reset_cache",
]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Decimal places retained on every similarity. Six is far finer than any
# threshold in use and keeps repeated extraction byte-identical.
PRECISION = 6

_model = None
_model_lock = threading.Lock()


def model_name() -> str:
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)


def _get_model():
    """Load the sentence transformer once.

    CPU only: introducing a GPU dependency is explicitly forbidden, and the
    corpus is small enough that it would buy nothing.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_grad_enabled(False)
                model = SentenceTransformer(model_name(), device="cpu")
                model.eval()
                _model = model
    return _model


@lru_cache(maxsize=8192)
def _embed_one(text: str):
    model = _get_model()
    # normalize_embeddings=True makes a dot product the cosine similarity.
    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed(text: str):
    """Return the unit-length embedding of ``text``."""
    return _embed_one(text)


def cosine_similarity(left: str, right: str) -> float:
    """Cosine similarity of two texts, clamped to [0, 1] and rounded.

    Cosine similarity is defined on [-1, 1], but every consumer here treats the
    value as "how much of this survived", where a negative number has no
    meaning distinct from zero. Clamping keeps the contract that scores are in
    [0, 1], which the specification requires for evidence survival.

    Empty text scores 0.0: an empty output cannot have survived into the final
    answer, and the model would otherwise return an arbitrary direction.
    """
    if not left or not left.strip() or not right or not right.strip():
        return 0.0

    similarity = float(_embed_one(left) @ _embed_one(right))
    return round(max(0.0, min(1.0, similarity)), PRECISION)


def reset_cache() -> None:
    """Clear the embedding cache. Used by tests."""
    _embed_one.cache_clear()
