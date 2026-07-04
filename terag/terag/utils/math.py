"""Small numerical helpers used by TERAG."""

from __future__ import annotations

import numpy as np


def cosine_similarity_score(vector_a, vector_b) -> float:
    """Return cosine similarity for two one-dimensional vectors."""
    a = np.asarray(vector_a, dtype=float).reshape(-1)
    b = np.asarray(vector_b, dtype=float).reshape(-1)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)
