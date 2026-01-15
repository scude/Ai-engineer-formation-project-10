from __future__ import annotations

import numpy as np


def cosine_similarity(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single vector and a matrix of vectors."""
    if vector.ndim != 1:
        raise ValueError("Input vector must be one-dimensional")
    if matrix.ndim != 2:
        raise ValueError("Matrix must be two-dimensional")

    vector_norm = np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Avoid division by zero
    if vector_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    denom = matrix_norms * vector_norm
    # Prevent division by zero for items with zero norm
    denom = np.where(denom == 0, 1e-12, denom)

    similarities = matrix @ vector / denom
    return similarities.astype(np.float32)

