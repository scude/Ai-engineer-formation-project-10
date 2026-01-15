"""LightFM-style item-to-item recommender utilities.

This module provides a lightweight approximation of the LightFM WARP
training loop (pure Python / NumPy) and helpers to extract item
neighbors for item-to-item recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

CONTEXT_COLUMNS: Sequence[str] = (
    "click_environment",
    "click_deviceGroup",
    "click_os",
    "click_country",
    "click_region",
    "click_referrer_type",
)


def session_weight_from_size(session_sizes: pd.Series | None) -> np.ndarray:
    """Return session-based weights with a safe fallback.

    The weighting follows 1 / log1p(session_size). Missing or
    non-positive values default to 1.0.
    """

    if session_sizes is None:
        return np.ones(0, dtype=np.float32)

    sizes = pd.to_numeric(session_sizes, errors="coerce").fillna(1.0).clip(lower=1.0)
    return (1.0 / np.log1p(sizes)).astype(np.float32)


@dataclass
class LightFMArtifacts:
    model: "LightFMApproximator"
    interactions: sparse.csr_matrix
    user_features: sparse.csr_matrix | None
    item_features: sparse.csr_matrix | None
    item_ids: List[int]


class LightFMApproximator:
    """A minimalist, deterministic approximation of LightFM with WARP-like updates."""

    def __init__(
        self,
        n_components: int = 32,
        learning_rate: float = 0.05,
        epochs: int = 15,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None
        self.user_feature_embeddings: np.ndarray | None = None

    def _user_representation(self, user_idx: int, user_features: sparse.csr_matrix | None) -> np.ndarray:
        assert self.user_embeddings is not None
        base = self.user_embeddings[user_idx]
        if user_features is None or self.user_feature_embeddings is None:
            return base
        feats = user_features.getrow(user_idx)
        if feats.nnz == 0:
            return base
        aggregated = feats @ self.user_feature_embeddings
        return base + np.asarray(aggregated).ravel()

    def fit(
        self,
        interactions: sparse.csr_matrix,
        sample_weight: sparse.csr_matrix | None = None,
        user_features: sparse.csr_matrix | None = None,
        item_features: sparse.csr_matrix | None = None,
    ) -> "LightFMApproximator":
        rng = np.random.default_rng(self.random_state)
        n_users, n_items = interactions.shape
        self.user_embeddings = rng.normal(0, 0.1, size=(n_users, self.n_components)).astype(np.float32)
        self.item_embeddings = rng.normal(0, 0.1, size=(n_items, self.n_components)).astype(np.float32)
        if user_features is not None:
            self.user_feature_embeddings = rng.normal(
                0,
                0.1,
                size=(user_features.shape[1], self.n_components),
            ).astype(np.float32)
        else:
            self.user_feature_embeddings = None

        coo = interactions.tocoo()
        weights = sample_weight.toarray().ravel() if sample_weight is not None else np.ones_like(coo.data)
        lr = self.learning_rate

        for _ in range(self.epochs):
            # Iterate over each observed interaction once per epoch
            for idx in range(len(coo.data)):
                u = coo.row[idx]
                i = coo.col[idx]
                w = weights[idx] if idx < len(weights) else 1.0
                u_vec = self._user_representation(u, user_features)
                pos = self.item_embeddings[i]

                # Sample a negative item uniformly until an unobserved one is found
                neg = rng.integers(0, n_items)
                while interactions[u, neg] > 0:
                    neg = rng.integers(0, n_items)
                neg_vec = self.item_embeddings[neg]

                diff = np.dot(u_vec, pos) - np.dot(u_vec, neg_vec)
                grad = 1.0 / (1.0 + np.exp(diff)) * w

                user_grad = (pos - neg_vec) * grad
                pos_grad = u_vec * grad
                neg_grad = -u_vec * grad

                self.user_embeddings[u] += lr * user_grad
                self.item_embeddings[i] += lr * pos_grad
                self.item_embeddings[neg] += lr * neg_grad

                if user_features is not None and self.user_feature_embeddings is not None:
                    feats = user_features.getrow(u)
                    if feats.nnz:
                        dense_feats = np.asarray(feats.todense()).ravel()
                        self.user_feature_embeddings += lr * dense_feats[:, None] * user_grad[None, :]

        return self

    def get_item_representations(self, item_features: sparse.csr_matrix | None = None) -> Tuple[np.ndarray, np.ndarray]:
        assert self.item_embeddings is not None
        if item_features is None:
            return np.zeros(self.item_embeddings.shape[0], dtype=np.float32), self.item_embeddings
        enriched = item_features @ self.item_embeddings
        return np.zeros(enriched.shape[0], dtype=np.float32), np.asarray(enriched)


def build_user_feature_matrix(train_df: pd.DataFrame, context_columns: Sequence[str]) -> Tuple[sparse.csr_matrix, List[str]]:
    feature_names: List[str] = []
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    present_cols = [c for c in context_columns if c in train_df.columns]
    if not present_cols:
        return sparse.csr_matrix((len(train_df["user_id"].unique()), 0)), feature_names

    user_ids = sorted(train_df["user_id"].unique())
    user_index = {uid: idx for idx, uid in enumerate(user_ids)}

    col_offset = 0
    for col in present_cols:
        counts = train_df.groupby(["user_id", col]).size().unstack(fill_value=0)
        freqs = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
        for feat in freqs.columns:
            feature_names.append(f"{col}={feat}")
        for user_id, row in freqs.iterrows():
            u_idx = user_index[user_id]
            for j, value in enumerate(row.tolist()):
                if value <= 0:
                    continue
                rows.append(u_idx)
                cols.append(col_offset + j)
                data.append(float(value))
        col_offset += freqs.shape[1]

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(feature_names)))
    return matrix, user_ids


def build_interaction_matrices(
    train_df: pd.DataFrame,
    context_columns: Sequence[str],
    *,
    use_user_features: bool = True,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix | None, List[int]]:
    user_codes, user_index = pd.factorize(train_df["user_id"], sort=True)
    item_codes, item_index = pd.factorize(train_df["article_id"], sort=True)
    weights = session_weight_from_size(train_df.get("session_size"))
    if len(weights) == 0:
        weights = np.ones(len(train_df), dtype=np.float32)

    interactions = sparse.csr_matrix(
        (np.ones(len(train_df), dtype=np.float32), (user_codes, item_codes)),
        shape=(len(user_index), len(item_index)),
    )
    sample_weight = sparse.csr_matrix((weights, (user_codes, item_codes)), shape=interactions.shape)

    if use_user_features:
        user_feature_matrix, ordered_users = build_user_feature_matrix(train_df, context_columns)
        # align rows with factorized users
        alignment = {uid: idx for idx, uid in enumerate(ordered_users)}
        mapping = [alignment[int(uid)] for uid in user_index]
        user_features = user_feature_matrix[mapping]
    else:
        user_features = None

    return interactions, sample_weight, user_features, [int(aid) for aid in item_index]


def precompute_item_neighbors(
    item_embeddings: np.ndarray,
    item_ids: Sequence[int],
    top_n: int,
) -> Dict[int, List[Tuple[int, float]]]:
    normalized = normalize(item_embeddings)
    sims = normalized @ normalized.T
    neighbors: Dict[int, List[Tuple[int, float]]] = {}
    for idx, iid in enumerate(item_ids):
        scores = sims[idx]
        top_idx = np.argsort(-scores)
        top_neighbors: List[Tuple[int, float]] = []
        for j in top_idx:
            if j == idx:
                continue
            top_neighbors.append((int(item_ids[j]), float(scores[j])))
            if len(top_neighbors) >= top_n:
                break
        neighbors[int(iid)] = top_neighbors
    return neighbors


def score_from_neighbors(
    user_history: Iterable[int],
    neighbors: Dict[int, List[Tuple[int, float]]],
    seen: set,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for item in user_history:
        for neighbor, sim in neighbors.get(item, []):
            if neighbor in seen:
                continue
            scores[neighbor] = scores.get(neighbor, 0.0) + sim
    return scores

