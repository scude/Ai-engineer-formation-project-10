from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd


def build_pure_covisit_similarity(
    clicks: pd.DataFrame, top_n_neighbors: int = 100, metric: str = "cosine"
) -> Dict[int, Dict[int, float]]:
    """Compute co-visitation similarity between items using unique user histories.

    Mirrors the notebook implementation so that serving matches experimentation results.
    """

    graph: Dict[int, Dict[int, int]] = defaultdict(dict)
    for _, group in clicks.groupby("user_id"):
        items = group.sort_values("timestamp")["clicked_article_id"].astype(int).tolist()
        unique_items = list(dict.fromkeys(items))
        for i, item_i in enumerate(unique_items):
            for item_j in unique_items[i + 1 :]:
                graph[item_i][item_j] = graph[item_i].get(item_j, 0) + 1
                graph[item_j][item_i] = graph[item_j].get(item_i, 0) + 1

    item_user_counts = clicks.groupby("clicked_article_id")["user_id"].nunique().to_dict()

    similarity: Dict[int, Dict[int, float]] = {}
    for item_i, neighbors in graph.items():
        sims = []
        for item_j, count in neighbors.items():
            if metric == "jaccard":
                denom = item_user_counts.get(item_i, 0) + item_user_counts.get(item_j, 0) - count
                if denom <= 0:
                    continue
                sim = count / denom
            else:
                denom = np.sqrt(item_user_counts.get(item_i, 0) * item_user_counts.get(item_j, 0))
                if denom == 0:
                    continue
                sim = count / denom
            sims.append((item_j, float(sim)))

        top_neighbors = sorted(sims, key=lambda x: x[1], reverse=True)[:top_n_neighbors]
        similarity[item_i] = {int(j): float(sim) for j, sim in top_neighbors}

    return similarity


def compute_normalized_popularity(clicks: pd.DataFrame) -> Dict[int, float]:
    popularity = clicks["clicked_article_id"].value_counts()
    if popularity.empty:
        return {}

    max_pop = float(popularity.max())
    normalized = (popularity / max_pop).astype(float)
    return {int(item): float(score) for item, score in normalized.items()}
