from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src import config
from src.models.similarity import cosine_similarity


@dataclass
class Recommendation:
    article_id: int
    score: float


class ContentRecommender:
    """Content-based recommender using precomputed artifacts."""

    def __init__(
        self,
        article_ids: np.ndarray,
        article_embeddings: np.ndarray,
        user_clicks: Dict[int, np.ndarray],
        popular_articles: np.ndarray,
        top_k: int | None = None,
    ) -> None:
        self.article_ids = article_ids
        self.article_embeddings = article_embeddings
        self.user_clicks = user_clicks
        self.popular_articles = popular_articles
        self.top_k = top_k or config.TOP_K_RECOMMENDATIONS
        self.article_id_to_index = {int(aid): idx for idx, aid in enumerate(article_ids)}

    def _get_user_history(self, user_id: int) -> np.ndarray:
        """
        Return clicked article ids for a user.
        Robust to user_clicks dict keys being int or str.
        """
        uid = int(user_id)

        history = self.user_clicks.get(uid)
        if history is None:
            history = self.user_clicks.get(str(uid))

        if history is None:
            return np.array([], dtype=np.int64)

        if isinstance(history, np.ndarray):
            return history.astype(np.int64, copy=False)

        # Fallback if history is list/set/etc.
        return np.array(list(history), dtype=np.int64)

    def _user_profile(self, user_id: int) -> np.ndarray | None:
        clicked_ids = self._get_user_history(user_id)
        if clicked_ids.size == 0:
            return None

        indices = [self.article_id_to_index[aid] for aid in clicked_ids if aid in self.article_id_to_index]
        if not indices:
            return None

        vectors = self.article_embeddings[indices]
        profile = vectors.mean(axis=0)
        return profile.astype(np.float32)

    def recommend(self, user_id: int) -> Tuple[List[Recommendation], str]:
        profile = self._user_profile(user_id)
        clicked_ids = set(self._get_user_history(user_id).tolist())

        if profile is None:
            top_ids = self.popular_articles[: self.top_k]
            recs = [Recommendation(article_id=int(aid), score=0.0) for aid in top_ids]
            return recs, "popular"

        scores = cosine_similarity(profile, self.article_embeddings)

        # Exclude already clicked articles
        mask = np.isin(self.article_ids, list(clicked_ids))
        scores_filtered = np.where(mask, -np.inf, scores)

        top_indices = np.argsort(scores_filtered)[-self.top_k :][::-1]
        recs = [
            Recommendation(article_id=int(self.article_ids[idx]), score=float(scores[idx]))
            for idx in top_indices
            if scores_filtered[idx] != -np.inf
        ]

        return recs, "content-based"

