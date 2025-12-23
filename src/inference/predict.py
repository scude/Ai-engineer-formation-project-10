from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src import config
from src.models.recommender import Recommendation, SurpriseRecommender


def _load_artifact(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return path


def _normalize_user_clicks(raw: Any) -> Dict[int, np.ndarray]:
    """
    Ensure user_clicks is a Dict[int, np.ndarray] regardless of how it was pickled.
    - Converts keys to int (handles str keys).
    - Converts values to numpy arrays of int64.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid user_clicks format: expected dict, got {type(raw)}")

    out: Dict[int, np.ndarray] = {}
    for k, v in raw.items():
        # Normalize key
        try:
            user_id = int(k)
        except (TypeError, ValueError):
            continue

        # Normalize value
        if v is None:
            out[user_id] = np.array([], dtype=np.int64)
            continue

        # v may be list/set/np.ndarray
        try:
            arr = np.array(list(v), dtype=np.int64) if not isinstance(v, np.ndarray) else v.astype(np.int64, copy=False)
        except TypeError:
            # Not iterable
            arr = np.array([], dtype=np.int64)

        out[user_id] = arr

    return out


@lru_cache(maxsize=1)
def load_recommender(artifacts_dir: str | None = None) -> SurpriseRecommender:
    base_dir = Path(artifacts_dir) if artifacts_dir else config.ARTIFACTS_DIR

    popular_articles_path = base_dir / config.POPULAR_ARTICLES_PATH.name
    surprise_model_path = base_dir / config.SURPRISE_MODEL_PATH.name
    surprise_items_path = base_dir / config.SURPRISE_ITEMS_PATH.name
    user_clicks_path = base_dir / config.USER_CLICKS_PATH.name

    popular_articles = np.load(_load_artifact(popular_articles_path))
    surprise_items = np.load(_load_artifact(surprise_items_path))

    with _load_artifact(surprise_model_path).open("rb") as f:
        surprise_model = pickle.load(f)

    with _load_artifact(user_clicks_path).open("rb") as f:
        raw_user_clicks: Any = pickle.load(f)

    user_clicks = _normalize_user_clicks(raw_user_clicks)

    return SurpriseRecommender(
        model=surprise_model,
        item_ids=surprise_items.tolist(),
        user_clicks=user_clicks,
        popularity=popular_articles.tolist(),
    )


def predict(user_id: int, artifacts_dir: str | None = None) -> Tuple[List[Recommendation], str]:
    recommender = load_recommender(artifacts_dir)
    return recommender.recommend(int(user_id))


def serialize_recommendations(
    user_id: int,
    recs: List[Recommendation],
    strategy: str,
    *,
    model_name: str | None = None,
    hyperparameters: Dict[str, Any] | None = None,
) -> str:
    payload = {
        "user_id": int(user_id),
        "recommendations": [{"article_id": rec.article_id, "score": rec.score} for rec in recs],
        "strategy": strategy,
        "model": model_name or config.MODEL_NAME,
        "hyperparameters": hyperparameters or config.MODEL_HYPERPARAMETERS,
    }
    return json.dumps(payload)
