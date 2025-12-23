from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = REPO_ROOT / "data" / "news-portal-user-interactions-by-globocom"

# Raw data paths
CLICKS_DIR: Path = DATA_ROOT / "clicks"
ARTICLES_EMBEDDINGS_PATH: Path = DATA_ROOT / "articles_embeddings.pickle"

# Artifacts directory (can be overridden via environment variable)
ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_EMBEDDINGS_MATRIX_PATH: Path = ARTIFACTS_DIR / "article_embeddings.npy"
ARTICLE_IDS_PATH: Path = ARTIFACTS_DIR / "article_ids.npy"
POPULAR_ARTICLES_PATH: Path = ARTIFACTS_DIR / "popular_articles.npy"
POPULARITY_SCORES_PATH: Path = ARTIFACTS_DIR / "popularity_scores.pkl"
USER_CLICKS_PATH: Path = ARTIFACTS_DIR / "user_clicks.pkl"
COVISIT_SIMILARITY_PATH: Path = ARTIFACTS_DIR / "covisit_similarity.pkl"
SURPRISE_MODEL_PATH: Path = ARTIFACTS_DIR / "surprise_model.pkl"
SURPRISE_ITEMS_PATH: Path = ARTIFACTS_DIR / "surprise_items.npy"

# Recommendation constants
TOP_K_RECOMMENDATIONS: int = 5

# Serving metadata
MODEL_NAME: str = "Surprise-SVD"
MODEL_HYPERPARAMETERS: dict = {
    "n_factors": 50,
    "reg_all": 0.02,
    "lr_all": 0.005,
}
DEFAULT_SIMILARITY_METRIC: str = "cosine"

__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "CLICKS_DIR",
    "ARTICLES_EMBEDDINGS_PATH",
    "ARTIFACTS_DIR",
    "ARTICLE_EMBEDDINGS_MATRIX_PATH",
    "ARTICLE_IDS_PATH",
    "POPULAR_ARTICLES_PATH",
    "POPULARITY_SCORES_PATH",
    "USER_CLICKS_PATH",
    "COVISIT_SIMILARITY_PATH",
    "TOP_K_RECOMMENDATIONS",
    "MODEL_NAME",
    "MODEL_HYPERPARAMETERS",
    "DEFAULT_SIMILARITY_METRIC",
]
