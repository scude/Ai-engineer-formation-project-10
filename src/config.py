from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = REPO_ROOT / "data" / "news-portal-user-interactions-by-globocom"

# Raw data paths
CLICKS_SAMPLE_PATH: Path = DATA_ROOT / "clicks_sample.csv"
ARTICLES_EMBEDDINGS_PATH: Path = DATA_ROOT / "articles_embeddings.pickle"

# Artifacts directory (can be overridden via environment variable)
ARTIFACTS_DIR: Path = Path(os.getenv("ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ARTICLE_EMBEDDINGS_MATRIX_PATH: Path = ARTIFACTS_DIR / "article_embeddings.npy"
ARTICLE_IDS_PATH: Path = ARTIFACTS_DIR / "article_ids.npy"
POPULAR_ARTICLES_PATH: Path = ARTIFACTS_DIR / "popular_articles.npy"
USER_CLICKS_PATH: Path = ARTIFACTS_DIR / "user_clicks.pkl"

# Recommendation constants
TOP_K_RECOMMENDATIONS: int = 5

__all__ = [
    "REPO_ROOT",
    "DATA_ROOT",
    "CLICKS_SAMPLE_PATH",
    "ARTICLES_EMBEDDINGS_PATH",
    "ARTIFACTS_DIR",
    "ARTICLE_EMBEDDINGS_MATRIX_PATH",
    "ARTICLE_IDS_PATH",
    "POPULAR_ARTICLES_PATH",
    "USER_CLICKS_PATH",
    "TOP_K_RECOMMENDATIONS",
]
