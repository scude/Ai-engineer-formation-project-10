from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.load_data import build_user_clicks, compute_popularity, extract_embedding_matrix, load_article_embeddings


def prepare_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    """Load and return article_ids and embedding matrix."""
    embeddings_df = load_article_embeddings()
    return extract_embedding_matrix(embeddings_df)


def prepare_user_clicks(clicks: pd.DataFrame) -> Dict[int, np.ndarray]:
    return build_user_clicks(clicks)


def prepare_popular_articles(clicks: pd.DataFrame) -> np.ndarray:
    return compute_popularity(clicks)

