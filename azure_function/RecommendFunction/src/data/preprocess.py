from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data.diagnostics import compute_overlap_summary, format_overlap_summary
from src.data.load_data import (
    align_embeddings_with_clicks,
    build_user_clicks,
    compute_popularity,
    extract_embedding_matrix,
    load_article_embeddings,
)


def prepare_embeddings(
    clicks: pd.DataFrame, embeddings_df: pd.DataFrame | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and validate they overlap with clicked articles.

    The recommendation step still needs embeddings for unseen articles, so we
    only use :func:`align_embeddings_with_clicks` as a guardrail: it raises if
    none of the clicked articles have embeddings, which would force a popular
    fallback even for active users.
    """

    embeddings_df = embeddings_df if embeddings_df is not None else load_article_embeddings()

    # Validate there is at least one overlapping article id; keep the full set
    # so we can recommend items the user has not clicked yet. If no overlap is
    # found we still proceed with the full embeddings so the pipeline can run,
    # but we surface the situation to help diagnose column mismatches.
    try:
        align_embeddings_with_clicks(embeddings_df, clicks)
    except ValueError as exc:
        print(
            "Warning: no clicked articles were found in the embeddings. "
            f"Proceeding with the full embeddings set. Details: {exc}"
        )

    summary = compute_overlap_summary(clicks, embeddings_df)
    print("Data summary:", format_overlap_summary(summary))

    return extract_embedding_matrix(embeddings_df)


def prepare_user_clicks(clicks: pd.DataFrame) -> Dict[int, np.ndarray]:
    return build_user_clicks(clicks)


def prepare_popular_articles(clicks: pd.DataFrame) -> np.ndarray:
    return compute_popularity(clicks)

