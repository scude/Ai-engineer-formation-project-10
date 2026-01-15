from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class OverlapSummary:
    total_clicks: int
    unique_users: int
    unique_clicked_articles: int
    embedding_articles: int
    overlap_count: int

    @property
    def missing_clicked_articles(self) -> int:
        return max(self.unique_clicked_articles - self.overlap_count, 0)


def compute_overlap_summary(clicks: pd.DataFrame, embeddings_df: pd.DataFrame) -> OverlapSummary:
    """Return a compact summary of click/embed overlap.

    This is intended for diagnostics so users can quickly see whether their
    dataset columns are aligned without digging into the raw files.
    """

    clicked_articles = set(clicks["clicked_article_id"].dropna().astype(int).unique())
    embedded_articles = set(embeddings_df["article_id"].dropna().astype(int).unique())
    overlap = clicked_articles & embedded_articles

    return OverlapSummary(
        total_clicks=len(clicks),
        unique_users=clicks["user_id"].nunique(dropna=True),
        unique_clicked_articles=len(clicked_articles),
        embedding_articles=len(embedded_articles),
        overlap_count=len(overlap),
    )


def format_overlap_summary(summary: OverlapSummary) -> str:
    """Render :class:`OverlapSummary` as a human-readable message."""

    parts: Iterable[str] = (
        f"Clicks: {summary.total_clicks} rows across {summary.unique_users} users.",
        f"Articles: {summary.unique_clicked_articles} clicked vs {summary.embedding_articles} embedded.",
        f"Overlap: {summary.overlap_count} articles with both clicks and embeddings",
    )

    if summary.missing_clicked_articles:
        parts = [*parts, f"Missing embeddings for {summary.missing_clicked_articles} clicked articles."]

    return " ".join(parts)
