from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src import config


def _resolve_column(df: pd.DataFrame, target: str, candidates: Iterable[str]) -> str:
    """Return the column from ``df`` matching one of the candidates.

    Args:
        df: DataFrame to inspect.
        target: Human-friendly name for error messages.
        candidates: Possible column names ordered by priority.

    Raises:
        ValueError: if none of the candidates are found.
    """

    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Clicks data missing expected {target} column. "
        f"Tried {', '.join(candidates)} but found {list(df.columns)}"
    )


def _load_clicks_file(path: Path) -> pd.DataFrame:
    """Load a single clicks CSV file and normalize critical columns."""

    df = pd.read_csv(path)

    user_col = _resolve_column(df, "user_id", ["user_id"])
    article_col = _resolve_column(
        df, "clicked article id", ["clicked_article_id", "click_article_id", "article_id"]
    )
    ts_col = _resolve_column(df, "timestamp", ["click_timestamp", "timestamp"])

    normalized = df.rename(
        columns={user_col: "user_id", article_col: "clicked_article_id", ts_col: "timestamp"}
    )
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])

    return normalized[["user_id", "clicked_article_id", "timestamp"]]


def load_clicks(clicks_dir: Path | None = None, limit_files: int | None = None) -> pd.DataFrame:
    """Load clicks from the Globo dataset directory.

    Args:
        clicks_dir: Directory containing ``clicks_hour_XXX.csv`` files or a
            single CSV file path. Defaults to :data:`config.CLICKS_DIR`.
        limit_files: Optional cap on the number of hourly files to read (useful
            for local experimentation).

    Returns:
        DataFrame with columns [user_id, clicked_article_id, timestamp].
    """

    source = Path(clicks_dir) if clicks_dir is not None else config.CLICKS_DIR

    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(source.glob("clicks_hour_*.csv"))
        if not files:
            raise FileNotFoundError(f"No clicks_hour_*.csv files found in {source}")
    else:
        raise FileNotFoundError(f"Clicks path {source} does not exist")

    if limit_files is not None:
        files = files[:limit_files]

    frames = [_load_clicks_file(path) for path in files]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def _normalize_embedding(embedding: Iterable[float] | float | int) -> np.ndarray:
    """Normalize a single embedding to a 1D float32 numpy array."""
    if np.isscalar(embedding):
        return np.asarray([embedding], dtype=np.float32)

    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim == 0:
        return np.asarray([arr.item()], dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.flatten()
    return arr


def load_article_embeddings() -> pd.DataFrame:
    """Load article embeddings from pickle and normalize the structure.

    Supports pickle content being a pandas DataFrame (with flexible column names),
    a dictionary mapping article_id to embedding vectors, or various array/record
    combinations produced by the Globo data dumps.
    """
    embeddings_path: Path = config.ARTICLES_EMBEDDINGS_PATH
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"articles_embeddings.pickle not found at {embeddings_path}. Ensure data is available."
        )

    with embeddings_path.open("rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, pd.DataFrame):
        article_col = _resolve_column(payload, "article_id", ["article_id", "id", "articleId"])
        embedding_col = _resolve_column(payload, "embedding", ["embedding", "embeddings", "vector", "vectors"])

        df = payload.rename(columns={article_col: "article_id", embedding_col: "embedding"}).copy()
        df["article_id"] = df["article_id"].astype(int)
        df["embedding"] = df["embedding"].apply(_normalize_embedding)
        return df[["article_id", "embedding"]]

    if isinstance(payload, dict):
        records = [
            {"article_id": int(article_id), "embedding": _normalize_embedding(vec)}
            for article_id, vec in payload.items()
        ]
        return pd.DataFrame.from_records(records, columns=["article_id", "embedding"])

    # Some dataset dumps use a tuple/list of (article_ids, embeddings_matrix)
    if isinstance(payload, (list, tuple)) and len(payload) == 2 and not isinstance(payload[0], dict):
        candidate_df = _build_df_from_arrays(payload[0], payload[1])
        if candidate_df is not None:
            return candidate_df

    # Handle list/array of pair-like rows or dict records
    if isinstance(payload, (list, tuple, np.ndarray)):
        candidate_df = _build_df_from_records(payload)
        if candidate_df is not None:
            return candidate_df

    raise TypeError(
        "Unsupported embeddings format. Expecting pandas.DataFrame, dict mapping article_id to embedding array, "
        "or a tuple/list with article ids and embeddings."
    )


def _build_df_from_arrays(article_ids: Iterable, embeddings: Iterable) -> pd.DataFrame | None:
    """Attempt to build a DataFrame from separate ids and embeddings arrays."""
    try:
        ids_array = np.asarray(list(article_ids))

        # If embeddings is already a numeric matrix (N, D), keep its rows directly
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            emb_array = embeddings
        else:
            emb_array = np.asarray(list(embeddings), dtype=object)
    except Exception:
        return None

    if ids_array.ndim != 1 or emb_array.shape[0] != ids_array.shape[0]:
        return None

    normalized_embeddings = [_normalize_embedding(vec) for vec in emb_array]
    return pd.DataFrame({"article_id": ids_array.astype(int), "embedding": normalized_embeddings})


def _build_df_from_records(records: Iterable) -> pd.DataFrame | None:
    """Handle diverse record shapes for embeddings payload."""
    try:
        materialized = list(records)
    except Exception:
        return None

    if not materialized:
        return None

    # If numpy array with shape (n, 2) convert to list of pairs
    if isinstance(records, np.ndarray) and records.ndim == 2 and records.shape[1] >= 2:
        materialized = records.tolist()

    first = materialized[0]

    # List of dict records
    if isinstance(first, dict) and {"article_id", "embedding"}.issubset(first.keys()):
        df = pd.DataFrame(materialized)
        df["article_id"] = df["article_id"].astype(int)
        df["embedding"] = df["embedding"].apply(_normalize_embedding)
        return df[["article_id", "embedding"]]

    # List/array of pair-like items: (article_id, embedding)
    if isinstance(first, (list, tuple, np.ndarray)) and len(first) >= 2:
        ids, embeddings = zip(*[(row[0], row[1]) for row in materialized])
        return _build_df_from_arrays(ids, embeddings)

    return None


def align_embeddings_with_clicks(embeddings_df: pd.DataFrame, clicks: pd.DataFrame) -> pd.DataFrame:
    """Keep only embeddings whose article_id appears in clicks.

    Raises:
        ValueError: if no embeddings match the clicked articles (content-based impossible).
    """
    click_article_ids = set(clicks["clicked_article_id"].dropna().astype(int).unique())
    aligned = embeddings_df[embeddings_df["article_id"].astype(int).isin(click_article_ids)].copy()

    if aligned.empty:
        raise ValueError(
            "No article embeddings match clicked_article_id from clicks data. "
            "Your articles_embeddings.pickle is not aligned with Globo article IDs."
        )

    aligned["article_id"] = aligned["article_id"].astype(int)
    return aligned[["article_id", "embedding"]]


def extract_embedding_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert embeddings DataFrame into aligned ids and matrix."""
    article_ids = df["article_id"].to_numpy(dtype=np.int64)
    matrix = np.stack(df["embedding"].to_numpy())
    return article_ids, matrix


def compute_popularity(clicks: pd.DataFrame) -> np.ndarray:
    """Return article_ids sorted by popularity (descending)."""
    popularity = clicks["clicked_article_id"].value_counts()
    return popularity.index.to_numpy(dtype=np.int64)


def build_user_clicks(clicks: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Map user_id to the unique set of clicked articles."""
    grouped = (
        clicks.dropna(subset=["clicked_article_id"])
        .groupby("user_id")["clicked_article_id"]
        .apply(lambda series: series.astype(int).unique().tolist())
    )
    return {int(user_id): np.asarray(values, dtype=np.int64) for user_id, values in grouped.items()}
