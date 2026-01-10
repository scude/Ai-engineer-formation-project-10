from __future__ import annotations

import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from urllib.parse import urlparse

from flask import Flask, jsonify, render_template, request

# Ensure src package is importable
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from src import config  # noqa: E402

app = Flask(__name__)

AZURE_FUNCTION_URL = os.getenv("AZURE_FUNCTION_URL", "http://localhost:7071/api/recommend")


def _normalize_user_ids(raw: Any) -> List[int]:
    """Extract a sorted list of unique ``user_id`` values from various payload shapes."""

    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas is always available via requirements
        pd = None

    user_id_candidates: Iterable[Any]

    if isinstance(raw, dict):
        user_id_candidates = raw.keys()
    elif isinstance(raw, (list, tuple, set)):
        user_id_candidates = raw
    elif pd is not None and isinstance(raw, (pd.Series, pd.Index)):
        user_id_candidates = raw.tolist()
    elif pd is not None and isinstance(raw, pd.DataFrame):
        normalized_cols = {col.lower().replace("-", "_"): col for col in raw.columns}
        for target in ("user_id", "userid", "user"):
            if target in normalized_cols:
                column = normalized_cols[target]
                break
        else:
            raise ValueError(
                "User catalog DataFrame is missing a user_id column; "
                f"found columns: {list(raw.columns)}"
            )

        user_id_candidates = raw[column].tolist()
    else:
        raise ValueError(f"Unsupported user_clicks structure: {type(raw)}")

    normalized = []
    for candidate in user_id_candidates:
        try:
            normalized.append(int(candidate))
        except (TypeError, ValueError):
            continue

    return sorted(set(normalized))


def load_user_catalog(max_suggestions: int = 200) -> Tuple[List[int], Set[int], int]:
    """Return a preview list of user IDs plus the full set for validation."""

    user_clicks_path = config.USER_CLICKS_PATH
    if not user_clicks_path.exists():
        return [], set(), 0

    with user_clicks_path.open("rb") as f:
        raw_user_clicks: Any = pickle.load(f)

    user_ids = _normalize_user_ids(raw_user_clicks)
    user_id_set = set(user_ids)
    return user_ids[:max_suggestions], user_id_set, len(user_id_set)


def load_article_catalog(max_suggestions: int = 200) -> Tuple[List[int], Set[int], int]:
    """Return a preview list of article IDs plus the full set for validation."""

    article_ids_path = config.ARTICLE_IDS_PATH
    if not article_ids_path.exists():
        return [], set(), 0

    article_ids = np.load(article_ids_path).astype(int).tolist()
    unique_ids = sorted(set(article_ids))
    return unique_ids[:max_suggestions], set(unique_ids), len(unique_ids)


@lru_cache(maxsize=1)
def load_content_artifacts() -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Load content-based artifacts for article similarity."""

    article_ids_path = config.ARTICLE_IDS_PATH
    embeddings_path = config.ARTICLE_EMBEDDINGS_MATRIX_PATH
    if not article_ids_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "Content-based artifacts missing. Ensure article_ids.npy and article_embeddings_pca_32.npy exist."
        )

    article_ids = np.load(article_ids_path).astype(int)
    embeddings = np.load(embeddings_path)
    if embeddings.shape[0] != article_ids.shape[0]:
        raise ValueError(
            "Embeddings row count does not match article IDs length. Rebuild content-based artifacts."
        )

    article_id_to_index = {int(aid): idx for idx, aid in enumerate(article_ids.tolist())}
    return article_ids, embeddings, article_id_to_index


def _normalize_user_clicks(raw: Any) -> Dict[int, np.ndarray]:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid user_clicks format: expected dict, got {type(raw)}")

    normalized: Dict[int, np.ndarray] = {}
    for key, value in raw.items():
        try:
            user_id = int(key)
        except (TypeError, ValueError):
            continue

        if value is None:
            normalized[user_id] = np.array([], dtype=np.int64)
            continue

        try:
            arr = np.array(list(value), dtype=np.int64) if not isinstance(value, np.ndarray) else value.astype(np.int64)
        except TypeError:
            arr = np.array([], dtype=np.int64)

        normalized[user_id] = arr

    return normalized


def load_active_users(max_users: int = 10, max_articles: int = 6) -> List[Dict[str, Any]]:
    user_clicks_path = config.USER_CLICKS_PATH
    if not user_clicks_path.exists():
        return []

    with user_clicks_path.open("rb") as f:
        raw_user_clicks: Any = pickle.load(f)

    user_clicks = _normalize_user_clicks(raw_user_clicks)
    active_users = sorted(user_clicks.items(), key=lambda item: item[1].size, reverse=True)

    results: List[Dict[str, Any]] = []
    for user_id, article_ids in active_users[:max_users]:
        cleaned_articles = sorted({int(article) for article in article_ids.tolist() if article is not None})
        results.append(
            {
                "user_id": user_id,
                "total_clicks": int(article_ids.size),
                "unique_articles": len(cleaned_articles),
                "sample_articles": cleaned_articles[:max_articles],
            }
        )

    return results


@app.route("/", methods=["GET"])
def index():
    user_id_suggestions, _, user_count = load_user_catalog()
    article_id_suggestions, _, article_count = load_article_catalog()
    active_users = load_active_users()
    return render_template(
        "index.html",
        user_id_suggestions=user_id_suggestions,
        user_count=user_count,
        article_id_suggestions=article_id_suggestions,
        article_count=article_count,
        active_users=active_users,
        default_hybrid_weight=config.MODEL_HYPERPARAMETERS["hybrid_cf_weight"],
        azure_function_url=AZURE_FUNCTION_URL,
        article_similarity_url="/article-similarity",
        recommendations=None,
        strategy=None,
        model=None,
        hyperparameters=None,
        error=None,
    )


def _azure_function_base_url() -> str:
    parsed = urlparse(AZURE_FUNCTION_URL)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return AZURE_FUNCTION_URL


def _azure_function_path() -> str:
    parsed = urlparse(AZURE_FUNCTION_URL)
    return parsed.path or "/api/recommend"


@app.route("/openapi.json", methods=["GET"])
def openapi_spec():
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "Content Recommendations API",
            "version": "1.0.0",
            "description": "API to generate content recommendations for a given user.",
        },
        "servers": [{"url": _azure_function_base_url()}],
        "paths": {
            _azure_function_path(): {
                "post": {
                    "summary": "Generate recommendations",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "user_id": {"type": "integer"},
                                        "hybrid_weight": {"type": "number", "format": "float"},
                                        "hybrid_cf_weight": {"type": "number", "format": "float"},
                                        "hybrid_cb_weight": {"type": "number", "format": "float"},
                                    },
                                    "required": ["user_id"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Recommendations payload",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "user_id": {"type": "integer"},
                                            "strategy": {"type": "string"},
                                            "model": {"type": "string"},
                                            "hyperparameters": {"type": "object"},
                                            "recommendations": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "article_id": {"type": "integer"},
                                                        "score": {"type": "number", "format": "float"},
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid request payload"},
                        "500": {"description": "Server error"},
                    },
                }
            },
            "/article-similarity": {
                "post": {
                    "summary": "Find similar articles",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "article_id": {"type": "integer"},
                                    },
                                    "required": ["article_id"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Similar articles payload",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "article_id": {"type": "integer"},
                                            "recommendations": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "article_id": {"type": "integer"},
                                                        "score": {"type": "number", "format": "float"},
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        },
                        "400": {"description": "Invalid request payload"},
                        "404": {"description": "Unknown article_id"},
                        "500": {"description": "Server error"},
                    },
                }
            },
        },
    }
    return jsonify(spec)


@app.route("/docs", methods=["GET"])
def docs():
    return render_template("docs.html")


@app.route("/article-similarity", methods=["POST"])
def article_similarity():
    payload = request.get_json(silent=True) or {}
    if "article_id" not in payload:
        return jsonify({"error": "'article_id' is required"}), 400

    try:
        article_id = int(payload["article_id"])
    except (TypeError, ValueError):
        return jsonify({"error": "'article_id' must be an integer"}), 400

    try:
        article_ids, embeddings, article_id_to_index = load_content_artifacts()
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 500

    if article_id not in article_id_to_index:
        return jsonify({"error": "Unknown article_id."}), 404

    from src.models.similarity import cosine_similarity

    target_index = article_id_to_index[article_id]
    target_vector = embeddings[target_index]
    scores = cosine_similarity(target_vector, embeddings)
    scores[target_index] = -np.inf

    top_k = min(config.TOP_K_RECOMMENDATIONS, len(article_ids) - 1)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    recommendations = [
        {"article_id": int(article_ids[idx]), "score": float(scores[idx])}
        for idx in top_indices
        if scores[idx] != -np.inf
    ]

    return jsonify({"article_id": article_id, "recommendations": recommendations})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
