from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import requests
from flask import Flask, render_template, request

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
    active_users = load_active_users()
    return render_template(
        "index.html",
        user_id_suggestions=user_id_suggestions,
        user_count=user_count,
        active_users=active_users,
        recommendations=None,
        strategy=None,
        model=None,
        hyperparameters=None,
        error=None,
    )


@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id", "").strip()
    user_id_suggestions, valid_user_ids, user_count = load_user_catalog()
    active_users = load_active_users()
    if not user_id:
        return render_template(
            "index.html",
            user_id_suggestions=user_id_suggestions,
            user_count=user_count,
            active_users=active_users,
            recommendations=None,
            strategy=None,
            model=None,
            hyperparameters=None,
            error="Please select a user ID.",
        )

    try:
        payload = {"user_id": int(user_id)}
    except ValueError:
        return render_template(
            "index.html",
            user_id_suggestions=user_id_suggestions,
            user_count=user_count,
            active_users=active_users,
            recommendations=None,
            strategy=None,
            model=None,
            hyperparameters=None,
            error="Invalid user ID provided.",
        )

    if valid_user_ids and payload["user_id"] not in valid_user_ids:
        return render_template(
            "index.html",
            user_id_suggestions=user_id_suggestions,
            user_count=user_count,
            active_users=active_users,
            recommendations=None,
            strategy=None,
            model=None,
            hyperparameters=None,
            error="Unknown user ID. Please enter an ID from the available users.",
        )

    try:
        response = requests.post(AZURE_FUNCTION_URL, json=payload, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")
        data = response.json()
        recommendations = data.get("recommendations", [])
        strategy = data.get("strategy")
        model = data.get("model")
        hyperparameters = data.get("hyperparameters")
        return render_template(
            "index.html",
            user_id_suggestions=user_id_suggestions,
            user_count=user_count,
            active_users=active_users,
            recommendations=recommendations,
            strategy=strategy,
            model=model,
            hyperparameters=hyperparameters,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "index.html",
            user_id_suggestions=user_id_suggestions,
            user_count=user_count,
            active_users=active_users,
            recommendations=None,
            strategy=None,
            model=None,
            hyperparameters=None,
            error=f"Failed to fetch recommendations: {exc}",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
