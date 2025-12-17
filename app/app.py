from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from flask import Flask, render_template, request

# Ensure src package is importable
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from src import config  # noqa: E402

app = Flask(__name__)

AZURE_FUNCTION_URL = os.getenv("AZURE_FUNCTION_URL", "http://localhost:7071/api/recommend")


def load_user_catalog(max_suggestions: int = 200) -> Tuple[List[int], Set[int], int]:
    """Return a preview list of user IDs plus the full set for validation."""

    user_clicks_path = config.USER_CLICKS_PATH
    if not user_clicks_path.exists():
        return [], set(), 0

    with user_clicks_path.open("rb") as f:
        user_clicks: Dict[int, list[int]] = pickle.load(f)

    user_ids = sorted(int(uid) for uid in user_clicks.keys())
    user_id_set = set(user_ids)
    return user_ids[:max_suggestions], user_id_set, len(user_ids)


@app.route("/", methods=["GET"])
def index():
    user_id_suggestions, _, user_count = load_user_catalog()
    return render_template(
        "index.html",
        user_id_suggestions=user_id_suggestions,
        user_count=user_count,
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
    if not user_id:
        return render_template(
            "index.html",
            user_id_suggestions=user_id_suggestions,
            user_count=user_count,
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
            recommendations=None,
            strategy=None,
            model=None,
            hyperparameters=None,
            error=f"Failed to fetch recommendations: {exc}",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

