from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import requests
from flask import Flask, render_template, request

# Ensure src package is importable
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.append(str(APP_ROOT))

from src import config  # noqa: E402

app = Flask(__name__)

AZURE_FUNCTION_URL = os.getenv("AZURE_FUNCTION_URL", "http://localhost:7071/api/recommend")


def load_user_ids() -> List[int]:
    user_clicks_path = config.USER_CLICKS_PATH
    if not user_clicks_path.exists():
        return []
    with user_clicks_path.open("rb") as f:
        user_clicks: Dict[int, list[int]] = pickle.load(f)
    return sorted(user_clicks.keys())


@app.route("/", methods=["GET"])
def index():
    user_ids = load_user_ids()
    return render_template("index.html", user_ids=user_ids, recommendations=None, strategy=None, error=None)


@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id")
    user_ids = load_user_ids()
    if not user_id:
        return render_template(
            "index.html",
            user_ids=user_ids,
            recommendations=None,
            strategy=None,
            error="Please select a user ID.",
        )

    try:
        payload = {"user_id": int(user_id)}
    except ValueError:
        return render_template(
            "index.html",
            user_ids=user_ids,
            recommendations=None,
            strategy=None,
            error="Invalid user ID provided.",
        )

    try:
        response = requests.post(AZURE_FUNCTION_URL, json=payload, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"API error: {response.text}")
        data = response.json()
        recommendations = data.get("recommendations", [])
        strategy = data.get("strategy")
        return render_template(
            "index.html",
            user_ids=user_ids,
            recommendations=recommendations,
            strategy=strategy,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "index.html",
            user_ids=user_ids,
            recommendations=None,
            strategy=None,
            error=f"Failed to fetch recommendations: {exc}",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)

