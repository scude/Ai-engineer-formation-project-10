from __future__ import annotations

import json
import logging
import os
import sys
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict

import azure.functions as func

# ---------------------------------------------------------
# Ensure src package is importable (robuste et déterministe)
# ---------------------------------------------------------
# __file__ = .../azure_function/RecommendFunction/recommend/__init__.py
# parents[2] = repo root
FUNCTION_ROOT = Path(__file__).resolve().parents[2]
if str(FUNCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(FUNCTION_ROOT))

from src import config  # noqa: E402
from src.inference.predict import predict, serialize_recommendations  # noqa: E402


def _resolve_artifacts_dir() -> str:
    """
    Resolve ARTIFACTS_DIR to an absolute path.
    Accepts relative paths defined from the Function App root.
    """
    raw = os.getenv("ARTIFACTS_DIR", "../../artifacts")

    if os.path.isabs(raw):
        resolved = Path(raw)
    else:
        resolved = (FUNCTION_ROOT / raw).resolve()

    logging.info("Resolved ARTIFACTS_DIR=%s", resolved)
    return str(resolved)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Received request for recommendations")

    # ----------------------------
    # Parse JSON payload
    # ----------------------------
    try:
        request_json: Dict[str, Any] = req.get_json()
    except ValueError:
        logging.exception("Invalid JSON payload")
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON payload"}),
            status_code=HTTPStatus.BAD_REQUEST,
            mimetype="application/json",
        )

    if "user_id" not in request_json:
        return func.HttpResponse(
            json.dumps({"error": "'user_id' is required"}),
            status_code=HTTPStatus.BAD_REQUEST,
            mimetype="application/json",
        )

    try:
        user_id = int(request_json["user_id"])
    except (TypeError, ValueError):
        return func.HttpResponse(
            json.dumps({"error": "'user_id' must be an integer"}),
            status_code=HTTPStatus.BAD_REQUEST,
            mimetype="application/json",
        )

    logging.info("Request user_id=%s", user_id)

    # ----------------------------
    # Resolve artifacts directory
    # ----------------------------
    artifacts_dir = _resolve_artifacts_dir()

    # ----------------------------
    # Generate recommendations
    # ----------------------------
    try:
        recs, strategy = predict(user_id=user_id, artifacts_dir=artifacts_dir)
        logging.info("Returned strategy=%s, num_recs=%s", strategy, len(recs))

    except FileNotFoundError as exc:
        logging.exception("Artifacts missing")
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            mimetype="application/json",
        )
    except Exception:  # noqa: BLE001
        logging.exception("Unexpected error while generating recommendations")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            mimetype="application/json",
        )

    # ----------------------------
    # Build response
    # ----------------------------
    response_body = serialize_recommendations(
        user_id,
        recs,
        strategy,
        model_name=config.MODEL_NAME,
        hyperparameters=config.MODEL_HYPERPARAMETERS,
    )

    return func.HttpResponse(
        response_body,
        status_code=HTTPStatus.OK,
        mimetype="application/json",
    )
