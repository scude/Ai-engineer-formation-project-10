import pickle
from pathlib import Path

import pandas as pd
import pytest

from app.app import load_user_catalog
from src import config


def _write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def test_load_user_catalog_deduplicates_series(monkeypatch, tmp_path):
    payload = pd.Series([1, 1, 2, 3, 3])
    user_clicks_path = tmp_path / "user_clicks.pkl"
    _write_payload(user_clicks_path, payload)

    monkeypatch.setattr(config, "USER_CLICKS_PATH", user_clicks_path)

    suggestions, user_ids, user_count = load_user_catalog(max_suggestions=10)

    assert suggestions == [1, 2, 3]
    assert user_ids == {1, 2, 3}
    assert user_count == 3


def test_load_user_catalog_uses_user_id_column(monkeypatch, tmp_path):
    payload = pd.DataFrame(
        {
            "user_id": [10, 10, 11],
            "clicked_article_id": [101, 102, 103],
            "timestamp": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        }
    )
    user_clicks_path = tmp_path / "user_clicks.pkl"
    _write_payload(user_clicks_path, payload)

    monkeypatch.setattr(config, "USER_CLICKS_PATH", user_clicks_path)

    suggestions, user_ids, user_count = load_user_catalog(max_suggestions=5)

    assert suggestions == [10, 11]
    assert user_ids == {10, 11}
    assert user_count == 2


def test_load_user_catalog_handles_alt_user_column(monkeypatch, tmp_path):
    payload = pd.DataFrame(
        {
            "userId": [99, 100, 100],
            "clicked_article_id": [501, 502, 503],
        }
    )
    user_clicks_path = tmp_path / "user_clicks.pkl"
    _write_payload(user_clicks_path, payload)

    monkeypatch.setattr(config, "USER_CLICKS_PATH", user_clicks_path)

    suggestions, user_ids, user_count = load_user_catalog(max_suggestions=5)

    assert suggestions == [99, 100]
    assert user_ids == {99, 100}
    assert user_count == 2


def test_load_user_catalog_errors_when_no_user_column(monkeypatch, tmp_path):
    payload = pd.DataFrame(
        {
            "clicked_article_id": [101, 102],
            "timestamp": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        }
    )
    user_clicks_path = tmp_path / "user_clicks.pkl"
    _write_payload(user_clicks_path, payload)

    monkeypatch.setattr(config, "USER_CLICKS_PATH", user_clicks_path)

    with pytest.raises(ValueError):
        load_user_catalog(max_suggestions=5)
