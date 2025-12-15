import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import config
from src.data.diagnostics import compute_overlap_summary, format_overlap_summary
from src.data.load_data import align_embeddings_with_clicks, load_article_embeddings, load_clicks
from src.inference.predict import load_recommender
from src.train.build_artifacts import main as build_artifacts_main


@pytest.fixture()
def temp_artifacts(monkeypatch, tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(config, "ARTICLE_EMBEDDINGS_MATRIX_PATH", artifacts_dir / "article_embeddings.npy")
    monkeypatch.setattr(config, "ARTICLE_IDS_PATH", artifacts_dir / "article_ids.npy")
    monkeypatch.setattr(config, "POPULAR_ARTICLES_PATH", artifacts_dir / "popular_articles.npy")
    monkeypatch.setattr(config, "USER_CLICKS_PATH", artifacts_dir / "user_clicks.pkl")

    return artifacts_dir


def test_load_clicks_normalizes_columns(tmp_path, monkeypatch):
    sample = pd.DataFrame({
        "user_id": [1, 2],
        "click_article_id": [10, 11],
        "click_timestamp": ["2021-01-01", "2021-01-02"],
    })
    clicks_dir = tmp_path / "clicks"
    clicks_dir.mkdir()
    sample_path = clicks_dir / "clicks_hour_000.csv"
    sample.to_csv(sample_path, index=False)

    monkeypatch.setattr(config, "CLICKS_DIR", clicks_dir)

    loaded = load_clicks()

    assert list(loaded.columns) == ["user_id", "clicked_article_id", "timestamp"]
    assert loaded["clicked_article_id"].tolist() == [10, 11]


def test_align_embeddings_with_clicks_raises_on_empty_overlap():
    embeddings_df = pd.DataFrame({"article_id": [1, 2], "embedding": [[1.0], [0.5]]})
    clicks_df = pd.DataFrame({"user_id": [1], "clicked_article_id": [99]})

    with pytest.raises(ValueError):
        align_embeddings_with_clicks(embeddings_df, clicks_df)


def test_load_article_embeddings_accepts_variant_columns(tmp_path, monkeypatch):
    # The Globo embeddings sometimes ship with generic column names
    embeddings = pd.DataFrame({"id": [1, 2], "vector": [[0.1, 0.2], [0.3, 0.4]]})
    embeddings_path = tmp_path / "articles_embeddings.pickle"
    with embeddings_path.open("wb") as f:
        pickle.dump(embeddings, f)

    monkeypatch.setattr(config, "ARTICLES_EMBEDDINGS_PATH", embeddings_path)

    loaded = load_article_embeddings()

    assert list(loaded.columns) == ["article_id", "embedding"]
    assert loaded["article_id"].tolist() == [1, 2]
    assert all(isinstance(vec, np.ndarray) for vec in loaded["embedding"])


def test_compute_overlap_summary_reports_missing_embeddings():
    clicks = pd.DataFrame({"user_id": [1, 1, 2], "clicked_article_id": [10, 20, 30]})
    embeddings = pd.DataFrame({"article_id": [20, 30], "embedding": [[0.1], [0.2]]})

    summary = compute_overlap_summary(clicks, embeddings)

    assert summary.total_clicks == 3
    assert summary.unique_users == 2
    assert summary.unique_clicked_articles == 3
    assert summary.embedding_articles == 2
    assert summary.overlap_count == 2
    assert summary.missing_clicked_articles == 1
    assert "Missing embeddings for 1 clicked articles" in format_overlap_summary(summary)


def test_end_to_end_build_and_predict(monkeypatch, tmp_path, temp_artifacts):
    clicks = pd.DataFrame({
        "user_id": [1, 1],
        "click_article_id": [10, 20],
        "click_timestamp": ["2021-01-01", "2021-01-02"],
    })
    clicks_dir = tmp_path / "clicks"
    clicks_dir.mkdir()
    clicks_path = clicks_dir / "clicks_hour_000.csv"
    clicks.to_csv(clicks_path, index=False)
    monkeypatch.setattr(config, "CLICKS_DIR", clicks_dir)

    embeddings = pd.DataFrame(
        {"article_id": [10, 20, 30], "embedding": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]}
    )
    embeddings_path = tmp_path / "articles_embeddings.pickle"
    with embeddings_path.open("wb") as f:
        pickle.dump(embeddings, f)
    monkeypatch.setattr(config, "ARTICLES_EMBEDDINGS_PATH", embeddings_path)

    build_artifacts_main()

    recommender = load_recommender(str(config.ARTIFACTS_DIR))
    recs, strategy = recommender.recommend(1)

    assert strategy == "content-based"
    returned_ids = {rec.article_id for rec in recs}
    assert 30 in returned_ids
    assert returned_ids.isdisjoint({10, 20})
