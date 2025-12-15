from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from src import config
from src.data.load_data import load_clicks_sample
from src.data.preprocess import prepare_embeddings, prepare_popular_articles, prepare_user_clicks


def main() -> None:
    artifacts_dir: Path = config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clicks_sample.csv...")
    clicks = load_clicks_sample()

    print("Preparing user clicks map...")
    user_clicks = prepare_user_clicks(clicks)
    with config.USER_CLICKS_PATH.open("wb") as f:
        pickle.dump(user_clicks, f)

    print("Computing popularity ranking...")
    popular_articles = prepare_popular_articles(clicks)
    np.save(config.POPULAR_ARTICLES_PATH, popular_articles)

    print("Loading article embeddings...")
    article_ids, article_embeddings = prepare_embeddings(clicks)
    np.save(config.ARTICLE_IDS_PATH, article_ids)
    np.save(config.ARTICLE_EMBEDDINGS_MATRIX_PATH, article_embeddings)

    print(
        f"Artifacts created in {artifacts_dir}:",
        f"\n- {config.USER_CLICKS_PATH.name}",
        f"\n- {config.POPULAR_ARTICLES_PATH.name}",
        f"\n- {config.ARTICLE_IDS_PATH.name}",
        f"\n- {config.ARTICLE_EMBEDDINGS_MATRIX_PATH.name}",
    )


if __name__ == "__main__":
    main()

