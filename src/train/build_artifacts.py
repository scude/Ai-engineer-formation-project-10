from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from src import config
from src.data.load_data import load_clicks
from src.data.preprocess import prepare_popular_articles, prepare_user_clicks
from src.models.covisit import build_pure_covisit_similarity, compute_normalized_popularity


def main() -> None:
    artifacts_dir: Path = config.ARTIFACTS_DIR
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clicks data...")
    clicks = load_clicks()

    print("Preparing user clicks map...")
    user_clicks = prepare_user_clicks(clicks)
    with config.USER_CLICKS_PATH.open("wb") as f:
        pickle.dump(user_clicks, f)

    print("Computing popularity ranking and scores...")
    popular_articles = prepare_popular_articles(clicks)
    np.save(config.POPULAR_ARTICLES_PATH, popular_articles)
    popularity_scores = compute_normalized_popularity(clicks)
    with config.POPULARITY_SCORES_PATH.open("wb") as f:
        pickle.dump(popularity_scores, f)

    print("Building co-visitation similarity graph...")
    similarity = build_pure_covisit_similarity(
        clicks,
        top_n_neighbors=config.MODEL_HYPERPARAMETERS["covisit_top_n_neighbors"],
        metric=config.DEFAULT_SIMILARITY_METRIC,
    )
    with config.COVISIT_SIMILARITY_PATH.open("wb") as f:
        pickle.dump(similarity, f)

    print(
        f"Artifacts created in {artifacts_dir}:",
        f"\n- {config.USER_CLICKS_PATH.name}",
        f"\n- {config.POPULAR_ARTICLES_PATH.name}",
        f"\n- {config.POPULARITY_SCORES_PATH.name}",
        f"\n- {config.COVISIT_SIMILARITY_PATH.name}",
    )


if __name__ == "__main__":
    main()

