from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from surprise import Dataset, Reader, SVD

from src import config
from src.data.load_data import load_clicks
from src.data.preprocess import prepare_popular_articles, prepare_user_clicks
from src.models.covisit import compute_normalized_popularity


def _train_surprise_svd(clicks):
    ratings = clicks[["user_id", "clicked_article_id"]].copy()
    ratings["rating"] = 1.0

    reader = Reader(rating_scale=(0, 1))
    trainset = Dataset.load_from_df(ratings, reader).build_full_trainset()

    algo = SVD(
        n_factors=config.MODEL_HYPERPARAMETERS["n_factors"],
        reg_all=config.MODEL_HYPERPARAMETERS["reg_all"],
        lr_all=config.MODEL_HYPERPARAMETERS["lr_all"],
        random_state=42,
    )
    algo.fit(trainset)

    item_ids = np.array([int(trainset.to_raw_iid(iid)) for iid in trainset.all_items()])
    return algo, item_ids


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

    print("Training Surprise SVD model...")
    svd_model, item_ids = _train_surprise_svd(clicks)
    with config.SURPRISE_MODEL_PATH.open("wb") as f:
        pickle.dump(svd_model, f)
    np.save(config.SURPRISE_ITEMS_PATH, item_ids)

    print(
        f"Artifacts created in {artifacts_dir}:",
        f"\n- {config.USER_CLICKS_PATH.name}",
        f"\n- {config.POPULAR_ARTICLES_PATH.name}",
        f"\n- {config.POPULARITY_SCORES_PATH.name}",
        f"\n- {config.SURPRISE_MODEL_PATH.name}",
        f"\n- {config.SURPRISE_ITEMS_PATH.name}",
    )


if __name__ == "__main__":
    main()

