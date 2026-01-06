from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from surprise import Dataset, Reader, SVDpp

from src import config
from src.data.load_data import load_clicks
from src.data.preprocess import prepare_embeddings, prepare_popular_articles, prepare_user_clicks
from src.models.covisit import compute_normalized_popularity


def _prepare_weighted_ratings(clicks):
    ratings = clicks[["user_id", "clicked_article_id", "session_size"]].copy()
    ratings["session_size"] = pd.to_numeric(ratings["session_size"], errors="coerce").fillna(1.0)
    grouped = (
        ratings.groupby(["user_id", "clicked_article_id"], as_index=False)
        .agg(avg_session_size=("session_size", "mean"))
    )
    grouped["rating"] = config.MODEL_HYPERPARAMETERS["rating_base"] + np.log1p(
        grouped["avg_session_size"].astype(float)
    )
    return grouped[["user_id", "clicked_article_id", "rating"]]


def _train_surprise_svdpp(clicks):
    ratings = _prepare_weighted_ratings(clicks)

    max_rating = float(ratings["rating"].max()) if not ratings.empty else 1.0
    reader = Reader(rating_scale=(0, max_rating))
    trainset = Dataset.load_from_df(ratings, reader).build_full_trainset()

    algo = SVDpp(
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

    print("Preparing PCA-reduced embeddings...")
    article_ids, embeddings = prepare_embeddings(clicks)
    max_components = min(embeddings.shape[0], embeddings.shape[1])
    n_components = min(config.PCA_COMPONENTS, max_components)
    if n_components < config.PCA_COMPONENTS:
        print(
            "Warning: PCA components reduced from"
            f" {config.PCA_COMPONENTS} to {n_components} due to limited embeddings"
            f" shape={embeddings.shape}."
        )
    if n_components < 1:
        raise ValueError(
            "Embeddings matrix has insufficient samples/features for PCA. "
            "Check that articles_embeddings.pickle matches clicked_article_id values."
        )
    pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")
    embeddings_reduced = pca.fit_transform(embeddings)
    np.save(config.ARTICLE_IDS_PATH, article_ids)
    np.save(config.ARTICLE_EMBEDDINGS_MATRIX_PATH, embeddings_reduced.astype(np.float32))

    print("Computing popularity ranking and scores...")
    popular_articles = prepare_popular_articles(clicks)
    np.save(config.POPULAR_ARTICLES_PATH, popular_articles)
    popularity_scores = compute_normalized_popularity(clicks)
    with config.POPULARITY_SCORES_PATH.open("wb") as f:
        pickle.dump(popularity_scores, f)

    print("Training Surprise SVD++ model with session weighting...")
    svd_model, item_ids = _train_surprise_svdpp(clicks)
    with config.SURPRISE_MODEL_PATH.open("wb") as f:
        pickle.dump(svd_model, f)
    np.save(config.SURPRISE_ITEMS_PATH, item_ids)

    print(
        f"Artifacts created in {artifacts_dir}:",
        f"\n- {config.USER_CLICKS_PATH.name}",
        f"\n- {config.ARTICLE_IDS_PATH.name}",
        f"\n- {config.ARTICLE_EMBEDDINGS_MATRIX_PATH.name}",
        f"\n- {config.POPULAR_ARTICLES_PATH.name}",
        f"\n- {config.POPULARITY_SCORES_PATH.name}",
        f"\n- {config.SURPRISE_MODEL_PATH.name}",
        f"\n- {config.SURPRISE_ITEMS_PATH.name}",
    )


if __name__ == "__main__":
    main()
