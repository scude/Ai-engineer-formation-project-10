import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.lightfm_item2item import (
    CONTEXT_COLUMNS,
    LightFMApproximator,
    build_interaction_matrices,
    precompute_item_neighbors,
    score_from_neighbors,
    session_weight_from_size,
)


def test_session_weight_from_size_applies_log_rule():
    series = pd.Series([1, 2, 10, None])
    weights = session_weight_from_size(series)
    expected = 1.0 / np.log1p(series.fillna(1.0).clip(lower=1.0))
    assert np.allclose(weights, expected.values.astype(np.float32))


def test_interaction_matrices_include_user_features():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "article_id": [10, 11, 10, 12],
            "session_size": [2, 2, 3, 3],
            "click_environment": ["web", "web", "app", "app"],
            "click_deviceGroup": ["mobile", "mobile", "mobile", "mobile"],
            "click_os": ["ios", "ios", "android", "android"],
            "click_country": ["fr", "fr", "fr", "fr"],
            "click_region": ["idf", "idf", "idf", "idf"],
            "click_referrer_type": ["direct", "direct", "direct", "direct"],
        }
    )

    interactions, sample_weight, user_features, item_ids = build_interaction_matrices(
        df, CONTEXT_COLUMNS, use_user_features=True
    )

    assert interactions.shape == (2, 3)
    assert sample_weight.shape == interactions.shape
    assert user_features is not None
    assert user_features.shape[1] > 0
    assert sorted(item_ids) == [10, 11, 12]


def test_neighbor_scoring_and_determinism():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    item_ids = [1, 2, 3]
    neighbors = precompute_item_neighbors(embeddings, item_ids, top_n=2)

    scores = score_from_neighbors([1], neighbors, seen={1})
    assert 2 in scores and 3 in scores

    model_a = LightFMApproximator(n_components=4, epochs=2, random_state=7)
    model_b = LightFMApproximator(n_components=4, epochs=2, random_state=7)
    interactions = sparse.csr_matrix(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32))
    weights = sparse.csr_matrix(np.ones_like(interactions.toarray()))

    model_a.fit(interactions, sample_weight=weights)
    model_b.fit(interactions, sample_weight=weights)

    assert np.allclose(model_a.item_embeddings, model_b.item_embeddings)

    _, items_a = model_a.get_item_representations()
    _, items_b = model_b.get_item_representations()
    assert np.allclose(items_a, items_b)
