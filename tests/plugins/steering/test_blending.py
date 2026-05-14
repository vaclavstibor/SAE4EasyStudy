"""Tests for SAE feature-conditioned reranking and signal blending."""

import numpy as np
import torch


def _dummy_recommender():
    from server.plugins.steering.recommendation.sae_recommender import SAERecommender

    recommender = SAERecommender(model_id="TopKSAE-1024")
    recommender._loaded = True
    recommender.item_ids = [101, 102]
    recommender.item_features = torch.tensor(
        [
            [0.1, 0.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    recommender.item_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    return recommender


def test_explicit_steering_uses_steering_primary_blend():
    recommender = _dummy_recommender()

    results = recommender.get_recommendations(
        feature_adjustments={0: 1.0},
        n_items=2,
        exclude_items=[],
        allowed_ids={101, 102},
        seed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        genre_bonus=np.array([1.0, 0.0], dtype=np.float32),
        return_debug=True,
    )

    assert results["results"][0]["movie_id"] == 102
    assert results["debug"]["blend_strategy"] == "steering_primary"
    assert results["debug"]["genre_weight"] == 0.0


def test_no_steering_keeps_profile_prior_blend():
    recommender = _dummy_recommender()

    results = recommender.get_recommendations(
        feature_adjustments={},
        n_items=2,
        exclude_items=[],
        allowed_ids={101, 102},
        seed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        genre_bonus=np.array([1.0, 0.0], dtype=np.float32),
        return_debug=True,
    )

    assert results["results"][0]["movie_id"] == 101
    assert results["debug"]["blend_strategy"] == "profile_prior"
