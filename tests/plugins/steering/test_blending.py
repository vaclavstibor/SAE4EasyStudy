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


# Reranking strategies (see equations.md §10).


def _recommender_with_decoder():
    """Same as ``_dummy_recommender`` but also wires a fake SAE model so
    that ``_decode_sae_profile_to_embedding_space`` returns a real vector.

    The decoder is an identity-like matrix between two equal-dim spaces
    so the test can reason about the perturbation direction directly.
    """
    from server.plugins.steering.recommendation.sae_recommender import (
        SAERecommender,
        TopKSAE,
    )

    recommender = SAERecommender(model_id="TopKSAE-1024")
    recommender._loaded = True
    recommender.item_ids = [101, 102, 103]
    # item 101: aligns with feature 0 only (sae_score = 1.0 for {0: 1})
    # item 102: aligns with feature 0 strongly (sae_score = 2.0 for {0: 1})
    # item 103: aligns with feature 1 (orthogonal to feature 0)
    recommender.item_features = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    # Item embeddings: 101 and 103 sit close to the seed direction;
    # 102 sits further away. With seed=(1, 0), pure CF would rank
    # 101 > 102 > 103. Steering toward feature 0 should bring 102 up.
    recommender.item_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.6, 0.8],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    # Minimal SAE module with a decoder that maps feature 0 →
    # direction (1, 0) and feature 1 → direction (0, 1) so the
    # latent perturbation math has a known target.
    sae = TopKSAE(input_dim=2, embedding_dim=2, k=1)
    with torch.no_grad():
        sae.decoder_w.copy_(torch.eye(2))
        sae.decoder_b.zero_()
    recommender.sae_model = sae
    return recommender


def test_feature_conditioned_strategy_is_default():
    """Calling without ``reranking_strategy`` keeps the historical math."""
    recommender = _recommender_with_decoder()

    results = recommender.get_recommendations(
        feature_adjustments={0: 1.0},
        n_items=3,
        exclude_items=[],
        allowed_ids={101, 102, 103},
        seed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        genre_bonus=None,
        return_debug=True,
    )

    assert results["debug"]["reranking_strategy"] == "feature-conditioned"
    # With a positive feature-0 adjustment, item 102 (highest sae_score)
    # should rank first.
    assert results["results"][0]["movie_id"] == 102


def test_latent_perturbation_strategy_rotates_seed_and_drops_additive_term():
    """``latent-perturbation`` must NOT add the SAE term to scores.

    Each result's ``steering_score`` must be 0 because the additive
    branch is bypassed; the influence shows up as a different cf_score
    (the seed has been rotated by α · decoder(sae_profile)).
    """
    recommender = _recommender_with_decoder()

    baseline = recommender.get_recommendations(
        feature_adjustments={},
        n_items=3,
        exclude_items=[],
        allowed_ids={101, 102, 103},
        seed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        return_debug=True,
    )
    rotated = recommender.get_recommendations(
        feature_adjustments={1: 1.0},  # push seed toward feature 1 (item 103)
        n_items=3,
        exclude_items=[],
        allowed_ids={101, 102, 103},
        seed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        reranking_strategy="latent-perturbation",
        reranking_params={"latent_perturbation_alpha": 0.9},
        return_debug=True,
    )

    assert rotated["debug"]["reranking_strategy"] == "latent-perturbation"
    # No additive SAE term in any item's score.
    for item in rotated["results"]:
        assert item["steering_score"] == 0.0
    # The perturbation has changed the cf scores compared to the
    # un-steered baseline.
    base_cf_by_id = {r["movie_id"]: r["cf_score"] for r in baseline["results"]}
    rotated_cf_by_id = {r["movie_id"]: r["cf_score"] for r in rotated["results"]}
    assert rotated_cf_by_id != base_cf_by_id


def test_constrained_subset_strategy_filters_out_non_conformant_items():
    """Items whose SAE score is below the threshold get -inf scores."""
    recommender = _recommender_with_decoder()

    results = recommender.get_recommendations(
        feature_adjustments={0: 1.0},  # only items 101 (sae=1) and 102 (sae=2) match
        n_items=3,
        exclude_items=[],
        allowed_ids={101, 102, 103},
        seed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        reranking_strategy="constrained-subset",
        reranking_params={"constrained_subset_tau": 0.5},  # τ × max(2.0) = 1.0
        return_debug=True,
    )

    assert results["debug"]["reranking_strategy"] == "constrained-subset"
    returned_ids = [r["movie_id"] for r in results["results"]]
    # Item 103 (sae=0) falls below the τ threshold → must not appear.
    assert 103 not in returned_ids
    assert set(returned_ids) == {101, 102}
    # No additive steering term in this strategy either.
    for item in results["results"]:
        assert item["steering_score"] == 0.0
    assert results["debug"]["constrained_subset_survivors"] == 2


def test_constrained_subset_strategy_falls_back_when_no_survivors():
    """If τ × max-sae filters out everything, ranking falls back to base."""
    recommender = _recommender_with_decoder()

    results = recommender.get_recommendations(
        # An adjustment that targets a feature no item uses → no
        # positive SAE scores anywhere → mask is empty.
        feature_adjustments={},
        n_items=3,
        exclude_items=[],
        allowed_ids={101, 102, 103},
        seed_embedding=np.array([1.0, 0.0], dtype=np.float32),
        reranking_strategy="constrained-subset",
        reranking_params={"constrained_subset_tau": 0.99},
        return_debug=True,
    )

    assert len(results["results"]) == 3  # nothing was filtered out
    assert results["debug"]["reranking_strategy"] == "constrained-subset"
