"""Example-based steering strategy wrapper."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID, Modalities
from ..recommendation.semantic_registry import load_semantic_clusters
from .base import SteeringModality, SteeringResult


DEFAULT_EXAMPLE_TOP_K = 6
DEFAULT_EXAMPLE_WEIGHT = 0.65


def derive_example_based_clusters(
    *,
    example_movie_ids: Iterable[int],
    recommender,
    semantic_clusters: Dict,
    top_k: int = DEFAULT_EXAMPLE_TOP_K,
    default_weight: float = DEFAULT_EXAMPLE_WEIGHT,
) -> Dict:
    example_ids = [int(movie_id) for movie_id in (example_movie_ids or []) if movie_id is not None]
    if not example_ids:
        return {"example_movie_ids": [], "clusters": [], "adjustments": {}, "matched_movie_count": 0}

    recommender.load()
    if recommender.item_features is None or recommender.item_ids is None:
        return {"example_movie_ids": example_ids, "clusters": [], "adjustments": {}, "matched_movie_count": 0}

    id_to_idx = {int(movie_id): idx for idx, movie_id in enumerate(recommender.item_ids)}
    activations = []
    for movie_id in example_ids:
        idx = id_to_idx.get(int(movie_id))
        if idx is None:
            continue
        row = recommender.item_features[idx]
        if hasattr(row, "cpu"):
            row = row.cpu().numpy()
        activations.append(row)

    if not activations:
        return {"example_movie_ids": example_ids, "clusters": [], "adjustments": {}, "matched_movie_count": 0}

    mean_activation = np.mean(np.asarray(activations), axis=0)
    cluster_rows: List[Dict] = []
    for cluster in semantic_clusters.get("clusters", []):
        neuron_ids = cluster.get("neuron_ids", [])
        values = [float(mean_activation[nid]) for nid in neuron_ids if nid < len(mean_activation)]
        if not values:
            continue
        cluster_score = float(np.mean(values))
        if cluster_score <= 0:
            continue
        strength = min(1.0, max(0.0, float(default_weight)))
        weight = min(0.95, max(0.0, strength * (1.0 + (cluster_score * 0.6))))
        cluster_rows.append(
            {
                "id": cluster.get("cluster_id"),
                "label": cluster.get("label") or str(cluster.get("cluster_id")),
                "description": cluster.get("description", ""),
                "weight": round(weight, 2),
                "direction": "boost",
                "activation_score": round(cluster_score, 4),
                "member_ids": neuron_ids,
            }
        )

    cluster_rows.sort(key=lambda row: (-row["activation_score"], row["label"].lower()))
    cluster_rows = cluster_rows[:top_k]
    return {
        "example_movie_ids": example_ids,
        "clusters": cluster_rows,
        "adjustments": {row["id"]: row["weight"] for row in cluster_rows},
        "matched_movie_count": len(activations),
    }


class ExampleSteering(SteeringModality):
    modality_id = Modalities.EXAMPLES

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        example_movie_ids = data.get("example_movie_ids") or data.get("liked_movies") or []
        active_sae_id = active_model.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
        semantic_registry = load_semantic_clusters(active_sae_id)

        from ..recommendation.sae_recommender import get_sae_recommender

        recommender = get_sae_recommender(model_id=active_sae_id)
        derived = derive_example_based_clusters(
            example_movie_ids=example_movie_ids,
            recommender=recommender,
            semantic_clusters=semantic_registry,
            top_k=int(active_model.get("example_selection_top_k", conf.get("example_selection_top_k", DEFAULT_EXAMPLE_TOP_K))),
            default_weight=float(active_model.get("example_selection_weight", conf.get("example_selection_weight", DEFAULT_EXAMPLE_WEIGHT))),
        )
        return SteeringResult(
            features=derived.get("clusters", []),
            adjustments=derived.get("adjustments", {}),
            metadata={
                "example_movie_ids": derived.get("example_movie_ids", []),
                "matched_movie_count": derived.get("matched_movie_count", 0),
                "example_top_k": int(active_model.get("example_selection_top_k", conf.get("example_selection_top_k", DEFAULT_EXAMPLE_TOP_K))),
                "example_strength": float(active_model.get("example_selection_weight", conf.get("example_selection_weight", DEFAULT_EXAMPLE_WEIGHT))),
                "matched_clusters": derived.get("clusters", []),
            },
        )

