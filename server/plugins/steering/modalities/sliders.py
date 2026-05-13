"""Slider steering strategy and slider refresh logic."""

from __future__ import annotations

from typing import Any, Dict

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID, Modalities
from ..recommendation.features import get_sae_features, personalized_features
from ..approach_state import get_approach_id_map, get_approach_token_set, set_approach_token_set
from ..recommendation.semantic_registry import is_near_duplicate_label, normalize_label
from .base import SteeringModality, SteeringResult


SLIDER_AMPLIFICATION = 2.0


class SliderSteering(SteeringModality):
    
    modality_id = Modalities.SLIDERS

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raw_adjustments = data.get("adjustments", {}) or {}
        adjustments = {
            str(feature_id): round(float(value) * SLIDER_AMPLIFICATION, 4)
            for feature_id, value in raw_adjustments.items()
            if abs(float(value)) > 0.001
        }
        return SteeringResult(features=[], adjustments=adjustments, metadata={"raw_adjustments": raw_adjustments})


def compute_updated_sliders(
    current_features: list,
    cumulative_adjustments: dict,
    liked_movie_ids: list,
    model_id: str = None,
    num_sliders: int = 21,
    phase_idx: int = 0,
) -> list:
    touched_ids = set()
    for feature_id, value in cumulative_adjustments.items():
        if feature_id.startswith("cluster_") and abs(float(value)) > 0.001:
            touched_ids.add(feature_id)

    shown_ids = get_approach_token_set("shown_sliders_per_phase", phase_idx)
    steered_ids = get_approach_token_set("steered_sliders_per_phase", phase_idx)

    if touched_ids:
        steered_ids.update({str(cluster_id) for cluster_id in touched_ids})
        set_approach_token_set("steered_sliders_per_phase", phase_idx, steered_ids)

    last_shown_map = get_approach_id_map("last_shown_movies_per_phase")
    last_shown_phase = last_shown_map.get(str(int(phase_idx)), [])
    profile_movies = sorted({int(mid) for mid in last_shown_phase if mid is not None})

    profile_pool = []
    if profile_movies:
        profile_pool = personalized_features(
            selected_movies=profile_movies,
            model_id=model_id or DEFAULT_TOPK_SAE_MODEL_ID,
            num_sliders=max(num_sliders * 4, num_sliders + 24),
        )
    global_pool = get_sae_features(top_k=num_sliders * 8, model_id=model_id)

    selected = []
    used_ids = set()
    seen_labels = set()
    source_counts = {"exploit": 0, "explore": 0}

    def append_from_pool(pool: list, target_size: int, source: str, allow_shown: bool = False):
        if not pool:
            return
        for feature in pool:
            if len(selected) >= target_size:
                break
            cluster_id = str(feature.get("id"))
            if cluster_id in used_ids:
                continue
            if (not allow_shown) and cluster_id in shown_ids:
                continue
            if cluster_id in steered_ids:
                continue
            if is_near_duplicate_label(feature.get("label", ""), seen_labels):
                continue
            selected.append(feature)
            used_ids.add(cluster_id)
            seen_labels.add(normalize_label(feature.get("label", "")))
            source_counts[source] = source_counts.get(source, 0) + 1

    append_from_pool(profile_pool, num_sliders, source="exploit", allow_shown=False)
    append_from_pool(profile_pool, num_sliders, source="exploit", allow_shown=False)
    if len(selected) < num_sliders:
        append_from_pool(global_pool, num_sliders, source="explore", allow_shown=True)
    if len(selected) < num_sliders:
        append_from_pool(profile_pool, num_sliders, source="exploit", allow_shown=True)

    shown_ids.update({str(feature.get("id")) for feature in selected if feature.get("id") is not None})
    set_approach_token_set("shown_sliders_per_phase", phase_idx, shown_ids)

    print(
        f"[compute_updated_sliders] phase={phase_idx} touched={len(touched_ids)} "
        f"shown_pool={len(shown_ids)} steered_pool={len(steered_ids)} returned={len(selected)} "
        f"(exploit={source_counts.get('exploit', 0)}, explore={source_counts.get('explore', 0)})"
    )
    return selected[:num_sliders]
