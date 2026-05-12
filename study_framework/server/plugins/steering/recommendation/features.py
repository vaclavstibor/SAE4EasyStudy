"""Feature selection pipeline for steering controls."""

import numpy as np

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID
from .semantic_registry import load_semantic_clusters
from ..study_config import normalize_feature_selection_algorithm


def select_cluster_features(model_id: str = None, top_k: int = 21) -> list:
    import re as _re

    from .llm_labels import get_llm_labels

    stop_words = {
        "the",
        "and",
        "of",
        "in",
        "a",
        "an",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "its",
        "that",
        "this",
        "but",
        "or",
        "as",
    }

    effective_model_id = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    semantic_clusters = load_semantic_clusters(effective_model_id)
    neuron_stats = get_llm_labels(model_id=effective_model_id)

    candidates = []
    for cluster in semantic_clusters["clusters"]:
        label = cluster["label"]
        if not label:
            continue
        neuron_ids = cluster["neuron_ids"]
        total_act = 0
        selectivities = []
        for neuron_id in neuron_ids:
            info = neuron_stats.get(neuron_id, {})
            total_act += info.get("activation_count", 0)
            selectivity = info.get("selectivity", 0)
            if selectivity > 0:
                selectivities.append(selectivity)
        mean_selectivity = np.mean(selectivities) if selectivities else 0
        if total_act < 50 or mean_selectivity < 0.3:
            continue
        candidates.append(
            {
                "cluster_id": cluster["cluster_id"],
                "label": label,
                "description": cluster.get("description", ""),
                "neuron_ids": neuron_ids,
                "score": mean_selectivity * np.log(total_act + 1),
                "total_act": total_act,
            }
        )

    candidates.sort(key=lambda x: -x["score"])

    selected = []
    used_words = {}
    max_word_uses = 2

    for candidate in candidates:
        if len(selected) >= top_k:
            break
        words = {
            word
            for word in _re.split(r"[\s·\-–—/,&]+", candidate["label"].lower())
            if len(word) > 2 and word not in stop_words
        }
        if any(used_words.get(word, 0) >= max_word_uses for word in words):
            continue
        selected.append(candidate)
        for word in words:
            used_words[word] = used_words.get(word, 0) + 1

    features = []
    for item in selected:
        features.append(
            {
                "id": item["cluster_id"],
                "label": item["label"],
                "category": "latent",
                "description": item["description"],
                "member_ids": item["neuron_ids"],
                "activation": 0.5,
                "movie_count": item["total_act"],
            }
        )
    features.sort(key=lambda feature: -feature["movie_count"])
    return features


def get_sae_features(top_k: int = 21, model_id: str = None) -> list:
    effective_model_id = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    features = select_cluster_features(model_id=effective_model_id, top_k=top_k)
    print(f"[get_sae_features] Selected {len(features)} cluster features")
    return features


def personalized_features(selected_movies: list, model_id: str = None, num_sliders: int = 21) -> list:
    import re as _re
    import torch as _torch

    from .sae_recommender import get_sae_recommender

    if not selected_movies:
        return get_sae_features(top_k=num_sliders, model_id=model_id)

    effective_model_id = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    recommender = get_sae_recommender(model_id=effective_model_id)
    recommender.load()

    if recommender.item_features is None or recommender.item_ids is None:
        return get_sae_features(top_k=num_sliders, model_id=model_id)

    id_to_idx = {int(mid): i for i, mid in enumerate(recommender.item_ids)}
    activations = []
    for movie_id in selected_movies:
        idx = id_to_idx.get(int(movie_id))
        if idx is not None:
            activation = recommender.item_features[idx]
            if isinstance(activation, _torch.Tensor):
                activation = activation.cpu().numpy()
            activations.append(activation)

    if not activations:
        return get_sae_features(top_k=num_sliders, model_id=model_id)

    mean_act = np.mean(activations, axis=0)
    print(f"[personalized_features] {len(activations)}/{len(selected_movies)} movies matched")

    semantic_clusters = load_semantic_clusters(effective_model_id)
    stop_words = {
        "the",
        "and",
        "of",
        "in",
        "a",
        "an",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "its",
        "that",
        "this",
        "but",
        "or",
        "as",
    }

    candidates = []
    for cluster in semantic_clusters["clusters"]:
        neuron_ids = cluster["neuron_ids"]
        cluster_score = float(np.mean([mean_act[n] for n in neuron_ids if n < len(mean_act)]))
        if cluster_score <= 0:
            continue
        total_act = sum(mean_act[n] for n in neuron_ids if n < len(mean_act))
        candidates.append(
            {
                "cluster_id": cluster["cluster_id"],
                "label": cluster["label"],
                "description": cluster.get("description", ""),
                "neuron_ids": neuron_ids,
                "score": cluster_score,
                "total_act": int(total_act * 100),
            }
        )

    candidates.sort(key=lambda x: -x["score"])

    selected = []
    used_words = {}
    max_word_uses = 2

    for candidate in candidates:
        if len(selected) >= num_sliders:
            break
        words = {
            word
            for word in _re.split(r"[\s·\-–—/,&]+", candidate["label"].lower())
            if len(word) > 2 and word not in stop_words
        }
        if any(used_words.get(word, 0) >= max_word_uses for word in words):
            continue
        selected.append(candidate)
        for word in words:
            used_words[word] = used_words.get(word, 0) + 1

    if selected:
        print(
            f"[personalized_features] {len(selected)} clusters "
            f"(top: {selected[0]['label']} score={selected[0]['score']:.4f})"
        )

    features = []
    for item in selected:
        features.append(
            {
                "id": item["cluster_id"],
                "label": item["label"],
                "category": "latent",
                "description": item["description"],
                "member_ids": item["neuron_ids"],
                "activation": 0.5,
                "movie_count": item["total_act"],
            }
        )
    return features


def select_slider_features(selected_movies: list, conf: dict, active_model_cfg: dict, num_sliders: int) -> list:
    algorithm = normalize_feature_selection_algorithm(
        active_model_cfg.get("feature_selection_algorithm", conf.get("feature_selection_algorithm"))
    )
    active_sae_model_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    if algorithm == "global_label_topk":
        return get_sae_features(top_k=num_sliders, model_id=active_sae_model_id)
    return personalized_features(
        selected_movies=selected_movies,
        model_id=active_sae_model_id,
        num_sliders=num_sliders,
    )

