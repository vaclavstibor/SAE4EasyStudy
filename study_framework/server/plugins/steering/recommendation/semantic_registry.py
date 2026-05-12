"""Semantic cluster registry and label deduplication helpers."""

import json
import os
import re

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID, FUZZY_LABEL_JACCARD_THRESHOLD

SEMANTIC_CLUSTERS_CACHE = {}


def normalize_label(label: str) -> str:
    return " ".join(label.lower().split())


def label_word_set(label: str) -> set:
    return {w for w in re.split(r"[\s·\-–—/,&]+", label.lower()) if len(w) > 2}


def is_near_duplicate_label(new_label: str, existing_labels: set) -> bool:
    norm = normalize_label(new_label)
    if norm in existing_labels:
        return True
    new_words = label_word_set(new_label)
    if not new_words:
        return False
    for existing in existing_labels:
        ex_words = label_word_set(existing)
        if not ex_words:
            continue
        intersection = new_words & ex_words
        union = new_words | ex_words
        if len(intersection) / len(union) >= FUZZY_LABEL_JACCARD_THRESHOLD:
            return True
    return False


def expand_feature_adjustments(raw_adjustments: dict, cluster_map: dict = None) -> dict:
    feature_adjustments = {}
    cluster_map = cluster_map or {}
    for key, val in (raw_adjustments or {}).items():
        delta = float(val)
        if abs(delta) < 0.0001:
            continue
        neuron_ids = cluster_map.get(key)
        if neuron_ids:
            for nid in neuron_ids:
                skey = str(nid)
                feature_adjustments[skey] = feature_adjustments.get(skey, 0.0) + delta
        else:
            feature_adjustments[key] = feature_adjustments.get(key, 0.0) + delta
    return feature_adjustments


def load_semantic_clusters(model_id: str = None) -> dict:
    resolved = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    if resolved in SEMANTIC_CLUSTERS_CACHE:
        return SEMANTIC_CLUSTERS_CACHE[resolved]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    path = os.path.join(data_dir, f"semantic_merged_{resolved}.json")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Semantic clusters not found: {path}. Copy from labeling/artifacts/."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    clusters = []
    cluster_map = {}
    neuron_to_cluster = {}
    for cluster in raw.get("clusters", []):
        cluster_id = cluster["cluster_id"]
        neuron_ids = [int(n) for n in cluster["neuron_ids"]]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": cluster["label"],
                "description": cluster.get("description", ""),
                "neuron_ids": neuron_ids,
                "support": cluster.get("support", len(neuron_ids)),
            }
        )
        cluster_map[cluster_id] = neuron_ids
        for neuron_id in neuron_ids:
            neuron_to_cluster[neuron_id] = cluster_id

    result = {
        "clusters": clusters,
        "cluster_map": cluster_map,
        "neuron_to_cluster": neuron_to_cluster,
    }
    SEMANTIC_CLUSTERS_CACHE[resolved] = result
    print(
        f"[clusters] Loaded {len(clusters)} clusters "
        f"({sum(len(v) for v in cluster_map.values())} neurons) from {path}"
    )
    return result
