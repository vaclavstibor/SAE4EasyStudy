"""Text steering strategy wrapper."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID, Modalities
from ..recommendation.semantic_registry import load_semantic_clusters
from .base import SteeringModality, SteeringResult

DEFAULT_TEXT_TOP_K = 6
DEFAULT_TEXT_WEIGHT = 0.55

_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_STOP_WORDS = {
    "and",
    "the",
    "with",
    "from",
    "that",
    "this",
    "show",
    "movies",
    "movie",
    "films",
    "film",
    "something",
    "like",
    "more",
    "less",
}
_POSITIVE_HINTS = (
    "more",
    "boost",
    "increase",
    "prefer",
    "want",
    "love",
    "enjoy",
    "i like",
    "i love",
    "similar to",
)
_NEGATIVE_HINTS = (
    "less",
    "avoid",
    "without",
    "exclude",
    "fewer",
    "remove",
    "not ",
    "no ",
    "never",
    "don't",
    "dont",
    "do not",
    "stop",
    "hate",
    "dislike",
    "but not",
    "but only",
    "but without",
    "i don't like",
    "i dont like",
    "i hate",
)
_INTENSITY_HIGH = (
    "much more",
    "way more",
    "a lot more",
    "strongly",
    "definitely",
    "really really",
    "much less",
    "way less",
    "a lot less",
)
_INTENSITY_LOW = (
    "slightly",
    "a bit",
    "a little",
    "somewhat",
    "kind of",
    "sort of",
)
_SEGMENT_BOUNDARY_RE = re.compile(r"[.;]|\bbut\b|\bhowever\b")


def _tokenize(text: str) -> List[str]:
    return [token for token in _WORD_RE.findall((text or "").lower()) if token not in _STOP_WORDS]


def _intensity_multiplier(text_lower: str) -> float:
    if any(marker in text_lower for marker in _INTENSITY_HIGH):
        return 1.35
    if any(marker in text_lower for marker in _INTENSITY_LOW):
        return 0.65
    return 1.0


def _split_query(query: str) -> List[Dict]:
    chunks = _SEGMENT_BOUNDARY_RE.split(query or "")
    segments = []
    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        lowered = text.lower()
        direction = -1 if any(marker in lowered for marker in _NEGATIVE_HINTS) else 1
        segments.append(
            {
                "text": text,
                "direction": direction,
                "intensity": _intensity_multiplier(lowered),
                "tokens": _tokenize(text),
            }
        )
    if not segments and (query or "").strip():
        lowered = query.strip().lower()
        segments.append(
            {
                "text": query.strip(),
                "direction": 1,
                "intensity": _intensity_multiplier(lowered),
                "tokens": _tokenize(query),
            }
        )
    return segments


def _score_cluster(segment: Dict, cluster: Dict) -> float:
    label = (cluster.get("label") or "").lower()
    description = (cluster.get("description") or "").lower()
    haystack = f"{label} {description}".strip()
    if not haystack:
        return 0.0
    score = 0.0
    phrase = segment.get("text", "").lower().strip()
    if phrase and phrase in haystack:
        score += 3.0
    tokens = segment.get("tokens") or []
    if not tokens:
        return score
    label_tokens = set(_tokenize(label))
    desc_tokens = set(_tokenize(description))
    for token in tokens:
        if token in label_tokens:
            score += 2.5
        elif token in desc_tokens:
            score += 1.25
        elif token in haystack:
            score += 0.75
    coverage = len([token for token in tokens if token in haystack]) / max(len(tokens), 1)
    score += coverage
    return score


def resolve_text_query_to_clusters(
    query: str,
    *,
    clusters: Iterable[Dict],
    top_k: int = DEFAULT_TEXT_TOP_K,
    default_weight: float = DEFAULT_TEXT_WEIGHT,
) -> Dict:
    segments = _split_query(query)
    if not segments:
        return {"query": query, "segments": [], "clusters": [], "adjustments": {}}

    scored = []
    for cluster in clusters or []:
        total_score = 0.0
        direction_votes = []
        intensity_sum = 0.0
        contributing_segments = 0
        for segment in segments:
            score = _score_cluster(segment, cluster)
            if score <= 0:
                continue
            total_score += score
            direction_votes.append(segment["direction"])
            intensity_sum += float(segment.get("intensity") or 1.0)
            contributing_segments += 1
        if total_score <= 0:
            continue
        avg_direction = 1 if not direction_votes else (1 if sum(direction_votes) >= 0 else -1)
        avg_intensity = (intensity_sum / contributing_segments) if contributing_segments else 1.0
        normalized_weight = min(0.95, max(0.25, default_weight + (total_score / 10.0)))
        normalized_weight = min(0.95, max(0.1, normalized_weight * avg_intensity))
        cluster_id = cluster.get("cluster_id") or cluster.get("id")
        scored.append(
            {
                "id": cluster_id,
                "label": cluster.get("label") or str(cluster_id),
                "description": cluster.get("description", ""),
                "weight": round(normalized_weight * avg_direction, 2),
                "direction": "boost" if avg_direction >= 0 else "suppress",
                "match_score": round(total_score, 3),
                "intensity": round(avg_intensity, 2),
                "member_ids": cluster.get("member_ids") or cluster.get("neuron_ids") or [],
            }
        )

    scored.sort(key=lambda row: (-abs(row["weight"]), -row["match_score"], row["label"].lower()))
    top_clusters = scored[:top_k]
    adjustments = {row["id"]: row["weight"] for row in top_clusters}
    return {
        "query": query,
        "segments": segments,
        "clusters": top_clusters,
        "adjustments": adjustments,
    }


class TextSteering(SteeringModality):
    modality_id = Modalities.TEXT

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        query = (data.get("query") or "").strip()
        active_sae_id = active_model.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
        semantic_registry = load_semantic_clusters(active_sae_id)
        resolved = resolve_text_query_to_clusters(
            query,
            clusters=semantic_registry.get("clusters", []),
            top_k=int(
                active_model.get(
                    "text_steering_top_k", conf.get("text_steering_top_k", DEFAULT_TEXT_TOP_K)
                )
            ),
            default_weight=float(
                active_model.get(
                    "text_steering_weight", conf.get("text_steering_weight", DEFAULT_TEXT_WEIGHT)
                )
            ),
        )
        return SteeringResult(
            features=resolved.get("clusters", []),
            adjustments=resolved.get("adjustments", {}),
            metadata={"query": query, "segments": resolved.get("segments", [])},
        )
