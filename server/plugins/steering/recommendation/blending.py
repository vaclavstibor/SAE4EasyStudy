"""Signal blending policies for recommendation ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

DEFAULT_CF_WEIGHT = 10.0
DEFAULT_GENRE_WEIGHT = 5.0

# Once the user explicitly steers features, the semantic intent should drive
# ranking. Profile priors remain only as a light tie-breaker.
EXPLICIT_CF_WEIGHT = 1.5
EXPLICIT_GENRE_WEIGHT = 0.0
EXPLICIT_PRIOR_TIEBREAKER_WEIGHT = 0.05


@dataclass(frozen=True)
class RecommendationBlendPlan:
    strategy: str
    cf_weight: float
    genre_weight: float
    prior_tiebreak_weight: float


def build_blend_plan(feature_adjustments: Dict[int, float]) -> RecommendationBlendPlan:
    has_explicit_steering = any(
        abs(float(value)) > 1e-6 for value in (feature_adjustments or {}).values()
    )
    if has_explicit_steering:
        return RecommendationBlendPlan(
            strategy="steering_primary",
            cf_weight=EXPLICIT_CF_WEIGHT,
            genre_weight=EXPLICIT_GENRE_WEIGHT,
            prior_tiebreak_weight=EXPLICIT_PRIOR_TIEBREAKER_WEIGHT,
        )
    return RecommendationBlendPlan(
        strategy="profile_prior",
        cf_weight=DEFAULT_CF_WEIGHT,
        genre_weight=DEFAULT_GENRE_WEIGHT,
        prior_tiebreak_weight=0.0,
    )
