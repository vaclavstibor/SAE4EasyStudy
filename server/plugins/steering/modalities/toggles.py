"""Toggle steering strategy."""

from __future__ import annotations

from typing import Any, Dict

from ..constants import Modalities
from .base import SteeringModality, SteeringResult


DEFAULT_TOGGLE_WEIGHT = 0.65


class ToggleSteering(SteeringModality):
    modality_id = Modalities.TOGGLES

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raw_adjustments = data.get("adjustments", {}) or {}
        default_weight = float(active_model.get("toggle_default_weight", conf.get("toggle_default_weight", DEFAULT_TOGGLE_WEIGHT)))
        adjustments = {}
        for feature_id, value in raw_adjustments.items():
            numeric = float(value)
            if abs(numeric) <= 0.001:
                continue
            sign = 1.0 if numeric > 0 else -1.0
            adjustments[str(feature_id)] = round(sign * default_weight, 4)
        return SteeringResult(features=[], adjustments=adjustments, metadata={"raw_adjustments": raw_adjustments})

