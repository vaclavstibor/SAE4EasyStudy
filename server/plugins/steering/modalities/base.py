"""Base protocol for SAE steering modality strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class SteeringResult:
    features: list
    adjustments: Dict[str, float]
    metadata: Dict[str, Any]


class SteeringModality:
    """Small strategy interface for one user-facing steering modality."""

    modality_id: str = ""

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        raise NotImplementedError
