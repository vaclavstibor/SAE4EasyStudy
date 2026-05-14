"""Registry for SAE steering modality strategies."""

from __future__ import annotations

from typing import Dict

from ..constants import Modalities
from .base import SteeringModality
from .examples import ExampleSteering
from .sliders import SliderSteering
from .text import TextSteering
from .toggles import ToggleSteering

_REGISTRY: Dict[str, SteeringModality] = {
    Modalities.SLIDERS: SliderSteering(),
    Modalities.TOGGLES: ToggleSteering(),
    Modalities.TEXT: TextSteering(),
    Modalities.EXAMPLES: ExampleSteering(),
}


def get_modality_strategy(modality_id: str) -> SteeringModality:
    normalized = (modality_id or "").strip().lower()
    if normalized not in _REGISTRY:
        raise KeyError(f"Unknown steering modality: {modality_id}")
    return _REGISTRY[normalized]


def registered_modalities() -> list:
    return sorted(_REGISTRY.keys())
