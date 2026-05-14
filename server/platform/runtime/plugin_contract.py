"""Public contract objects that study plugins expose to the platform kernel."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    description: str
    hidden_from_admin: bool = False


@dataclass
class StudyPluginContract:
    metadata: PluginMetadata
    blueprint: Any
    config_schema: dict[str, Any] = field(default_factory=dict)
    modalities: dict[str, Any] = field(default_factory=dict)
    results_hooks: dict[str, Any] = field(default_factory=dict)
    persistence_hooks: dict[str, Any] = field(default_factory=dict)
