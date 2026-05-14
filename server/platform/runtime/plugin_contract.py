"""Public contract objects that study plugins expose to the platform kernel."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    author: str
    author_contact: str
    description: str


@dataclass
class StudyPluginContract:
    metadata: PluginMetadata
    blueprint: Any
    config_schema: dict[str, Any] = field(default_factory=dict)
    modalities: dict[str, Any] = field(default_factory=dict)
    results_hooks: dict[str, Any] = field(default_factory=dict)
    persistence_hooks: dict[str, Any] = field(default_factory=dict)
