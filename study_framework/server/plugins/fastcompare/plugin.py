"""Minimal plugin-first contract for the FastCompare study plugin."""

from flask import Blueprint

from server.platform.runtime import PluginMetadata, StudyPluginContract

bp = Blueprint(
    "fastcompare",
    __name__,
    url_prefix="/fastcompare",
    template_folder="templates",
)

from . import routes  # noqa: E402,F401

PLUGIN = StudyPluginContract(
    metadata=PluginMetadata(
        name="fastcompare",
        version="0.1.0",
        author="Study Framework",
        author_contact="noreply@example.com",
        description="Minimal comparison-study plugin skeleton used as a plugin-first reference.",
    ),
    blueprint=bp,
    config_schema={"comparison_modes": ["pairwise", "listwise"]},
)


def get_plugin():
    return PLUGIN
