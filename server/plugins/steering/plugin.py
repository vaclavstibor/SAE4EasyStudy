"""Explicit plugin contract for the SAE steering study plugin."""

import os

from flask import Blueprint, session

from server.platform.runtime import PluginMetadata, StudyPluginContract
from server.platform.shared.common import load_languages

from .constants import (
    PLUGIN_AUTHOR,
    PLUGIN_AUTHOR_CONTACT,
    PLUGIN_DESCRIPTION,
    PLUGIN_NAME,
    PLUGIN_VERSION,
)

bp = Blueprint(
    PLUGIN_NAME,
    __name__,
    url_prefix=f"/{PLUGIN_NAME}",
    template_folder="templates",
    static_folder="static",
)

languages = load_languages(os.path.dirname(__file__))


def get_lang():
    default_lang = "en"
    if "lang" in session and session["lang"] and session["lang"] in languages:
        return session["lang"]
    return default_lang


@bp.context_processor
def plugin_name():
    return {"plugin_name": PLUGIN_NAME}


from .routes import admin, api, study  # noqa: E402,F401
from .routes.results import journey, views  # noqa: E402,F401
from .routes.steering import actions, views as steering_views  # noqa: E402,F401


PLUGIN = StudyPluginContract(
    metadata=PluginMetadata(
        name=PLUGIN_NAME,
        version=PLUGIN_VERSION,
        author=PLUGIN_AUTHOR,
        author_contact=PLUGIN_AUTHOR_CONTACT,
        description=PLUGIN_DESCRIPTION,
    ),
    blueprint=bp,
    modalities={
        "sliders": "server.plugins.steering.modalities.sliders",
        "toggles": "server.plugins.steering.modalities.toggles",
        "text": "server.plugins.steering.modalities.text",
        "examples": "server.plugins.steering.modalities.examples",
    },
    persistence_hooks={
        "models_module": "server.plugins.steering.persistence.models",
        "audit_module": "server.plugins.steering.service",
    },
    results_hooks={
        "analytics": "server.plugins.steering.results",
    },
)


def get_plugin():
    return PLUGIN
