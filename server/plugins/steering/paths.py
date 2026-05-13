"""Filesystem path helpers for the SAE plugin."""

import shutil
from pathlib import Path

from .constants import PLUGIN_NAME

PLUGIN_ROOT = Path(__file__).resolve().parents[2]


def get_cache_path(guid, name=""):
    return str(PLUGIN_ROOT / "cache" / PLUGIN_NAME / guid / name)


def get_uploads_path(name=""):
    return str(PLUGIN_ROOT / "cache" / PLUGIN_NAME / "uploads" / name)


def get_bundled_questionnaire_path(name=""):
    base_dir = Path(__file__).resolve().parents[2] / "static" / "questionnairs"
    return str(base_dir / name) if name else str(base_dir)


def ensure_questionnaire_cached(guid, filename):
    if not guid or not filename:
        return False

    destination = Path(get_cache_path(guid, filename))
    if destination.is_file():
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    for source in (
        Path(get_uploads_path(filename)),
        Path(get_bundled_questionnaire_path(filename)),
    ):
        if source.is_file():
            shutil.copy2(source, destination)
            return True
    return False
