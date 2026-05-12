"""Platform-level questionnaire cache helpers.

These helpers do not depend on any plugin. Each user study's questionnaires are
cached at ``cache/<parent_plugin>/<guid>/`` and may also be sourced from
``cache/<parent_plugin>/uploads/`` (researcher upload) or the bundled
``static/questionnairs/`` directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from server.platform.persistence.base_models import UserStudy
from server.platform.shared.common import get_abs_project_root_path


def _cache_dir(parent_plugin: str, guid: str) -> Path:
    return get_abs_project_root_path() / "cache" / parent_plugin / guid


def _uploads_dir(parent_plugin: str) -> Path:
    return get_abs_project_root_path() / "cache" / parent_plugin / "uploads"


def _bundled_dir() -> Path:
    return get_abs_project_root_path() / "static" / "questionnairs"


def ensure_questionnaire_cached_for(parent_plugin: str, guid: str, filename: str) -> bool:
    """Materialize ``filename`` into the study cache, return True on success."""
    if not parent_plugin or not guid or not filename:
        return False

    destination = _cache_dir(parent_plugin, guid) / filename
    if destination.is_file():
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    for source in (
        _uploads_dir(parent_plugin) / filename,
        _bundled_dir() / filename,
    ):
        if source.is_file():
            shutil.copy2(source, destination)
            return True
    return False


def ensure_questionnaire_cached_for_study(guid: str, filename: str) -> bool:
    """Resolve the parent plugin via ``UserStudy`` and cache the questionnaire."""
    if not guid or not filename:
        return False
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if user_study is None:
        return False
    return ensure_questionnaire_cached_for(user_study.parent_plugin, guid, filename)
