"""Initialization helpers safe to run in a subprocess."""

import json
import traceback
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from server.platform.persistence.base_models import UserStudy
from server.platform.persistence.db import resolve_database_url

from ..paths import ensure_questionnaire_cached, get_cache_path
from ..study_config import get_phase_questionnaire_filename, normalize_study_config


def long_initialization(guid):
    engine = create_engine(resolve_database_url())
    db_session = Session(engine)
    user_study = None
    try:
        user_study = db_session.query(UserStudy).filter(UserStudy.guid == guid).first()
        conf = normalize_study_config(json.loads(user_study.settings))

        Path(get_cache_path(guid)).mkdir(parents=True, exist_ok=True)
        Path(get_cache_path(guid, "sae_model")).mkdir(parents=True, exist_ok=True)
        Path(get_cache_path(guid, "embeddings")).mkdir(parents=True, exist_ok=True)
        ensure_questionnaire_cached(guid, conf.get("questionnaire_file"))
        phase_files = set()
        if conf.get("phase_questionnaire_file"):
            phase_files.add(conf["phase_questionnaire_file"])
        for idx, _model in enumerate(conf.get("models", [])):
            phase_file = get_phase_questionnaire_filename(conf, idx)
            if phase_file:
                phase_files.add(phase_file)
        for phase_file in phase_files:
            ensure_questionnaire_cached(guid, phase_file)

        print(f"Initialized SAE steering study with GUID: {guid}")
        user_study.initialized = True
        user_study.active = True
    except Exception:
        if user_study is not None:
            user_study.initialization_error = traceback.format_exc()
            print(f"Error during SAE steering initialization: {user_study.initialization_error}")
    db_session.commit()
    db_session.expunge_all()
    db_session.close()
