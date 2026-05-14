"""Participation/session lifecycle helpers."""

import datetime
import json
import secrets

from flask import request, session

from server.platform.persistence.base_models import Participation, UserStudy
from server.platform.persistence.db import db

from ..study_config import normalize_study_config


def sync_prolific_session_from_request():
    if "PROLIFIC_PID" in request.args:
        session["PROLIFIC_PID"] = request.args.get("PROLIFIC_PID")
        session["PROLIFIC_STUDY_ID"] = request.args.get("STUDY_ID")
        session["PROLIFIC_SESSION_ID"] = request.args.get("SESSION_ID")
    else:
        session.pop("PROLIFIC_PID", None)
        session.pop("PROLIFIC_STUDY_ID", None)
        session.pop("PROLIFIC_SESSION_ID", None)


def persist_approach_order_on_participation(raw_order, effective_names, model_names):
    participation_id = session.get("participation_id")
    if not participation_id:
        return
    try:
        participation = Participation.query.filter(Participation.id == participation_id).first()
        if participation is None:
            return
        try:
            extra = json.loads(participation.extra_data) if participation.extra_data else {}
            if not isinstance(extra, dict):
                extra = {}
        except Exception:
            extra = {}
        extra["approach_order"] = list(raw_order)
        extra["effective_order"] = list(effective_names)
        extra["model_names"] = list(model_names)
        participation.extra_data = json.dumps(extra)
        db.session.commit()
    except Exception as exc:  # pragma: no cover
        print(f"[persist_approach_order_on_participation] Failed to persist order: {exc}")


def log_approach_order_once(raw_order, models):
    if session.get("approach_order_logged"):
        return
    participation_id = session.get("participation_id")
    if not participation_id:
        return
    if Participation.query.filter(Participation.id == participation_id).first() is None:
        print(
            f"[log_approach_order_once] participation_id={participation_id} gone; "
            "clearing stale session state."
        )
        for key in ("participation_id", "approach_order", "approach_order_logged"):
            session.pop(key, None)
        return
    model_names = [m.get("name", f"Model {i}") for i, m in enumerate(models)]
    effective_names = [models[idx].get("name", f"Model {idx}") for idx in raw_order]
    # Set the flag before record_event because record_event -> ensure_study_run ->
    # _approach_order -> get_effective_models loops back into this function.
    session["approach_order_logged"] = True
    persist_approach_order_on_participation(raw_order, effective_names, model_names)
    from .audit import record_event

    record_event(
        "approach-order-assigned",
        participation_id=participation_id,
        raw_payload={
            "approach_order": list(raw_order),
            "model_names": model_names,
            "effective_order": effective_names,
        },
        allow_no_approach=True,
    )


def _is_valid_approach_order(raw_order, count):
    return (
        isinstance(raw_order, list)
        and len(raw_order) == count
        and sorted(int(idx) for idx in raw_order) == list(range(count))
    )


def get_effective_models(conf):
    conf = normalize_study_config(conf)
    models = list(conf.get("models", []))
    if len(models) <= 1 or not conf.get("enable_comparison", False):
        return models

    count = len(models)
    if not conf.get("randomize_approach_order", True):
        raw_order = list(range(count))
        session["approach_order"] = raw_order
        log_approach_order_once(raw_order, models)
        return models

    raw_order = session.get("approach_order")
    if not _is_valid_approach_order(raw_order, count):
        raw_order = list(range(count))
        secrets.SystemRandom().shuffle(raw_order)
        session["approach_order"] = raw_order
        session.pop("approach_order_logged", None)
        print(f"[get_effective_models] Assigned per-participant approach order: {raw_order}")

    log_approach_order_once(raw_order, models)
    return [models[idx] for idx in raw_order]


def ensure_participation_for_guid(guid: str, get_lang):
    existing_id = session.get("participation_id")
    if existing_id and session.get("user_study_guid") == guid:
        if Participation.query.filter(Participation.id == existing_id).first():
            return
        print(
            f"[ensure_participation_for_guid] Stale participation_id={existing_id} "
            "in session (row missing in DB); regenerating."
        )
        for key in (
            "participation_id",
            "uuid",
            "approach_order",
            "approach_order_logged",
            "user_study_id",
            "user_study_guid",
        ):
            session.pop(key, None)

    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        raise ValueError(f"Unknown study guid: {guid}")

    if "uuid" not in session:
        session["uuid"] = secrets.token_urlsafe(16)

    extra_data = {}
    if "PROLIFIC_PID" in session:
        extra_data["PROLIFIC_PID"] = session["PROLIFIC_PID"]
        extra_data["PROLIFIC_STUDY_ID"] = session.get("PROLIFIC_STUDY_ID")
        extra_data["PROLIFIC_SESSION_ID"] = session.get("PROLIFIC_SESSION_ID")

    participant_email = session.get("PROLIFIC_PID") or ""
    participation = Participation(
        participant_email=participant_email,
        user_study_id=user_study.id,
        time_joined=datetime.datetime.utcnow(),
        time_finished=None,
        age_group=None,
        gender=None,
        education=None,
        ml_familiar=None,
        language=get_lang(),
        uuid=session["uuid"],
        extra_data=json.dumps(extra_data),
    )
    db.session.add(participation)
    db.session.commit()

    session["participation_id"] = participation.id
    session["user_study_id"] = user_study.id
    session["user_study_guid"] = guid
    session.pop("approach_order_logged", None)
