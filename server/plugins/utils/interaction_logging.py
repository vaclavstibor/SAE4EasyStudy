"""EasyStudy-native interaction logging API.

This is the canonical home for `log_interaction`, `log_message`, and
`study_ended` as defined in upstream EasyStudy. It is used by other
EasyStudy-native plugins (fastcompare, empty_template) and any future
upstream plugin that depends on the legacy `Interaction` / `Message`
log tables.

The SAE Steering plugin does not call any function in this module; it
records audit data in typed tables via
:mod:`server.plugins.steering.service.audit`.
"""

from __future__ import annotations

import datetime
import json

from server.platform.persistence.base_models import Interaction, Message, Participation
from server.platform.persistence.db import db


def _get_participation(participation_id):
    if participation_id is None:
        return None
    return Participation.query.filter(Participation.id == participation_id).first()


def _compose_payload(
    payload=None,
    *,
    event_source=None,
    event_name=None,
    event_version=None,
    event_schema=None,
    **kwargs,
):
    body = {}
    if isinstance(payload, dict):
        body.update(payload)
    body.update(kwargs)
    if event_source is not None:
        body["_event_source"] = event_source
    if event_name is not None:
        body["_event_name"] = event_name
    if event_version is not None:
        body["_event_version"] = event_version
    if event_schema is not None:
        body["_event_schema"] = event_schema
    return body


def _insert_interaction(
    participation_id,
    interaction_type,
    payload,
    *,
    timestamp=None,
    commit=True,
):
    if participation_id is None:
        return False
    participation = _get_participation(participation_id)
    if participation is None:
        print(
            f"[log_interaction] Skipping '{interaction_type}': participation "
            f"id={participation_id} no longer exists."
        )
        return False
    interaction = Interaction(
        participation=participation.id,
        interaction_type=interaction_type,
        time=timestamp or datetime.datetime.utcnow(),
        data=json.dumps(payload, ensure_ascii=False),
    )
    db.session.add(interaction)
    if commit:
        db.session.commit()
    return True


def log_structured_interaction(
    participation_id,
    interaction_type,
    payload=None,
    *,
    event_source=None,
    event_name=None,
    event_version=None,
    event_schema=None,
    timestamp=None,
    commit=True,
    **kwargs,
):
    composed_payload = _compose_payload(
        payload=payload,
        event_source=event_source,
        event_name=event_name,
        event_version=event_version,
        event_schema=event_schema,
        **kwargs,
    )
    return _insert_interaction(
        participation_id,
        interaction_type,
        composed_payload,
        timestamp=timestamp,
        commit=commit,
    )


def log_interaction(participation_id, interaction_type, **kwargs):
    return log_structured_interaction(participation_id, interaction_type, payload=kwargs)


def log_message(participation_id, **kwargs):
    message = Message(
        time=datetime.datetime.utcnow(),
        data=json.dumps(kwargs, ensure_ascii=False),
        participation=participation_id,
    )
    db.session.add(message)
    db.session.commit()


def study_ended(participation_id, **kwargs):
    participation = _get_participation(participation_id)
    if participation is None:
        print(f"[study_ended] Skipping: participation id={participation_id} no longer exists.")
        return
    if participation.time_finished:
        return

    _insert_interaction(
        participation.id,
        "study-ended",
        kwargs,
        commit=True,
    )

    Participation.query.filter(Participation.id == participation_id).update(
        {"time_finished": datetime.datetime.utcnow()}
    )
    db.session.commit()
