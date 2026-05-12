"""EasyStudy-native interaction route handlers.

These endpoints back the participant-flow JS (selected-item, deselected-item,
changed-viewport, loaded-page, on-input, on-message). They write to the
EasyStudy-native `Interaction` and `Message` tables, which are the logging
surface used by fastcompare and any future upstream plugin.

The SAE Steering plugin does not call any handler here; its analytics read
typed tables in :mod:`server.plugins.steering.persistence.models`.
"""

from __future__ import annotations

import datetime
import json

import flask
from flask import request

from server.platform.persistence.base_models import Interaction, Participation
from server.platform.persistence.db import db

from .interaction_logging import log_message

NOISY_INPUT_TYPES = {"mouse-enter", "mouse-leave"}


def strip_viewport_items(payload):
    if not isinstance(payload, dict):
        return payload
    context = payload.get("context")
    if isinstance(context, dict):
        extra = context.get("extra")
        if isinstance(extra, dict) and "items" in extra:
            items = extra.get("items")
            if isinstance(items, list):
                extra["items_count"] = len(items)
            extra.pop("items", None)
    return payload


def persist_interaction(interaction_type: str, data):
    participation_id = flask.session.get("participation_id")
    if participation_id is None:
        return "no-participation", 204

    interaction = Interaction(
        participation=participation_id,
        interaction_type=interaction_type,
        time=datetime.datetime.utcnow(),
        data=json.dumps(data, ensure_ascii=False),
    )
    db.session.add(interaction)
    db.session.commit()
    return "OK", 200


def changed_viewport():
    body, status = persist_interaction("changed-viewport", strip_viewport_items(request.get_json()))
    return body, status


def selected_item():
    body, status = persist_interaction("selected-item", request.get_json())
    return body, status


def deselected_item():
    body, status = persist_interaction("deselected-item", request.get_json())
    return body, status


def loaded_page():
    body, status = persist_interaction("loaded-page", request.get_json())
    return body, status


def on_input():
    payload = request.get_json() or {}
    input_type = payload.get("input_type") if isinstance(payload, dict) else None
    if input_type in NOISY_INPUT_TYPES:
        return "filtered", 204
    body, status = persist_interaction("on-input", payload)
    return body, status


def on_message():
    if "participation_id" in flask.session:
        participation = Participation.query.filter(
            Participation.id == flask.session["participation_id"]
        ).first().id
    else:
        participation = None
    log_message(participation, **request.get_json())
    return "OK"


def register_interaction_routes(blueprint):
    blueprint.add_url_rule("/changed-viewport", view_func=changed_viewport, methods=["POST"])
    blueprint.add_url_rule("/selected-item", view_func=selected_item, methods=["POST"])
    blueprint.add_url_rule("/deselected-item", view_func=deselected_item, methods=["POST"])
    blueprint.add_url_rule("/loaded-page", view_func=loaded_page, methods=["POST"])
    blueprint.add_url_rule("/on-input", view_func=on_input, methods=["POST"])
    blueprint.add_url_rule("/on-message", view_func=on_message, methods=["POST"])
