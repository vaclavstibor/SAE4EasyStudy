"""Steering page routes."""

from flask import redirect, render_template, url_for

from ...plugin import bp
from ...constants import PLUGIN_NAME
from ...service.session_controller import build_steering_page_context
from ..study import get_min_resolution_settings, phase_questionnaire_exists


@bp.route("/steering", methods=["GET"])
def steering():
    params = build_steering_page_context(get_min_resolution_settings, phase_questionnaire_exists)
    return render_template("steering_interface.html", **params)


@bp.route("/steering-interface", methods=["GET"], endpoint="steering_interface")
def steering_interface_legacy():
    return redirect(url_for(f"{PLUGIN_NAME}.steering"))
