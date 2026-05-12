"""Minimal contract-based entrypoint for the empty template plugin."""

from flask import Blueprint, abort, redirect, render_template, request

from server.platform.runtime import PluginMetadata, StudyPluginContract

PLUGIN_NAME = "emptytemplate"

bp = Blueprint(
    PLUGIN_NAME,
    __name__,
    url_prefix=f"/{PLUGIN_NAME}",
    template_folder="templates",
)


@bp.route("/create", methods=["GET"])
def create():
    return render_template("empty_template_create.html")


@bp.route("/join", methods=["GET"])
def join():
    if "guid" not in request.args:
        abort(400, "guid must be available in arguments")
    return render_template("empty_template_join.html")


@bp.route("/initialize", methods=["GET"])
def initialize():
    return redirect(request.args.get("continuation_url", f"/{PLUGIN_NAME}/join"))


@bp.route("/dispose", methods=["DELETE"])
def dispose():
    return "OK"


PLUGIN = StudyPluginContract(
    metadata=PluginMetadata(
        name=PLUGIN_NAME,
        version="0.1.0",
        author="Study Framework",
        author_contact="noreply@example.com",
        description="Minimal template plugin for creating new study plugins.",
    ),
    blueprint=bp,
)


def get_plugin():
    return PLUGIN
