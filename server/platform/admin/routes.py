"""Administration endpoints: study list, study creation, DB backup, and exports."""

import datetime
import json
import os
import secrets
from pathlib import Path

import flask
import sqlalchemy
from flask_login import current_user, login_required

from server.platform.persistence.base_models import Participation, UserStudy
from server.platform.persistence.db import db
from server.platform.shared.common import gen_url_prefix

main = flask.Blueprint("main", __name__)


@main.route("/administration")
def administration():
    if current_user.is_authenticated:
        return flask.render_template("administration.html", current_user=current_user.email)
    return flask.redirect(flask.url_for("auth.login"))


@main.route("/administration/db-backup", methods=["GET"])
def administration_db_backup():
    if not current_user.is_authenticated:
        return flask.redirect(flask.url_for("auth.login"))

    backup_dir = Path(os.environ.get("BACKUP_DIR", "/app/backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        [path for path in backup_dir.glob("db_*.gz") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return flask.jsonify({"error": "No DB backup available."}), 404

    latest = candidates[0]
    return flask.send_file(
        str(latest),
        as_attachment=True,
        download_name=latest.name,
        mimetype="application/gzip",
    )


@main.route("/", methods=["GET"])
def index():
    if current_user.is_authenticated:
        return flask.redirect(flask.url_for("main.administration"))
    return flask.redirect(flask.url_for("auth.login"))


@main.route("/notify")
def notify():
    if current_user.is_authenticated:
        return flask.render_template("notify.html", guid=flask.request.args.get("guid"))
    return flask.redirect(flask.url_for("auth.login"))


def get_loaded_plugins():
    endpoints = {str(rule) for rule in flask.current_app.url_map.iter_rules()}
    plugin_contracts = flask.current_app.extensions.get("study_plugin_contracts", [])
    return [
        {
            "plugin_name": plugin.metadata.name,
            "plugin_description": plugin.metadata.description,
            "plugin_version": plugin.metadata.version,
            "plugin_author": plugin.metadata.author,
            "create_url": f"/{plugin.metadata.name}/create",
        }
        for plugin in plugin_contracts
        if f"/{plugin.metadata.name}/create" in endpoints
    ]


def get_loaded_plugin_names():
    return {plugin["plugin_name"] for plugin in get_loaded_plugins()}


@main.route("/loaded-plugins")
@login_required
def loaded_plugins():
    return get_loaded_plugins()


@main.route("/existing-user-studies")
@login_required
def existing_user_studies():
    result = (
        db.session.query(UserStudy, sqlalchemy.func.count(Participation.participant_email))
        .outerjoin(Participation, UserStudy.id == Participation.user_study_id)
        .group_by(UserStudy.id)
        .all()
    )

    def filter_cond(study):
        if current_user.is_admin():
            return True
        return study.creator == current_user.get_id()

    return flask.jsonify(
        [
            {
                "id": study.id,
                "creator": study.creator,
                "guid": study.guid,
                "parent_plugin": study.parent_plugin,
                "settings": study.settings,
                "time_created": study.time_created,
                "participants": participant_count,
                "join_url": gen_user_study_invitation_url(study.parent_plugin, study.guid),
                "active": study.active,
                "initialized": study.initialized,
                "results": gen_user_study_results_url(study.parent_plugin, study.guid),
                "error": study.initialization_error,
            }
            for study, participant_count in result
            if filter_cond(study)
        ]
    )


def gen_user_study_url(guid):
    return f"/user-study/{guid}"


def gen_user_study_results_url(parent_plugin, guid):
    return f"/results/{parent_plugin}/{guid}"


def gen_user_study_invitation_url(parent_plugin, guid):
    return f"{gen_url_prefix()}/{parent_plugin}/join?guid={guid}"


def get_vars(obj):
    return {name: value for name, value in vars(obj).items() if not name.startswith("_")}


@main.route("/results/<parent_plugin>/<guid>", methods=["GET"])
@login_required
def get_results(parent_plugin, guid):
    try:
        url = flask.url_for(f"{parent_plugin}.results", guid=guid)
    except Exception:
        url = flask.url_for("utils.results", guid=guid)
    return flask.redirect(url)


@main.route("/user-study", methods=["GET"])
@login_required
def get_user_study():
    user_study_id = flask.request.args.get("user_study_id")
    if not user_study_id:
        flask.abort(400, "user_study_id is required")
    studies = UserStudy.query.filter(UserStudy.id == user_study_id).all()
    if len(studies) > 1:
        flask.abort(500, "Multiple studies share the same id")
    if studies:
        return flask.jsonify(get_vars(studies[0]))
    return "Not found", 404


@main.route("/user-study/<id>", methods=["DELETE"])
@login_required
def delete_user_study(id):
    study_query = UserStudy.query.filter(UserStudy.id == id)
    study = study_query.first()
    if study is None:
        return "Not found", 404
    guid = study.guid
    parent_plugin = study.parent_plugin
    study_query.delete()
    db.session.commit()
    return flask.redirect(flask.url_for(f"{parent_plugin}.dispose", guid=guid))


@main.route("/user-study-active", methods=["POST"])
@login_required
def set_user_study_active():
    data = flask.request.get_json()
    user_study_id = data["user_study_id"]
    new_state = bool(data["active"])
    study = UserStudy.query.filter(UserStudy.id == user_study_id).first()
    if study is None:
        return "Not found", 404
    if not study.initialized:
        return "Cannot activate study that was not initialized yet", 500
    study.active = new_state
    db.session.commit()
    return "OK"


@main.route("/user-studies", methods=["GET"])
@login_required
def get_user_studies():
    studies = UserStudy.query.all()
    return flask.jsonify([get_vars(study) for study in studies])


@main.route("/participations", methods=["GET"])
@login_required
def get_participations():
    participations = Participation.query.all()
    return flask.jsonify([get_vars(participation) for participation in participations])


@main.route("/user-study-participants", methods=["GET"])
@login_required
def get_user_study_participants():
    user_study_id = flask.request.args.get("user_study_id")
    participants = (
        Participation.query.filter(Participation.user_study_id == user_study_id)
        .with_entities(Participation.participant_email)
        .all()
    )
    return flask.jsonify([{"participant_email": row[0]} for row in participants])


@main.route("/user-participated-user-studies", methods=["GET"])
@login_required
def get_user_participated_user_studies():
    user_email = flask.request.args.get("user_email")
    studies = Participation.query.filter(
        Participation.participant_email == user_email
    ).with_entities(Participation.user_study_id)
    return flask.jsonify([{"user_study_id": row[0]} for row in studies])


@main.route("/add-participant", methods=["POST"])
def add_participant():
    json_data = flask.request.get_json()
    user_study = UserStudy.query.filter(UserStudy.guid == json_data["user_study_guid"]).first()
    if not user_study:
        return "GUID not found", 404

    extra_data = {}
    if "PROLIFIC_PID" in flask.session:
        extra_data["PROLIFIC_PID"] = flask.session["PROLIFIC_PID"]
        extra_data["PROLIFIC_STUDY_ID"] = flask.session["PROLIFIC_STUDY_ID"]
        extra_data["PROLIFIC_SESSION_ID"] = flask.session["PROLIFIC_SESSION_ID"]

    participation = Participation(
        participant_email=json_data["user_email"],
        user_study_id=user_study.id,
        time_joined=datetime.datetime.utcnow(),
        time_finished=None,
        age_group=json_data["age_group"],
        gender=json_data["gender"],
        education=json_data["education"],
        ml_familiar=json_data["ml_familiar"],
        language=json_data["lang"],
        uuid=flask.session["uuid"],
        extra_data=json.dumps(extra_data),
    )
    db.session.add(participation)
    db.session.commit()

    flask.session["participation_id"] = participation.id
    flask.session["user_study_id"] = user_study.id
    flask.session["user_study_guid"] = json_data["user_study_guid"]
    return "OK"


@main.route("/create-user-study", methods=["POST"])
@login_required
def create_user_study():
    guid = secrets.token_urlsafe(24)
    json_data = flask.request.get_json()

    if "parent_plugin" not in json_data:
        return "Bad Request - parent plugin was not specified", 400
    if json_data["parent_plugin"] not in get_loaded_plugin_names():
        return "Bad Request - invalid parent plugin", 400
    if "config" not in json_data:
        json_data["config"] = {}

    study = UserStudy(
        creator=current_user.email,
        guid=guid,
        parent_plugin=json_data["parent_plugin"],
        settings=json.dumps(json_data["config"]),
        time_created=datetime.datetime.utcnow(),
        active=False,
        initialized=False,
        initialization_error=None,
    )
    db.session.add(study)
    db.session.commit()

    return flask.redirect(
        flask.url_for(
            f"{json_data['parent_plugin']}.initialize",
            continuation_url=flask.url_for("main.administration"),
            guid=guid,
        ),
        Response={"status": "success", "url": gen_user_study_url(guid)},
    )
