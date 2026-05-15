"""HTTP routes for the FastCompare skeleton plugin."""

from flask import jsonify, redirect, render_template, request, url_for

from server.platform.persistence.base_models import UserStudy
from server.platform.persistence.db import db
from server.plugins.fastcompare.plugin import bp
from server.plugins.fastcompare.service import build_home_payload


@bp.get("/")
def index():
    return render_template("fastcompare/index.html", payload=build_home_payload())


@bp.get("/create")
def create():
    payload = build_home_payload()
    payload["allow_create"] = True
    return render_template("fastcompare/index.html", payload=payload)


@bp.get("/join")
def join():
    payload = build_home_payload()
    payload["guid"] = request.args.get("guid", "")
    payload["joined"] = True
    return render_template("fastcompare/index.html", payload=payload)


@bp.get("/initialize")
def initialize():
    # Synchronous activation so the admin "Initialize" button immediately
    # flips the study to initialized + active. Heavy training that the
    # upstream version used to do has no fixture to consume here.
    guid = request.args.get("guid")
    if guid:
        study = UserStudy.query.filter_by(guid=guid).first()
        if study is not None:
            study.initialized = True
            study.active = True
            db.session.commit()
    continuation_url = request.args.get("continuation_url")
    if continuation_url:
        return redirect(continuation_url)
    return redirect(url_for("fastcompare.join", **request.args))


@bp.delete("/dispose")
def dispose():
    return "OK"


@bp.get("/results")
def results():
    return render_template("fastcompare/index.html", payload=build_home_payload())


@bp.get("/health")
def health():
    return jsonify(build_home_payload())
