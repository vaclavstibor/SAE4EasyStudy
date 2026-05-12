"""HTTP routes for the FastCompare skeleton plugin."""

from flask import jsonify, redirect, render_template, request, url_for

from server.plugins.fastcompare.plugin import bp
from server.plugins.fastcompare.service import build_home_payload


@bp.get("/")
def index():
    return render_template("fastcompare/index.html", payload=build_home_payload())


@bp.get("/create")
def create():
    return render_template("fastcompare/index.html", payload=build_home_payload())


@bp.get("/join")
def join():
    return render_template("fastcompare/index.html", payload=build_home_payload())


@bp.get("/initialize")
def initialize():
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
