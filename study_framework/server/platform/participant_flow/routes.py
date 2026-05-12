"""Shared participant-flow pages and endpoints owned by the platform."""

from __future__ import annotations

import secrets
from pathlib import Path

import flask
from flask import Blueprint, jsonify, make_response, render_template, request, url_for
from flask_login import current_user
from markupsafe import Markup
from werkzeug.utils import secure_filename

from server.platform.persistence.base_models import UserStudy
from server.platform.shared.common import (
    get_abs_project_root_path,
    get_tr,
    load_languages,
    load_user_study_config,
    load_user_study_config_by_guid,
    multi_lang,
)
from server.platform.shared.questionnaire_cache import ensure_questionnaire_cached_for_study
from server.plugins.utils.interaction_logging import study_ended
from server.plugins.utils.interaction_routes import register_interaction_routes

UTILS_BLUEPRINT_NAME = "utils"
PARTICIPANT_FLOW_ROOT = Path(__file__).resolve().parent

bp = Blueprint(
    UTILS_BLUEPRINT_NAME,
    __name__,
    url_prefix=f"/{UTILS_BLUEPRINT_NAME}",
    template_folder=str(PARTICIPANT_FLOW_ROOT / "templates"),
    static_folder=str(PARTICIPANT_FLOW_ROOT / "static"),
)

register_interaction_routes(bp)

languages = load_languages(str(PARTICIPANT_FLOW_ROOT))


def get_lang() -> str:
    default_lang = "en"
    if "lang" in flask.session and flask.session["lang"] and flask.session["lang"] in languages:
        return flask.session["lang"]
    return default_lang


def include_file(name: str) -> Markup:
    repo_root = get_abs_project_root_path().resolve()
    relative_name = str(name or "").lstrip("/")
    target_path = (repo_root / relative_name).resolve()
    if repo_root not in target_path.parents and target_path != repo_root:
        raise FileNotFoundError(f"Refusing to include file outside repository root: {name}")
    if not target_path.is_file():
        raise FileNotFoundError(f"Included file does not exist: {name}")
    return Markup(target_path.read_text(encoding="utf-8"))


def emit_assets(plugin_name: str, filename: str) -> Markup:
    asset_url = url_for(f"{plugin_name}.static", filename=filename)
    suffix = Path(filename).suffix.lower()
    if suffix == ".js":
        return Markup(f'<script src="{asset_url}"></script>')
    if suffix == ".css":
        return Markup(f'<link rel="stylesheet" href="{asset_url}">')
    return Markup(asset_url)


@bp.context_processor
def participant_flow_context():
    return {
        "plugin_name": UTILS_BLUEPRINT_NAME,
        "include_file": include_file,
    }


@bp.route("/results", methods=["GET"])
def results():
    return "Default results implementation"


def _error_params(tr, *, alert_text: str, **params):
    params["hint_lead"] = tr("error_hint_lead")
    params["header"] = tr("error_header")
    params["hint"] = tr("error_hint")
    params["alert_text"] = alert_text
    return params


def _resolve_min_resolution(config: dict, tr) -> dict:
    nested = config.get("min_resolution") if isinstance(config.get("min_resolution"), dict) else {}

    def _safe_int(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    width = _safe_int(
        config.get("min_resolution_width", config.get("min_width", nested.get("width"))),
        1280,
    )
    height = _safe_int(
        config.get("min_resolution_height", config.get("min_height", nested.get("height"))),
        720,
    )
    return {
        "min_resolution_width": width,
        "min_resolution_height": height,
        "min_resolution_error": config.get("min_resolution_error", tr("join_min_resolution_error")),
    }


@bp.route("/join", methods=["GET"])
@multi_lang
def join():
    if "continuation_url" not in request.args:
        flask.abort(400, "Continuation url must be provided")
    if "guid" not in request.args:
        flask.abort(400, "Guid must be provided")

    params = dict(request.args)
    params["email"] = current_user.email if current_user.is_authenticated else ""
    params["lang"] = get_lang()

    tr = get_tr(languages, get_lang())
    params["title"] = tr("join_title")
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["participant_details"] = tr("join_participant_details")
    params["please_enter_details"] = tr("join_please_enter_details")
    params["about_study"] = tr("join_about_study")
    params["study_details"] = tr("join_study_details")
    params["enter_email"] = tr("join_enter_email")
    params["enter_email_hint"] = tr("join_enter_email_hint")
    params["enter_gender"] = tr("join_enter_gender")
    params["enter_gender_hint"] = tr("join_enter_gender_hint")
    params["enter_age"] = tr("join_enter_age")
    params["enter_age_hint"] = tr("join_enter_age_hint")
    params["enter_education"] = tr("join_enter_education")
    params["enter_education_hint"] = tr("join_enter_education_hint")
    params["enter_ml_familiar"] = tr("join_enter_ml_familiar")
    params["enter_ml_familiar_hint"] = tr("join_enter_ml_familiar_hint")
    params["gender_male"] = tr("join_gender_male")
    params["gender_female"] = tr("join_gender_female")
    params["gender_other"] = tr("join_gender_other")
    params["education_no_formal"] = tr("join_education_no_formal")
    params["education_primary"] = tr("join_education_primary")
    params["education_high"] = tr("join_education_high")
    params["education_bachelor"] = tr("join_education_bachelor")
    params["education_master"] = tr("join_education_master")
    params["education_doctoral"] = tr("join_education_doctoral")
    params["yes"] = tr("yes")
    params["no"] = tr("no")
    params["informed_consent_header"] = tr("join_informed_consent_header")
    params["informed_consent_p1"] = tr("join_informed_consent_p1")
    params["informed_consent_p2"] = tr("join_informed_consent_p2")
    params["informed_consent_p3"] = tr("join_informed_consent_p3")
    params["informed_consent_p31"] = tr("join_informed_consent_p31")
    params["informed_consent_p32"] = tr("join_informed_consent_p32")
    params["informed_consent_p33"] = tr("join_informed_consent_p33")
    params["informed_consent_p4"] = tr("join_informed_consent_p4")
    params["informed_consent_p5"] = tr("join_informed_consent_p5")
    params["informed_consent_p6"] = tr("join_informed_consent_p6")
    params["start_user_study"] = tr("join_start_user_study")
    params["guid_not_found"] = tr("join_guid_not_found")
    params["server_error"] = tr("join_server_error")
    params["min_resolution_error"] = tr("join_min_resolution_error")
    params["english"] = tr("join_english")

    study = UserStudy.query.filter(UserStudy.guid == request.args.get("guid")).first()
    if study is None:
        return render_template(
            "error.html",
            **_error_params(tr, alert_text=tr("join_guid_not_found"), **params),
        )
    if not study.initialized:
        return render_template(
            "error.html",
            **_error_params(tr, alert_text=tr("error_not_initialized"), **params),
        )
    if not study.active:
        return render_template(
            "error.html",
            **_error_params(tr, alert_text=tr("error_not_active"), **params),
        )

    if "uuid" not in flask.session:
        flask.session["uuid"] = secrets.token_urlsafe(16)

    if "PROLIFIC_PID" in request.args:
        flask.session["PROLIFIC_PID"] = request.args.get("PROLIFIC_PID")
        flask.session["PROLIFIC_STUDY_ID"] = request.args.get("STUDY_ID")
        flask.session["PROLIFIC_SESSION_ID"] = request.args.get("SESSION_ID")
    else:
        for key in ("PROLIFIC_PID", "PROLIFIC_STUDY_ID", "PROLIFIC_SESSION_ID"):
            flask.session.pop(key, None)

    params["informed_consent_override"] = None
    params["about_override"] = None
    params["footer_override"] = None
    config = load_user_study_config_by_guid(request.args.get("guid")) or {}
    if "text_overrides" in config:
        params["informed_consent_override"] = config["text_overrides"].get("informed_consent")
        params["about_override"] = config["text_overrides"].get("about")
        params["footer_override"] = config["text_overrides"].get("footer")

    if "disable_demographics" in config:
        params["disable_demographics"] = config["disable_demographics"]

    return render_template("join.html", **params)


@bp.route("/preference-elicitation", methods=["GET", "POST"])
@multi_lang
def preference_elicitation():
    for required_param in ("continuation_url", "initial_data_url", "search_item_url"):
        if required_param not in request.args:
            flask.abort(400, f"{required_param} must be provided by the consuming plugin")

    config = load_user_study_config(flask.session["user_study_id"]) or {}
    impl = config.get("selected_preference_elicitation", "")
    flask.session["elicitation_movies"] = []

    params = {
        "impl": impl,
        "consuming_plugin": request.args.get("consuming_plugin"),
    }

    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["load_more"] = tr("elicitation_load_more")
    params["finish"] = tr("elicitation_finish")
    params["search"] = tr("elicitation_search")
    params["cancel_search"] = tr("elicitation_cancel_search")
    params["enter_name"] = tr("elicitation_enter_name")
    params["header"] = tr("elicitation_header")
    params["hint_lead"] = tr("elicitation_hint_lead")
    params["hint"] = tr("elicitation_hint")
    params["title"] = tr("elicitation_title")
    params["continuation_url"] = request.args.get("continuation_url")
    params["initial_data_url"] = request.args.get("initial_data_url")
    params["search_item_url"] = request.args.get("search_item_url")
    params["not_enough_movies_detail"] = tr("elicitation_not_enough_movies_detail")
    params["not_enough_movies_header"] = tr("elicitation_not_enough_movies_header")
    params["elicitation_hint_override"] = None
    params["footer_override"] = None
    params.update(_resolve_min_resolution(config, tr))

    if "text_overrides" in config:
        params["elicitation_hint_override"] = config["text_overrides"].get("elicitation_hint")
        params["footer_override"] = config["text_overrides"].get("footer")

    return render_template("preference_elicitation.html", **params)


def prepare_basic_statistics(n_algorithms, algorithm_names):
    res = {}
    counts = {name: 0 for name in algorithm_names}
    for variants, permutation in zip(
        flask.session["selected_variants"], flask.session["orig_permutation"]
    ):
        algo_idx_to_name = {idx: algo_name for algo_name, idx in permutation.items()}
        for variant in variants:
            counts[algo_idx_to_name[variant]] += 1

    avg_ratings = {name: 0.0 for name in algorithm_names}
    for ratings in flask.session["a_r"]:
        for algo_name, rating in ratings.items():
            avg_ratings[algo_name] += rating
    avg_ratings = {
        algo_name: round(sum_ratings / len(flask.session["a_r"]), 1)
        if len(flask.session["a_r"])
        else 0
        for algo_name, sum_ratings in avg_ratings.items()
    }

    res["n_selected"] = sum(len(items) for items in flask.session["selected_movie_indices"])
    res["n_recommended"] = int(flask.session["iteration"]) * flask.session["rec_k"] * n_algorithms
    res["n_selected_per_algorithm"] = counts
    res["n_avg_rating_per_algorithm"] = avg_ratings
    res["n_selected_elicitation"] = len(flask.session["elicitation_selected_movies"])
    res["n_total_elicitation"] = len(flask.session["elicitation_movies"])
    return res


@bp.route("/finish", methods=["GET", "POST"])
def finish():
    conf = load_user_study_config(flask.session["user_study_id"]) or {}
    study_ended(flask.session["participation_id"], iteration=flask.session["iteration"])

    params = {}
    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = tr("finish_title")
    params["header"] = tr("finish_header")
    params["hint"] = tr("finish_hint")
    params["statistics"] = tr("finish_statistics")
    params["selected_items"] = tr("finish_selected_items")
    params["out_of"] = tr("out_of")
    params["selected_per_algorithm"] = tr("finish_selected_per_algorithm")
    params["avg_rating_per_algorithm"] = tr("finish_avg_rating_per_algorithm")
    params["selected_during_elicitation"] = tr("finish_selected_during_elicitation")
    params["table_algo_name"] = tr("finish_table_algo_name")
    params["table_n_selected"] = tr("finish_table_n_selected")
    params["table_n_shown"] = tr("finish_table_n_shown")
    params["table_avg_rating"] = tr("finish_table_avg_rating")
    params["finish_user_study"] = tr("finish_finish_user_study")

    prolific_code = conf.get("prolific_code")
    if "PROLIFIC_PID" in flask.session:
        params["prolific_pid"] = flask.session["PROLIFIC_PID"]
        params["prolific_url"] = (
            f"https://app.prolific.com/submissions/complete?cc={prolific_code}"
            if prolific_code
            else None
        )
    else:
        params["prolific_pid"] = None
        params["prolific_url"] = None

    auto_redirect_text = tr("finish_auto_redirect")
    if not auto_redirect_text or auto_redirect_text == "finish_auto_redirect":
        auto_redirect_text = "Redirecting back to Prolific shortly"
    params["auto_redirect"] = auto_redirect_text
    params["finished_text_override"] = None
    params["footer_override"] = None

    if "text_overrides" in conf:
        params["finished_text_override"] = conf["text_overrides"].get("finished_text")
        params["footer_override"] = conf["text_overrides"].get("footer")

    if conf.get("show_final_statistics"):
        params["show_final_statistics"] = True
        algorithm_names = [item["displayed_name"] for item in conf["algorithm_parameters"]]
        params.update(prepare_basic_statistics(conf["n_algorithms_to_compare"], algorithm_names))
    else:
        params["show_final_statistics"] = False

    return render_template("finish.html", **params)


@bp.route("/movie-search", methods=["GET"])
def movie_search():
    attrib = request.args.get("attrib")
    pattern = request.args.get("pattern")
    if not attrib or attrib not in ["movie"]:
        return make_response("", 404)
    if not pattern:
        return make_response("", 404)

    lang = get_lang()
    tr = None if lang == "en" else get_tr(languages, lang)
    # Lazy import: keeps the platform module free of top-level plugin coupling
    # while still letting the shared elicitation flow drive the EasyStudy-native
    # dataset loader that lives in `plugins/utils`.
    from server.plugins.utils.preference_elicitation import search_for_movie

    return jsonify(search_for_movie(attrib, pattern, tr))


@bp.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files["file"]
    safe_name = secure_filename(uploaded_file.filename)
    cache_dir = (
        get_abs_project_root_path()
        / "cache"
        / (request.form.get("plugin_name") or UTILS_BLUEPRINT_NAME)
        / "uploads"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    upload_name = f"{secrets.token_urlsafe(16)}_{safe_name}"
    uploaded_file.save(cache_dir / upload_name)
    return {"upload_name": upload_name}


@bp.route("/final-questionnaire")
def final_questionnaire():
    user_study = UserStudy.query.filter(UserStudy.id == flask.session["user_study_id"]).first()
    if user_study is None:
        return "Study not found", 404

    conf = load_user_study_config(user_study.id) or {}
    params = {}
    tr = get_tr(languages, get_lang())
    params["contacts"] = tr("footer_contacts")
    params["contact"] = tr("footer_contact")
    params["t1"] = tr("footer_t1")
    params["t2"] = tr("footer_t2")
    params["title"] = request.args.get("title_override") or tr("questionnaire_title")
    params["header"] = request.args.get("header_override") or tr("questionnaire_header")
    hint_override = request.args.get("hint_override")
    params["hint"] = tr("questionnaire_hint") if hint_override is None else hint_override
    params["continuation_url"] = request.args.get("continuation_url")
    params["finish"] = request.args.get("finish_override") or tr("questionnaire_finish")
    params["hide_embedded_questionnaire_heading"] = (
        request.args.get("hide_embedded_questionnaire_heading") or ""
    ).strip().lower() in {"1", "true", "yes"}
    params.update(_resolve_min_resolution(conf, tr))

    explicit_file = request.args.get("questionnaire_file")
    questionnaire_key = request.args.get("questionnaire_key", "questionnaire_file")
    questionnaire_filename = explicit_file or conf.get(questionnaire_key)
    if not questionnaire_filename:
        tr = get_tr(languages, get_lang())
        return render_template(
            "error.html",
            **_error_params(
                tr,
                alert_text="No questionnaire file is configured for this study step.",
                footer_override=conf.get("text_overrides", {}).get("footer"),
            ),
        ), 404
    if not ensure_questionnaire_cached_for_study(user_study.guid, questionnaire_filename):
        tr = get_tr(languages, get_lang())
        return render_template(
            "error.html",
            **_error_params(
                tr,
                alert_text=(
                    "Configured questionnaire file "
                    f"'{questionnaire_filename}' could not be resolved."
                ),
                footer_override=conf.get("text_overrides", {}).get("footer"),
            ),
        ), 404
    params["questionnaire_file"] = (
        f"cache/{user_study.parent_plugin}/{user_study.guid}/{questionnaire_filename}"
    )

    params["footer_override"] = None
    if "text_overrides" in conf:
        params["footer_override"] = conf["text_overrides"].get("footer")

    return render_template("final_questionnaire.html", **params)
