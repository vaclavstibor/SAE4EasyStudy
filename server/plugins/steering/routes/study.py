"""Participation flow and phase transition endpoints."""

import datetime

from flask import abort, redirect, render_template, request, session, url_for

from server.platform.persistence.base_models import Participation, UserStudy
from server.platform.persistence.db import db
from server.platform.shared.common import (
    get_tr,
    load_user_study_config,
    load_user_study_config_by_guid,
    multi_lang,
)

from ..constants import PLUGIN_NAME
from ..paths import ensure_questionnaire_cached
from ..plugin import bp, get_lang, languages
from ..service import audit
from ..service.participation import (
    ensure_participation_for_guid,
    get_effective_models,
    sync_prolific_session_from_request,
)
from ..study_config import (
    approach_label,
    get_active_model_config,
    get_phase_questionnaire_filename,
    normalize_study_config,
)


def get_min_resolution_settings(conf):
    min_resolution_cfg = (
        conf.get("min_resolution") if isinstance(conf.get("min_resolution"), dict) else {}
    )

    def safe_int(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    width = safe_int(
        conf.get("min_resolution_width", conf.get("min_width", min_resolution_cfg.get("width"))),
        1280,
    )
    height = safe_int(
        conf.get("min_resolution_height", conf.get("min_height", min_resolution_cfg.get("height"))),
        720,
    )
    error_message = conf.get(
        "min_resolution_error",
        (
            f"This study requires at least {width}x{height} resolution. "
            "Please resize your browser window (or switch to a larger screen) "
            "before continuing."
        ),
    )
    return width, height, error_message


def _get_current_study_guid() -> str:
    guid = session.get("user_study_guid", "")
    if guid:
        return guid
    try:
        user_study = UserStudy.query.filter(UserStudy.id == session.get("user_study_id")).first()
        if user_study:
            return user_study.guid
    except Exception:
        pass
    return ""


def questionnaire_exists(conf):
    if not conf or "questionnaire_file" not in conf:
        return False
    return ensure_questionnaire_cached(_get_current_study_guid(), conf.get("questionnaire_file"))


def phase_questionnaire_exists(conf, phase_idx=None):
    phase_questionnaire_file = get_phase_questionnaire_filename(conf, phase_idx)
    if not phase_questionnaire_file:
        return False
    return ensure_questionnaire_cached(_get_current_study_guid(), phase_questionnaire_file)


def _extract_questionnaire_answers(form) -> dict:
    answers = {}
    for key, value in form.items():
        if key in {"final_questionnaire_data", "csrf_token"}:
            continue
        answers[key] = value
    return answers


def _finalize_study_completion() -> None:
    participation_id = session.get("participation_id")
    if participation_id is None:
        return
    audit.complete_study_run(participation_id)
    Participation.query.filter(
        Participation.id == participation_id,
        Participation.time_finished.is_(None),
    ).update({"time_finished": datetime.datetime.utcnow()})
    db.session.commit()


@bp.route("/join", methods=["GET"])
@multi_lang
def join():
    if "guid" not in request.args:
        abort(400, "GUID must be provided")
    guid = request.args.get("guid")
    conf = normalize_study_config(load_user_study_config_by_guid(guid))
    if conf.get("skip_participation_details", True):
        sync_prolific_session_from_request()
        ensure_participation_for_guid(guid, get_lang)
        return redirect(url_for(f"{PLUGIN_NAME}.on_joined", **request.args))
    return redirect(
        url_for(
            "utils.join",
            continuation_url=url_for(f"{PLUGIN_NAME}.on_joined"),
            **request.args,
        )
    )


@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    return redirect(url_for(f"{PLUGIN_NAME}.study_intro"))


@bp.route("/study-intro", methods=["GET"])
def study_intro():
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    tr = get_tr(languages, get_lang())

    models = get_effective_models(conf)
    comparison_mode = conf.get("comparison_mode", "side_by_side")
    num_phases = len(models) if comparison_mode == "sequential" else 1
    num_iterations = conf.get("num_iterations", 3)
    min_resolution_width, min_resolution_height, min_resolution_error = get_min_resolution_settings(
        conf
    )
    has_questionnaire = (
        bool(conf.get("questionnaire_file"))
        or bool(conf.get("phase_questionnaire_file"))
        or any(
            model.get("phase_questionnaire_file") or conf.get("phase_questionnaire_file")
            for model in models
        )
    )

    steering_label_a = ""
    steering_label_b = ""
    if num_phases > 1 and len(models) >= 2:
        mode_a = models[0].get("steering_mode", conf.get("steering_mode", "sliders"))
        mode_b = models[1].get("steering_mode", conf.get("steering_mode", "sliders"))
        steering_labels = {
            "sliders": "sliders",
            "text": "text input",
            "both": "sliders + text",
            "none": "movie selection only",
        }
        steering_label_a = steering_labels.get(mode_a, mode_a)
        steering_label_b = steering_labels.get(mode_b, mode_b)

    custom_intro_html = None
    if "text_overrides" in conf and "study_intro" in conf["text_overrides"]:
        custom_intro_html = conf["text_overrides"]["study_intro"]

    params = {
        "title": conf.get("study_title", tr("sae_steering_title")),
        "subtitle": conf.get("study_subtitle", "Interactive Recommendation Study"),
        "custom_intro_html": custom_intro_html,
        "time_estimate": conf.get("time_estimate", "10-15 minutes"),
        "start_button_text": conf.get("start_button_text", "I give my consent, let's continue"),
        "num_phases": num_phases,
        "num_iterations": num_iterations,
        "has_questionnaire": has_questionnaire,
        "min_resolution_width": min_resolution_width,
        "min_resolution_height": min_resolution_height,
        "min_resolution_error": min_resolution_error,
        "steering_label_a": steering_label_a,
        "steering_label_b": steering_label_b,
        "notes": conf.get(
            "intro_notes",
            [
                "There are no right or wrong answers (except for attention checks).",
                "Your data is anonymous and used for research purposes only.",
            ],
        ),
        "study_parts": [
            {
                "title": "Preference elicitation",
                "description": (
                    "Choose a few movies you like so the system can estimate "
                    "your starting taste profile."
                ),
            },
            {
                "title": "Implicit feedback approach",
                "description": (
                    "Refine recommendations without sliders; the system learns "
                    "from your interactions with shown movies."
                ),
            },
            {
                "title": "Explicit feedback approach",
                "description": (
                    "Refine recommendations by directly adjusting steering "
                    "sliders representing interpretable features."
                ),
            },
            {
                "title": "Questionnaires",
                "description": (
                    "When an approach or the full study ends, you will "
                    "continue to the relevant questionnaire automatically."
                ),
            },
        ],
        "continuation_url": url_for(
            "utils.preference_elicitation",
            continuation_url=url_for(f"{PLUGIN_NAME}.show_features"),
            consuming_plugin=PLUGIN_NAME,
            initial_data_url=url_for(f"{PLUGIN_NAME}.get_initial_data"),
            search_item_url=url_for(f"{PLUGIN_NAME}.item_search"),
        ),
    }
    return render_template("study_intro.html", **params)


@bp.route("/show-features", methods=["GET"])
def show_features():
    selected_movies_raw = request.args.get("selectedMovies", "")
    selected_indices = [int(movie) for movie in selected_movies_raw.split(",") if movie]

    from server.plugins.utils.data_loading import load_ml_dataset

    loader = load_ml_dataset()
    selected_movies = []
    for idx in selected_indices:
        movie_id = loader.movie_index_to_id.get(idx)
        selected_movies.append(int(movie_id) if movie_id is not None else idx)

    session["elicitation_selected_movies"] = selected_movies
    session["iteration"] = 1
    session["cumulative_adjustments"] = {}
    session["feature_adjustments"] = {}
    session["boosted_liked_ids"] = []
    session["current_phase"] = 0
    session["phase_data"] = {}
    session["seen_movies_per_phase"] = {}
    session["persistent_liked_by_phase"] = {}
    session["shown_sliders_per_phase"] = {}
    session["steered_sliders_per_phase"] = {}
    session["last_shown_movies_per_phase"] = {}
    session["iteration_preferences_approved"] = False
    session["iteration_locked_final"] = False

    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    get_effective_models(conf)

    participation_id = session.get("participation_id")
    if participation_id:
        approach_order = session.get("approach_order")
        audit.ensure_study_run(participation_id, approach_order=approach_order)
        audit.record_elicitation_completed(selected_movies, participation_id=participation_id)
        audit.ensure_approach_run(
            participation_id,
            approach_index=0,
            conf=conf,
            approach_order=approach_order,
        )
    return redirect(url_for(f"{PLUGIN_NAME}.steering"))


def _do_advance_phase(next_phase_idx):
    session["current_phase"] = next_phase_idx
    session["iteration"] = 1
    session["cumulative_adjustments"] = {}
    session["feature_adjustments"] = {}
    session["boosted_liked_ids"] = []
    session["iteration_preferences_approved"] = False
    session["iteration_locked_final"] = False
    session.pop("excluded_movies_from_text", None)
    if session.get("participation_id"):
        conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
        audit.ensure_approach_run(
            session.get("participation_id"), approach_index=next_phase_idx, conf=conf
        )
    return redirect(url_for(f"{PLUGIN_NAME}.steering"))


@bp.route("/next-phase", methods=["GET"])
def next_phase():
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    if not conf:
        return redirect(url_for(f"{PLUGIN_NAME}.finish_user_study"))

    models = get_effective_models(conf)
    current_phase = session.get("current_phase", 0)
    next_phase_idx = current_phase + 1
    current_phase_model_name = get_active_model_config(conf, current_phase).get(
        "name", approach_label(current_phase)
    )
    phase_questionnaire_title = f"Questionnaire for {current_phase_model_name}"

    participation_id = session.get("participation_id")
    if participation_id:
        approach_name = (
            models[current_phase].get("name", f"Model {current_phase}")
            if current_phase < len(models)
            else "unknown"
        )
        approach_run = audit.ensure_approach_run(
            participation_id, approach_index=current_phase, conf=conf
        )
        current_liked = session.get("persistent_liked_by_phase", {}).get(
            str(int(current_phase)), []
        )
        audit.complete_approach_run(
            current_phase,
            participation_id=participation_id,
            summary={
                "approach_name": approach_name,
                "iterations_used": session.get("iteration", 1),
                "final_liked_count": len(current_liked or []),
                "final_liked_movie_ids": current_liked or [],
                "total_slider_changes": int(approach_run.total_slider_changes or 0),
            },
        )

    if next_phase_idx >= len(models):
        phase_questionnaire_file = get_phase_questionnaire_filename(conf, current_phase)
        if phase_questionnaire_exists(conf, current_phase):
            session["pending_next_phase"] = None
            return redirect(
                url_for(
                    "utils.final_questionnaire",
                    questionnaire_file=phase_questionnaire_file,
                    continuation_url=url_for(f"{PLUGIN_NAME}._advance_phase"),
                    title_override=phase_questionnaire_title,
                    header_override=phase_questionnaire_title,
                    hint_override="",
                    finish_override="Continue to the rest of the study",
                    hide_embedded_questionnaire_heading=1,
                )
            )
        return redirect(url_for(f"{PLUGIN_NAME}.finish_user_study"))

    phase_questionnaire_file = get_phase_questionnaire_filename(conf, current_phase)
    if phase_questionnaire_exists(conf, current_phase):
        session["pending_next_phase"] = next_phase_idx
        return redirect(
            url_for(
                "utils.final_questionnaire",
                questionnaire_file=phase_questionnaire_file,
                continuation_url=url_for(f"{PLUGIN_NAME}._advance_phase"),
                title_override=phase_questionnaire_title,
                header_override=phase_questionnaire_title,
                hint_override="",
                finish_override="Continue to the rest of the study",
                hide_embedded_questionnaire_heading=1,
            )
        )

    return _do_advance_phase(next_phase_idx)


@bp.route("/_advance-phase", methods=["GET", "POST"])
def _advance_phase():
    if request.method == "POST":
        participation_id = session.get("participation_id")
        if participation_id and "final_questionnaire_data" in request.form:
            data = _extract_questionnaire_answers(request.form)
            current_phase = session.get("current_phase", 0)
            audit.record_questionnaire_response(
                "approach",
                data,
                participation_id=participation_id,
                approach_index=current_phase,
                questionnaire_file=get_phase_questionnaire_filename(
                    normalize_study_config(load_user_study_config(session.get("user_study_id"))),
                    current_phase,
                ),
            )

    next_phase_idx = session.pop("pending_next_phase", "missing")
    if next_phase_idx is None:
        return redirect(url_for(f"{PLUGIN_NAME}.finish_user_study"))
    if next_phase_idx == "missing":
        return redirect(url_for(f"{PLUGIN_NAME}.steering"))
    return _do_advance_phase(next_phase_idx)


@bp.route("/_complete-study", methods=["GET", "POST"])
def _complete_study():
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    participation_id = session.get("participation_id")
    if (
        request.method == "POST"
        and "final_questionnaire_data" in request.form
        and participation_id is not None
    ):
        audit.record_questionnaire_response(
            "final",
            _extract_questionnaire_answers(request.form),
            participation_id=participation_id,
            questionnaire_file=conf.get("questionnaire_file"),
        )
    _finalize_study_completion()
    return redirect(url_for("utils.finish"))


@bp.route("/finish-user-study")
@multi_lang
def finish_user_study():
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    has_final_q = conf and questionnaire_exists(conf)
    print(
        f"[finish_user_study] conf has questionnaire_file: "
        f"{bool(conf and 'questionnaire_file' in conf)}, "
        f"file exists: {has_final_q}, guid: {session.get('user_study_guid', '(none)')}"
    )
    if has_final_q:
        return redirect(
            url_for(
                "utils.final_questionnaire",
                questionnaire_file=conf.get("questionnaire_file"),
                continuation_url=url_for(f"{PLUGIN_NAME}._complete_study"),
                header_override="Final Comparison Questionnaire",
                hint_override="",
                hide_embedded_questionnaire_heading=1,
            )
        )
    _finalize_study_completion()
    return redirect(url_for("utils.finish"))
