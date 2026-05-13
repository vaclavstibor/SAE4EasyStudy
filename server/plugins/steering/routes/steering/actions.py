"""Steering action endpoints."""

import traceback

from flask import jsonify, request, session

from server.platform.shared.common import load_user_study_config

from ...plugin import bp
from ...constants import (
    DEFAULT_STEERING_MODE,
    DEFAULT_TEXT_COMPOSITION_MODE,
    DEFAULT_TOPK_SAE_MODEL_ID,
    SUPPORTED_TEXT_COMPOSITION_MODES,
    TEXT_STEERING_MAX_QUERY_CHARS,
)
from ...recommendation.semantic_registry import load_semantic_clusters
from ...service import audit
from ...service.audit import AuditContractError
from ...study_config import get_active_model_config, get_active_sae_model_id, normalize_study_config
from ...service.iteration_controller import apply_feature_adjustment_iteration
from ...modalities.registry import get_modality_strategy


@bp.route("/adjust-features", methods=["POST"])
def adjust_features():
    try:
        data = request.get_json(force=True) or {}
        return jsonify(apply_feature_adjustment_iteration(data))
    except Exception as exc:
        print(f"Error in adjust_features: {exc}")
        traceback.print_exc()
        return jsonify(
            {
                "status": "error",
                "message": str(exc),
                "recommendations": [],
                "recommendations_a": [],
                "recommendations_b": [],
            }
        ), 200


def _compose_text_adjustments(mode: str, previous: dict, current: dict) -> dict:
    mode = (mode or DEFAULT_TEXT_COMPOSITION_MODE).strip().lower()
    if mode not in SUPPORTED_TEXT_COMPOSITION_MODES:
        mode = DEFAULT_TEXT_COMPOSITION_MODE
    if not previous or mode == "replace":
        return dict(current or {})
    if mode == "intersect":
        keys = set(previous.keys()) & set((current or {}).keys())
        return {key: float(current[key]) for key in keys}
    merged = {key: float(value) for key, value in (previous or {}).items()}
    for key, value in (current or {}).items():
        merged[key] = round(max(-0.95, min(0.95, float(merged.get(key, 0.0)) + float(value))), 2)
    return merged


@bp.route("/parse-text-steering", methods=["POST"])
def parse_text_steering():
    data = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Missing query", "features": [], "adjustments": {}}), 200
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    text_cfg = conf.get("text_steering") if isinstance(conf.get("text_steering"), dict) else {}
    max_chars = int(text_cfg.get("max_query_chars") or TEXT_STEERING_MAX_QUERY_CHARS)
    if len(query) > max_chars:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Text query too long (max {max_chars} characters).",
                    "features": [],
                    "adjustments": {},
                    "max_chars": max_chars,
                }
            ),
            400,
        )

    active_model = get_active_model_config(conf)
    active_sae_id = active_model.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    resolved = get_modality_strategy("text").apply(data, conf=conf, active_model=active_model)

    composition_mode = (
        active_model.get("text_composition_mode")
        or text_cfg.get("composition_mode")
        or DEFAULT_TEXT_COMPOSITION_MODE
    )
    composition_mode = str(composition_mode).strip().lower()
    if composition_mode not in SUPPORTED_TEXT_COMPOSITION_MODES:
        composition_mode = DEFAULT_TEXT_COMPOSITION_MODE
    previous = (session.get("last_text_steering") or {}).get("adjustments") or {}
    composed_adjustments = _compose_text_adjustments(composition_mode, previous, resolved.adjustments)

    participation_id = session.get("participation_id")
    if participation_id:
        audit.record_text_steering(
            query,
            resolved,
            participation_id=participation_id,
            approach_index=int(session.get("current_phase", 0)),
            active_model=active_model,
            iteration=session.get("iteration", 1),
            composition_mode=composition_mode,
        )
    session["last_text_steering"] = {"query": query, "adjustments": composed_adjustments}
    matched = len(resolved.adjustments or {})
    response = {
        "status": "ok",
        "query": query,
        "features": resolved.features,
        "adjustments": composed_adjustments,
        "raw_adjustments": resolved.adjustments,
        "composition_mode": composition_mode,
        "matched": matched,
    }
    if matched == 0:
        # NFR-12: ambiguous input degrades gracefully; UI shows a hint, audit row is written.
        response["status"] = "no-match"
        response["message"] = (
            "We could not match your text to any feature. Try different wording, e.g. specific genres or themes."
        )
    return jsonify(response)


@bp.route("/apply-example-steering", methods=["POST"])
def apply_example_steering():
    data = request.get_json(force=True) or {}
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    active_model = get_active_model_config(conf)
    derived = get_modality_strategy("examples").apply(data, conf=conf, active_model=active_model)
    participation_id = session.get("participation_id")
    if participation_id:
        movies = [
            {"movie_id": mid}
            for mid in derived.metadata.get("example_movie_ids", [])
        ]
        audit.record_example_steering(
            participation_id=participation_id,
            approach_index=int(session.get("current_phase", 0)),
            iteration=session.get("iteration", 1),
            active_model=active_model,
            movies=movies,
            example_strength=derived.metadata.get("example_strength"),
            example_top_k=derived.metadata.get("example_top_k"),
        )
    session["last_example_steering"] = {
        "example_movie_ids": derived.metadata.get("example_movie_ids", []),
        "adjustments": derived.adjustments,
    }
    return jsonify(
        {
            "status": "ok",
            "features": derived.features,
            "adjustments": derived.adjustments,
            "matched_movie_count": derived.metadata.get("matched_movie_count", 0),
        }
    )


@bp.route("/search-features", methods=["GET"])
def search_features():
    query = request.args.get("q", "").strip().lower()
    if len(query) < 2:
        return jsonify([])
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    active_sae_id = get_active_sae_model_id(conf)
    try:
        semantic_clusters = load_semantic_clusters(active_sae_id)
    except Exception as exc:
        print(f"[search-features] Error: {exc}")
        return jsonify([])

    current_feature_ids = {feature["id"] for feature in session.get("current_features", [])}
    results = []
    for cluster in semantic_clusters["clusters"]:
        label = cluster["label"]
        if query in label.lower():
            label_lower = label.lower()
            if label_lower == query:
                match_rank = 0
            elif label_lower.startswith(query):
                match_rank = 1
            elif f" {query}" in label_lower:
                match_rank = 2
            else:
                match_rank = 3
            results.append(
                {
                    "id": cluster["cluster_id"],
                    "label": label,
                    "description": cluster.get("description", ""),
                    "member_ids": cluster["neuron_ids"],
                    "movie_count": cluster.get("support", len(cluster["neuron_ids"])),
                    "already_shown": cluster["cluster_id"] in current_feature_ids,
                    "match_rank": match_rank,
                }
            )
    results.sort(key=lambda row: (row["already_shown"], row["match_rank"], -row["movie_count"], len(row["label"])))
    participation_id = session.get("participation_id")
    if participation_id:
        active_model = get_active_model_config(conf)
        audit.record_feature_search(
            query,
            results[:20],
            participation_id=participation_id,
            approach_index=int(session.get("current_phase", 0)),
            iteration=session.get("iteration", 1),
            active_model=active_model,
        )
    return jsonify(results[:20])


@bp.route("/approve-preferences", methods=["POST"])
def approve_preferences():
    data = request.get_json(force=True) or {}
    session["iteration_preferences_approved"] = True
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    active_model = get_active_model_config(conf)
    is_final_confirmation = bool(session.get("iteration_locked_final", False))
    participation_id = session.get("participation_id")
    if participation_id:
        audit.record_event(
            "preferences-approved",
            participation_id=participation_id,
            approach_index=int(session.get("current_phase", 0)),
            iteration=session.get("iteration", 1),
            steering_mode=active_model.get("steering_mode", DEFAULT_STEERING_MODE),
            modality="approval",
            raw_payload={
                "liked_movies": data.get("liked_movies", []),
                "is_final_confirmation": is_final_confirmation,
            },
        )
    return jsonify({"status": "ok", "approved": True})


@bp.route("/log-movie-feedback", methods=["POST"])
def log_movie_feedback():
    try:
        data = request.get_json(force=True)
        participation_id = session.get("participation_id")
        if participation_id:
            audit.record_movie_feedback(
                data,
                participation_id=participation_id,
                approach_index=int(session.get("current_phase", 0)),
                iteration=int(session.get("iteration", 1)),
            )
        return jsonify({"status": "ok"})
    except AuditContractError as exc:
        print(f"[log_movie_feedback] Contract error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        print(f"[log_movie_feedback] Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 200


@bp.route("/log-ui-event", methods=["POST"])
def log_ui_event():
    try:
        data = request.get_json(force=True) or {}
        participation_id = session.get("participation_id")
        if not participation_id:
            return jsonify({"status": "skip"}), 200
        event_type = data.pop("event_type", "ui-event")
        allowed = {"ui-event", "slider-adjusted", "slider-restored-from-history"}
        if event_type not in allowed:
            raise AuditContractError(f"Unsupported UI audit event: {event_type}")
        audit.record_event(
            event_type,
            participation_id=participation_id,
            approach_index=int(session.get("current_phase", 0)),
            iteration=session.get("iteration", 1),
            modality="ui",
            source=data.get("source"),
            search_query=data.get("found_via_query"),
            raw_payload=data,
        )
        return jsonify({"status": "ok"})
    except Exception as exc:
        print(f"[log_ui_event] Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 200


@bp.route("/reset", methods=["POST"])
def reset_steering():
    """Dedicated reset endpoint (FR-12).

    Clears the in-session adjustment state and writes one SaeResetAction row +
    one SaeSteeringEvent envelope. Returns the cleared state so the UI can mirror
    it without round-tripping through /adjust-features.
    """
    try:
        data = request.get_json(force=True) or {}
        scope = str(data.get("scope") or "all-features")
        trigger = str(data.get("trigger") or "manual-ui-reset")
        conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
        active_model = get_active_model_config(conf)
        participation_id = session.get("participation_id")
        if participation_id:
            audit.record_global_reset(
                participation_id=participation_id,
                approach_index=int(session.get("current_phase", 0)),
                iteration=int(session.get("iteration", 1)),
                trigger=trigger,
                scope=scope,
                active_model=active_model,
            )
        session["cumulative_adjustments"] = {}
        session["feature_adjustments"] = {}
        session["user_touched_features"] = []
        session["excluded_movies_from_text"] = []
        session["last_text_steering"] = {}
        return jsonify({"status": "ok", "scope": scope})
    except AuditContractError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except Exception as exc:
        print(f"[reset] Error: {exc}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@bp.route("/autosave", methods=["POST"])
def autosave():
    try:
        data = request.get_json(force=True) or {}
        participation_id = session.get("participation_id")
        if not participation_id:
            return jsonify({"status": "skip", "reason": "no participation"}), 200
        audit.record_autosave_snapshot(
            data,
            participation_id=participation_id,
            iteration=int(session.get("iteration", 1)),
        )
        return jsonify({"status": "ok"})
    except Exception as exc:
        print(f"[autosave] Error: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 200
