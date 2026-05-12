"""Recommendation iteration orchestration for steering requests."""

from __future__ import annotations

from flask import session

from server.platform.shared.common import load_user_study_config
from server.plugins.utils.data_loading import load_ml_dataset

from ..constants import (
    DEFAULT_RERANKING_STRATEGY,
    DEFAULT_STEERING_MODE,
    DEFAULT_TOPK_SAE_MODEL_ID,
    Modalities,
    SUPPORTED_RERANKING_STRATEGIES,
    get_default_models,
)
from .participation import get_effective_models
from ..approach_state import get_approach_id_map, get_approach_movie_set, remember_shown_movies, set_approach_movie_set
from ..recommendation.service import generate_steered_recommendations, generate_steered_recommendations_for_model, unwrap_recommendation_payload
from ..recommendation.semantic_registry import expand_feature_adjustments
from . import audit
from ..study_config import get_active_model_config, get_study_dataset_variant, normalize_steering_mode, normalize_study_config
from .engine import update_elsa_seed_with_likes
from ..modalities.registry import get_modality_strategy
from ..modalities.sliders import SLIDER_AMPLIFICATION, compute_updated_sliders


def preference_confirmation_error():
    return {
        "status": "error",
        "message": "Please confirm your movie selections before continuing.",
        "recommendations": [],
        "recommendations_a": [],
        "recommendations_b": [],
    }


def _merge_adjustments(base: dict, extra: dict) -> dict:
    merged = dict(base or {})
    for key, value in (extra or {}).items():
        skey = str(key)
        merged[skey] = round(float(merged.get(skey, 0.0)) + float(value), 4)
    return {key: value for key, value in merged.items() if abs(float(value)) > 0.001}


def apply_feature_adjustment_iteration(data: dict) -> dict:
    raw_adjustments = data.get("adjustments", {})
    request_interaction_mode = data.get("interaction_mode", "cumulative")
    excluded_movies_from_text = data.get("excluded_movies", [])
    client_liked = [movie for movie in data.get("liked_movies", []) if movie is not None]
    suppressed_genres = data.get("suppressed_features", data.get("suppressed_genres", []))
    search_context = data.get("search_context", {})
    control_state = data.get("control_state", [])
    preferences_approved = bool(data.get("preferences_approved", session.get("iteration_preferences_approved", False)))

    cluster_map = session.get("cluster_map", {})
    feature_adjustments = expand_feature_adjustments(raw_adjustments=raw_adjustments, cluster_map=cluster_map)

    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    if not conf:
        conf = normalize_study_config(
            {
                "enable_comparison": True,
                "models": get_default_models(),
                "interaction_mode": request_interaction_mode,
                "num_iterations": 3,
                "num_recommendations": 20,
            }
        )
    max_iterations = conf.get("num_iterations", 3)
    enable_comparison = conf.get("enable_comparison", False)
    interaction_mode = conf.get("interaction_mode", request_interaction_mode)
    models = get_effective_models(conf)
    is_sequential_cfg = conf.get("comparison_mode", "side_by_side") == "sequential" and len(models) >= 2
    current_phase = session.get("current_phase", 0)
    if is_sequential_cfg:
        enable_comparison = False

    num_recommendations = max(1, int(conf.get("num_recommendations", 20)))
    active_model_cfg = get_active_model_config(conf)
    active_sae_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    steering_mode_for_iteration = active_model_cfg.get("steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE))
    reranking_strategy = str(conf.get("reranking_strategy") or DEFAULT_RERANKING_STRATEGY).strip().lower()
    if reranking_strategy not in SUPPORTED_RERANKING_STRATEGIES:
        reranking_strategy = DEFAULT_RERANKING_STRATEGY

    if not preferences_approved:
        return preference_confirmation_error()

    shown_map = get_approach_id_map("last_shown_movies_per_phase")
    if is_sequential_cfg:
        remember_shown_movies(current_phase, shown_map.get(str(int(current_phase)), []))
    elif enable_comparison and len(models) >= 2:
        remember_shown_movies(0, shown_map.get("0", []))
        remember_shown_movies(1, shown_map.get("1", []))
    else:
        remember_shown_movies(0, shown_map.get("0", []))

    participation_id = session.get("participation_id")
    previous_adjustments = {} if interaction_mode == "reset" else dict(session.get("cumulative_adjustments", {}))
    user_touched = set(session.get("user_touched_features", []))

    previous_adjustments_before = dict(previous_adjustments)
    for key, val in (raw_adjustments or {}).items():
        if abs(float(val)) > 0.001:
            user_touched.add(str(key))

    for key, val in feature_adjustments.items():
        skey = str(key)
        prev = float(previous_adjustments.get(skey, 0))
        raw_delta = float(val)
        new = raw_delta * SLIDER_AMPLIFICATION
        if abs(raw_delta) > 0.001:
            previous_adjustments[skey] = round(prev + new, 4)
        elif skey in previous_adjustments and abs(prev) < 0.001:
            del previous_adjustments[skey]

    session["cumulative_adjustments"] = previous_adjustments
    session["user_touched_features"] = list(user_touched)
    model_adjustments = {k: v for k, v in previous_adjustments.items() if abs(float(v)) > 0.001}
    feature_adjustments = model_adjustments
    session["feature_adjustments"] = previous_adjustments

    active_phase_for_profile = current_phase if is_sequential_cfg else 0
    current_liked_set = {int(movie) for movie in client_liked if movie is not None}
    set_approach_movie_set("persistent_liked_by_phase", active_phase_for_profile, current_liked_set)

    example_adjustments = {}
    example_metadata = {}
    if active_model_cfg.get("use_selected_movies_as_examples") and current_liked_set:
        example_result = get_modality_strategy(Modalities.EXAMPLES).apply(
            {"example_movie_ids": list(current_liked_set)},
            conf=conf,
            active_model=active_model_cfg,
        )
        example_adjustments = expand_feature_adjustments(
            raw_adjustments=example_result.adjustments,
            cluster_map=cluster_map,
        )
        example_metadata = example_result.metadata or {}

    recommendation_adjustments = _merge_adjustments(feature_adjustments, example_adjustments)

    current_liked = set(int(movie) for movie in client_liked)
    already_boosted = set(int(movie) for movie in session.get("boosted_liked_ids", []))
    new_likes = [movie for movie in current_liked if movie not in already_boosted]
    removed_likes = [movie for movie in already_boosted if movie not in current_liked]
    like_weight = float(active_model_cfg.get("selection_signal_weight", conf.get("selection_signal_weight", 0.5)))
    if new_likes or removed_likes:
        update_elsa_seed_with_likes(current_liked, active_sae_id, like_weight=like_weight, like_cap=10)
    session["boosted_liked_ids"] = list(current_liked)

    loader = load_ml_dataset(ml_variant=get_study_dataset_variant(conf))
    seed_genres = set()
    for movie_id in session.get("elicitation_selected_movies", []):
        try:
            row = loader.movies_df_indexed.loc[int(movie_id)]
            for genre in str(row.genres).split("|"):
                genre = genre.strip()
                if genre and genre != "(no genres listed)":
                    seed_genres.add(genre)
        except (KeyError, AttributeError):
            pass
    for movie_id in current_liked:
        try:
            row = loader.movies_df_indexed.loc[int(movie_id)]
            for genre in str(row.genres).split("|"):
                genre = genre.strip()
                if genre and genre != "(no genres listed)":
                    seed_genres.add(genre)
        except (KeyError, AttributeError):
            pass
    session["seed_genres"] = list(seed_genres)

    current_iteration = session.get("iteration", 1)
    if participation_id:
        active_phase = session.get("current_phase", 0)
        audit.record_feature_adjustment(
            participation_id=participation_id,
            approach_index=int(active_phase),
            raw_adjustments=raw_adjustments,
            recommendation_adjustments=recommendation_adjustments,
            previous_adjustments=previous_adjustments_before,
            resulting_adjustments=previous_adjustments,
            active_model=active_model_cfg,
            iteration=current_iteration,
            modality=normalize_steering_mode(steering_mode_for_iteration),
            search_context=search_context,
            control_state=control_state,
            liked_movies=client_liked,
            example_adjustments=example_adjustments,
            example_metadata=example_metadata,
            cluster_map=cluster_map,
        )

    selected_movies = session.get("elicitation_selected_movies", [])
    excluded_movie_ids = list(set(excluded_movies_from_text or []))
    if excluded_movie_ids:
        session["excluded_movies_from_text"] = excluded_movie_ids

    is_sequential = conf.get("comparison_mode", "side_by_side") == "sequential" and len(models) >= 2
    total_phases = len(models) if is_sequential else 1
    is_final_iteration_in_phase = (current_iteration + 1) > max_iterations
    is_final_of_study = is_final_iteration_in_phase and ((current_phase + 1) >= total_phases if is_sequential else True)

    response_data = {
        "status": "success",
        "iteration": current_iteration + 1,
        "max_iterations": max_iterations,
        "is_final_iteration": is_final_iteration_in_phase,
        "is_final_of_study": is_final_of_study,
        "is_sequential": is_sequential,
        "current_phase": current_phase,
        "total_phases": total_phases,
        "interaction_mode": interaction_mode,
        "reranking_strategy": reranking_strategy,
    }
    debug_insights = {}

    if is_sequential:
        active_model = session.get("active_model_config", models[current_phase])
        seen_in_phase = get_approach_movie_set("seen_movies_per_phase", current_phase)
        payload = generate_steered_recommendations_for_model(
            loader=loader,
            selected_movies=list(set(selected_movies + excluded_movie_ids + list(seen_in_phase))),
            feature_adjustments=recommendation_adjustments,
            model_config=active_model,
            k=num_recommendations,
            suppressed_genres=suppressed_genres,
        )
        recommendations, debug_insights = unwrap_recommendation_payload(payload)
        response_data["recommendations"] = recommendations
        if participation_id:
            audit.record_recommendations_shown(
                recommendations,
                participation_id=participation_id,
                approach_index=current_phase,
                iteration=current_iteration,
                list_id="recs-single",
                steering_mode=steering_mode_for_iteration,
                debug_payload=debug_insights,
            )
        shown_map = get_approach_id_map("last_shown_movies_per_phase")
        shown_map[str(int(current_phase))] = [int(row.get("movie_idx")) for row in recommendations if row.get("movie_idx") is not None]
        session["last_shown_movies_per_phase"] = shown_map
    elif enable_comparison and len(models) >= 2:
        payload_a = generate_steered_recommendations_for_model(
            loader=loader,
            selected_movies=list(set(selected_movies + excluded_movie_ids + list(get_approach_movie_set("seen_movies_per_phase", 0)))),
            feature_adjustments=recommendation_adjustments,
            model_config=models[0],
            k=num_recommendations,
            suppressed_genres=suppressed_genres,
        )
        payload_b = generate_steered_recommendations_for_model(
            loader=loader,
            selected_movies=list(set(selected_movies + excluded_movie_ids + list(get_approach_movie_set("seen_movies_per_phase", 1)))),
            feature_adjustments=recommendation_adjustments,
            model_config=models[1],
            k=num_recommendations,
            suppressed_genres=suppressed_genres,
        )
        recommendations_a, debug_a = unwrap_recommendation_payload(payload_a)
        recommendations_b, debug_b = unwrap_recommendation_payload(payload_b)
        debug_insights = {"model_a": debug_a, "model_b": debug_b}
        response_data["recommendations_a"] = recommendations_a
        response_data["recommendations_b"] = recommendations_b
        response_data["recommendations"] = recommendations_a
        if participation_id:
            audit.record_recommendations_shown(
                recommendations_a,
                participation_id=participation_id,
                approach_index=0,
                iteration=current_iteration,
                list_id="recs-model-a",
                steering_mode=models[0].get("steering_mode", steering_mode_for_iteration),
                debug_payload=debug_a,
            )
            audit.record_recommendations_shown(
                recommendations_b,
                participation_id=participation_id,
                approach_index=1,
                iteration=current_iteration,
                list_id="recs-model-b",
                steering_mode=models[1].get("steering_mode", steering_mode_for_iteration),
                debug_payload=debug_b,
            )
        shown_map = get_approach_id_map("last_shown_movies_per_phase")
        shown_map["0"] = [int(row.get("movie_idx")) for row in recommendations_a if row.get("movie_idx") is not None]
        shown_map["1"] = [int(row.get("movie_idx")) for row in recommendations_b if row.get("movie_idx") is not None]
        session["last_shown_movies_per_phase"] = shown_map
    else:
        seen_single = get_approach_movie_set("seen_movies_per_phase", 0)
        if models:
            recommendations_payload = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=list(set(selected_movies + excluded_movie_ids + list(seen_single))),
                feature_adjustments=recommendation_adjustments,
                model_config=models[0],
                k=num_recommendations,
                suppressed_genres=suppressed_genres,
            )
            recommendations, debug_insights = unwrap_recommendation_payload(recommendations_payload)
        else:
            recommendations = generate_steered_recommendations(
                loader=loader,
                selected_movies=list(set(selected_movies + excluded_movie_ids + list(seen_single))),
                feature_adjustments=recommendation_adjustments,
                k=num_recommendations,
            )
        response_data["recommendations"] = recommendations
        if participation_id:
            audit.record_recommendations_shown(
                recommendations,
                participation_id=participation_id,
                approach_index=0,
                iteration=current_iteration,
                list_id="recs-single",
                steering_mode=steering_mode_for_iteration,
                debug_payload=debug_insights,
            )
        shown_map = get_approach_id_map("last_shown_movies_per_phase")
        shown_map["0"] = [int(row.get("movie_idx")) for row in recommendations if row.get("movie_idx") is not None]
        session["last_shown_movies_per_phase"] = shown_map

    if steering_mode_for_iteration == Modalities.TEXT:
        response_data["updated_features"] = []
    elif steering_mode_for_iteration in (Modalities.SLIDERS, Modalities.HYBRID, Modalities.TOGGLES):
        updated_features = compute_updated_sliders(
            current_features=session.get("current_features", []),
            cumulative_adjustments={cid: 1.0 for cid in user_touched if cid.startswith("cluster_")},
            liked_movie_ids=list(current_liked_set),
            model_id=active_sae_id,
            num_sliders=conf.get("num_sliders", 16),
            phase_idx=active_phase_for_profile,
        )
        if updated_features and updated_features != session.get("current_features", []):
            session["current_features"] = updated_features
            response_data["updated_features"] = updated_features

    response_data["debug_insights"] = debug_insights or {}
    session["iteration"] = current_iteration + 1
    session["iteration_preferences_approved"] = False
    session["iteration_locked_final"] = bool(is_final_iteration_in_phase)
    return response_data

