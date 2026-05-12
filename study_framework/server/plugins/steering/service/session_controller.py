"""Steering page state assembly and iteration orchestration."""

from __future__ import annotations

import traceback

from flask import session

from server.platform.shared.common import get_tr, load_user_study_config

from ..plugin import get_lang, languages
from ..constants import (
    DEFAULT_STEERING_MODE,
    DEFAULT_TOPK_SAE_MODEL_ID,
    TEXT_STEERING_MAX_QUERY_CHARS,
    get_default_models,
)
from ..recommendation.features import select_slider_features
from .participation import get_effective_models
from ..approach_state import get_approach_id_map, get_approach_movie_set, get_approach_token_set, set_approach_token_set
from ..recommendation.service import generate_steered_recommendations, generate_steered_recommendations_for_model, unwrap_recommendation_payload
from ..recommendation.semantic_registry import load_semantic_clusters
from . import audit
from ..study_config import (
    get_active_model_config,
    get_steering_guidance,
    get_steering_subtitle,
    get_study_dataset_variant,
    normalize_study_config,
)


def _default_config(interaction_mode="cumulative"):
    return normalize_study_config(
        {
            "enable_comparison": True,
            "models": get_default_models(),
            "interaction_mode": interaction_mode,
            "num_iterations": 3,
            "num_recommendations": 20,
            "steering_mode": DEFAULT_STEERING_MODE,
        }
    )


def build_steering_page_context(get_min_resolution_settings, phase_questionnaire_exists):
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id"))) or _default_config()
    min_resolution_width, min_resolution_height, min_resolution_error = get_min_resolution_settings(conf)
    tr = get_tr(languages, get_lang())
    selected_movies = session.get("elicitation_selected_movies", [])

    current_phase_tmp = session.get("current_phase", 0)
    active_model_cfg = get_active_model_config(conf, current_phase_tmp)
    active_sae_model_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)

    num_sliders = conf.get("num_sliders", 16)
    features = select_slider_features(
        selected_movies=selected_movies,
        conf=conf,
        active_model_cfg=active_model_cfg,
        num_sliders=num_sliders,
    )[:num_sliders]

    semantic_clusters = load_semantic_clusters(active_sae_model_id)
    cluster_map = semantic_clusters["cluster_map"]
    all_clusters_catalog = [
        {
            "id": cluster["cluster_id"],
            "label": cluster["label"],
            "description": cluster.get("description", ""),
            "member_ids": cluster["neuron_ids"],
            "movie_count": cluster.get("support", len(cluster["neuron_ids"])),
        }
        for cluster in semantic_clusters["clusters"]
    ]

    session["current_features"] = features
    session["cluster_map"] = cluster_map
    shown_phase = (
        current_phase_tmp
        if (conf.get("comparison_mode", "side_by_side") == "sequential" and len(get_effective_models(conf)) >= 2)
        else 0
    )
    shown_ids = get_approach_token_set("shown_sliders_per_phase", shown_phase)
    shown_ids.update({str(feature.get("id")) for feature in features if feature.get("id") is not None})
    set_approach_token_set("shown_sliders_per_phase", shown_phase, shown_ids)

    max_iterations = conf.get("num_iterations", 3)
    comparison_mode = conf.get("comparison_mode", "side_by_side")
    enable_comparison = conf.get("enable_comparison", False)
    interaction_mode = conf.get("interaction_mode", "reset")
    models = get_effective_models(conf)
    num_recommendations = max(1, int(conf.get("num_recommendations", 20)))

    is_sequential = comparison_mode == "sequential" and len(models) >= 2
    current_phase = session.get("current_phase", 0)
    total_phases = len(models) if is_sequential else 1

    active_model = active_model_cfg
    if is_sequential:
        enable_comparison = False
        active_model = models[current_phase] if current_phase < len(models) else models[0]
        session["active_model_config"] = active_model
    model_a_name = models[0].get("name", "Model A") if len(models) > 0 else "Model A"
    model_b_name = models[1].get("name", "Model B") if len(models) > 1 else "Model B"

    next_phase_name = ""
    if is_sequential and current_phase + 1 < len(models):
        next_phase_name = models[current_phase + 1].get("name", f"Model {chr(66 + current_phase)}")
    has_phase_questionnaire_for_current_phase = bool(
        is_sequential and phase_questionnaire_exists(conf, current_phase)
    )

    initial_recs_a = []
    initial_recs_b = []
    initial_recs = []
    try:
        import numpy as np
        import torch as _torch_init
        from server.plugins.utils.data_loading import load_ml_dataset
        from ..recommendation.sae_recommender import get_sae_recommender

        loader = load_ml_dataset(ml_variant=get_study_dataset_variant(conf))
        elsa_seed = None
        seed_genres = set()
        if selected_movies:
            try:
                recommender = get_sae_recommender(model_id=active_sae_model_id)
                recommender.load()
                if recommender.item_embeddings is not None and recommender.item_ids is not None:
                    id_to_idx = {int(mid): i for i, mid in enumerate(recommender.item_ids)}
                    embeddings = []
                    for mid in selected_movies:
                        idx = id_to_idx.get(int(mid))
                        if idx is not None:
                            emb = recommender.item_embeddings[idx]
                            if isinstance(emb, _torch_init.Tensor):
                                emb = emb.cpu().numpy()
                            embeddings.append(emb)
                        try:
                            row = loader.movies_df_indexed.loc[int(mid)]
                            for genre in str(row.genres).split("|"):
                                genre = genre.strip()
                                if genre and genre != "(no genres listed)":
                                    seed_genres.add(genre)
                        except (KeyError, AttributeError):
                            pass
                    if embeddings:
                        elsa_seed = np.mean(embeddings, axis=0).astype(np.float32)
            except Exception as exc:
                print(f"[steering] Could not compute ELSA seed: {exc}")
                traceback.print_exc()

        session["elsa_seed"] = elsa_seed.tolist() if elsa_seed is not None else None
        session["elsa_seed_movie_count"] = len(selected_movies) if elsa_seed is not None else 0
        session["seed_genres"] = list(seed_genres)
        session["cumulative_adjustments"] = {}
        session["feature_adjustments"] = {}

        empty_adj = {}
        if is_sequential:
            seen_for_phase = get_approach_movie_set("seen_movies_per_phase", current_phase)
            payload = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=list(set(selected_movies + list(seen_for_phase))),
                feature_adjustments=empty_adj,
                model_config=active_model,
                k=num_recommendations,
            )
            initial_recs, initial_debug = unwrap_recommendation_payload(payload)
            participation_id = session.get("participation_id")
            if participation_id:
                audit.record_recommendations_shown(
                    initial_recs,
                    participation_id=participation_id,
                    approach_index=current_phase,
                    iteration=session.get("iteration", 1),
                    list_id="recs-single",
                    steering_mode=active_model.get("steering_mode", DEFAULT_STEERING_MODE),
                    debug_payload=initial_debug,
                )
        elif enable_comparison and len(models) >= 2:
            seen_a = get_approach_movie_set("seen_movies_per_phase", 0)
            seen_b = get_approach_movie_set("seen_movies_per_phase", 1)
            payload_a = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=list(set(selected_movies + list(seen_a))),
                feature_adjustments=empty_adj,
                model_config=models[0],
                k=num_recommendations,
            )
            payload_b = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=list(set(selected_movies + list(seen_b))),
                feature_adjustments=empty_adj,
                model_config=models[1],
                k=num_recommendations,
            )
            initial_recs_a, initial_debug_a = unwrap_recommendation_payload(payload_a)
            initial_recs_b, initial_debug_b = unwrap_recommendation_payload(payload_b)
            participation_id = session.get("participation_id")
            if participation_id:
                audit.record_recommendations_shown(
                    initial_recs_a,
                    participation_id=participation_id,
                    approach_index=0,
                    iteration=session.get("iteration", 1),
                    list_id="recs-model-a",
                    steering_mode=models[0].get("steering_mode", DEFAULT_STEERING_MODE),
                    debug_payload=initial_debug_a,
                )
                audit.record_recommendations_shown(
                    initial_recs_b,
                    participation_id=participation_id,
                    approach_index=1,
                    iteration=session.get("iteration", 1),
                    list_id="recs-model-b",
                    steering_mode=models[1].get("steering_mode", DEFAULT_STEERING_MODE),
                    debug_payload=initial_debug_b,
                )
        else:
            if models:
                seen_single = get_approach_movie_set("seen_movies_per_phase", 0)
                payload = generate_steered_recommendations_for_model(
                    loader=loader,
                    selected_movies=list(set(selected_movies + list(seen_single))),
                    feature_adjustments=empty_adj,
                    model_config=models[0],
                    k=num_recommendations,
                )
                initial_recs, initial_debug = unwrap_recommendation_payload(payload)
                participation_id = session.get("participation_id")
                if participation_id:
                    audit.record_recommendations_shown(
                        initial_recs,
                        participation_id=participation_id,
                        approach_index=0,
                        iteration=session.get("iteration", 1),
                        list_id="recs-single",
                        steering_mode=models[0].get("steering_mode", DEFAULT_STEERING_MODE),
                        debug_payload=initial_debug,
                    )
            else:
                initial_recs = generate_steered_recommendations(
                    loader=loader,
                    selected_movies=selected_movies,
                    feature_adjustments=empty_adj,
                    k=num_recommendations,
                )

        shown_map = get_approach_id_map("last_shown_movies_per_phase")
        if is_sequential:
            shown_map[str(int(current_phase))] = [int(row.get("movie_idx")) for row in initial_recs if row.get("movie_idx") is not None]
        elif enable_comparison and len(models) >= 2:
            shown_map["0"] = [int(row.get("movie_idx")) for row in initial_recs_a if row.get("movie_idx") is not None]
            shown_map["1"] = [int(row.get("movie_idx")) for row in initial_recs_b if row.get("movie_idx") is not None]
        else:
            shown_map["0"] = [int(row.get("movie_idx")) for row in initial_recs if row.get("movie_idx") is not None]
        session["last_shown_movies_per_phase"] = shown_map
    except Exception as exc:
        print(f"[steering] Could not generate initial recs: {exc}")
        traceback.print_exc()

    steering_mode = (
        active_model.get("steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE))
        if is_sequential
        else active_model_cfg.get("steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE))
    )

    text_cfg = conf.get("text_steering") if isinstance(conf.get("text_steering"), dict) else {}
    last_text_steering = session.get("last_text_steering") or {}
    previous_text_query = str(last_text_steering.get("query") or "").strip()

    return {
        "title": active_model_cfg.get("name", tr("sae_steering_title")),
        "features": features,
        "iteration": session.get("iteration", 1),
        "max_iterations": max_iterations,
        "steering_mode": steering_mode,
        "enabled_modalities": active_model_cfg.get("enabled_modalities", conf.get("enabled_modalities", [])),
        "submit": tr("get_recommendations"),
        "enable_comparison": enable_comparison,
        "interaction_mode": interaction_mode,
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "initial_recs_a": initial_recs_a,
        "initial_recs_b": initial_recs_b,
        "initial_recs": initial_recs,
        "is_sequential": is_sequential,
        "current_phase": current_phase,
        "total_phases": total_phases,
        "next_phase_name": next_phase_name,
        "has_phase_questionnaire_for_current_phase": has_phase_questionnaire_for_current_phase,
        "seed_adjustments": {},
        "cluster_map": cluster_map,
        "all_clusters_catalog": all_clusters_catalog,
        "feature_selection_algorithm": active_model_cfg.get(
            "feature_selection_algorithm", conf.get("feature_selection_algorithm")
        ),
        "preferences_approved": bool(session.get("iteration_preferences_approved", False)),
        "iteration_locked_final": bool(session.get("iteration_locked_final", False)),
        "num_recommendations": num_recommendations,
        "header_subtitle": get_steering_subtitle(steering_mode),
        "header_guidance": get_steering_guidance(steering_mode),
        "min_resolution_width": min_resolution_width,
        "min_resolution_height": min_resolution_height,
        "min_resolution_error": min_resolution_error,
        "toggle_default_weight": active_model_cfg.get(
            "toggle_default_weight", conf.get("toggle_default_weight", 0.65)
        ),
        "text_steering_composition_mode": (
            active_model_cfg.get("text_composition_mode")
            or text_cfg.get("composition_mode")
            or "replace"
        ),
        "text_steering_max_chars": int(text_cfg.get("max_query_chars") or TEXT_STEERING_MAX_QUERY_CHARS),
        "previous_text_query": previous_text_query,
        "reranking_strategy": conf.get("reranking_strategy", "feature-conditioned"),
    }

