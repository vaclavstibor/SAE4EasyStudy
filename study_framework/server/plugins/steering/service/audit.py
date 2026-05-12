"""Strict typed audit persistence for SAE steering studies.

Every user action lands in one or more typed domain tables and one minimal
``SaeSteeringEvent`` envelope row. The typed rows carry the data; the event
envelope carries only ids + timestamps for timeline ordering.
"""

from __future__ import annotations

import datetime
from typing import Any, Iterable, List, Optional

from server.platform.persistence.base_models import Participation, UserStudy
from server.platform.persistence.db import db
from server.platform.shared.common import load_user_study_config
from server.plugins.steering.persistence.models import (
    SaeApproachRun,
    SaeElicitationPick,
    SaeExampleSteering,
    SaeExampleSteeringMovie,
    SaeFeatureAdjustment,
    SaeFeatureSearch,
    SaeFeatureSearchHit,
    SaeMovieFeedback,
    SaeQuestionnaireResponse,
    SaeRecommendationItem,
    SaeRecommendationSet,
    SaeResetAction,
    SaeSteeringEvent,
    SaeStudyRun,
    SaeTextSteeringMatch,
    SaeTextSteeringQuery,
)
from server.plugins.utils.data_loading import load_ml_dataset

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID
from ..recommendation.semantic_registry import (
    expand_feature_adjustments,
    load_semantic_clusters,
)
from ..study_config import (
    get_active_model_config,
    get_study_dataset_variant,
    normalize_study_config,
)


SCHEMA_VERSION = 1


class AuditContractError(RuntimeError):
    """Raised when a committee-critical audit row would be incomplete."""


def utcnow() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _json_dict(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
    except Exception as exc:
        raise AuditContractError("Expected JSON object") from exc
    if not isinstance(loaded, dict):
        raise AuditContractError("Expected JSON object")
    return loaded


def _participation(participation_id: Optional[int] = None) -> Participation:
    if not participation_id:
        raise AuditContractError("Missing participation_id")
    participation = Participation.query.filter(Participation.id == int(participation_id)).first()
    if participation is None:
        raise AuditContractError(f"Participation {participation_id} does not exist")
    return participation


def _study(participation: Participation) -> UserStudy:
    study = UserStudy.query.filter(UserStudy.id == participation.user_study_id).first()
    if study is None:
        raise AuditContractError(f"UserStudy {participation.user_study_id} does not exist")
    return study


def _conf(participation: Participation) -> dict:
    return normalize_study_config(load_user_study_config(participation.user_study_id))


def _movie_snapshot(loader, movie_id: Any) -> dict:
    mid = int(movie_id)
    if mid not in loader.movies_df_indexed.index:
        raise AuditContractError(f"Movie {mid} missing from dataset")
    row = loader.movies_df_indexed.loc[mid]
    return {
        "movie_id": mid,
        "title": str(row.title),
        "genres": str(row.genres),
    }


def _approach_order(conf: dict, raw_order: Optional[list] = None) -> tuple[list, list]:
    canonical_models = list(normalize_study_config(conf).get("models", []))
    if raw_order is None:
        raw_order = list(range(len(canonical_models)))
    if len(raw_order) != len(canonical_models):
        raise AuditContractError("Approach order length does not match effective models")
    effective_names = [canonical_models[int(idx)].get("name") for idx in raw_order]
    if any(not name for name in effective_names):
        raise AuditContractError("Approach order contains unnamed approach")
    return list(raw_order), effective_names


def ensure_study_run(
    participation_id: int,
    *,
    approach_order: Optional[list] = None,
) -> SaeStudyRun:
    participation = _participation(participation_id)
    existing = SaeStudyRun.query.filter(SaeStudyRun.participation_id == participation.id).first()
    if existing:
        return existing
    study = _study(participation)
    conf = _conf(participation)
    raw_order, effective_names = _approach_order(conf, raw_order=approach_order)
    run = SaeStudyRun(
        participation_id=participation.id,
        user_study_id=study.id,
        schema_version=SCHEMA_VERSION,
        study_guid=study.guid,
        config_snapshot=conf,
        approach_order=raw_order,
        effective_order=effective_names,
        started_at=utcnow(),
        status="active",
    )
    db.session.add(run)
    db.session.commit()
    return run


def ensure_approach_run(
    participation_id: int,
    *,
    approach_index: int,
    conf: Optional[dict] = None,
    approach_order: Optional[list] = None,
) -> SaeApproachRun:
    participation = _participation(participation_id)
    conf = normalize_study_config(conf or load_user_study_config(participation.user_study_id))
    idx = int(approach_index)
    study_run = ensure_study_run(participation.id, approach_order=approach_order)
    existing = SaeApproachRun.query.filter(
        SaeApproachRun.study_run_id == study_run.id,
        SaeApproachRun.approach_index == idx,
    ).first()
    if existing:
        return existing
    model = get_active_model_config(conf, idx)
    for key in ("id", "name", "steering_mode", "enabled_modalities", "sae", "base"):
        if model.get(key) in (None, ""):
            raise AuditContractError(f"Approach {idx} missing {key}")
    text_cfg = (conf.get("text_steering") or {}) if isinstance(conf.get("text_steering"), dict) else {}
    composition_mode = str(
        model.get("text_composition_mode")
        or text_cfg.get("composition_mode")
        or "replace"
    )
    run = SaeApproachRun(
        study_run_id=study_run.id,
        participation_id=participation.id,
        approach_index=idx,
        approach_id=str(model["id"]),
        approach_name=str(model["name"]),
        steering_mode=str(model["steering_mode"]),
        enabled_modalities=list(model["enabled_modalities"]),
        sae_model_id=str(model["sae"]),
        base_model_id=str(model["base"]),
        composition_mode=composition_mode,
        reranking_strategy=str(conf.get("reranking_strategy", "feature-conditioned")),
        started_at=utcnow(),
        status="active",
        summary={},
        total_slider_changes=0,
    )
    db.session.add(run)
    db.session.commit()
    return run


def record_event(
    event_type: str,
    *,
    participation_id: int,
    approach_index: Optional[int] = None,
    iteration: Optional[int] = None,
    modality: Optional[str] = None,
    steering_mode: Optional[str] = None,
    source: Optional[str] = None,
    search_query: Optional[str] = None,
    raw_payload: Optional[dict] = None,
    allow_no_approach: bool = False,
    approach_order: Optional[list] = None,
) -> SaeSteeringEvent:
    participation = _participation(participation_id)
    conf = _conf(participation)
    study_run = ensure_study_run(participation.id, approach_order=approach_order)
    approach_run = None
    if not allow_no_approach:
        if approach_index is None:
            raise AuditContractError(
                "approach_index is required unless allow_no_approach=True"
            )
        approach_run = ensure_approach_run(
            participation.id,
            approach_index=approach_index,
            conf=conf,
            approach_order=approach_order,
        )
        approach_index = approach_run.approach_index
        approach_name = approach_run.approach_name
        steering_mode = steering_mode or approach_run.steering_mode
    else:
        approach_name = None
    event = SaeSteeringEvent(
        study_run_id=study_run.id,
        approach_run_id=approach_run.id if approach_run else None,
        participation_id=participation.id,
        event_type=event_type,
        approach_index=approach_index,
        approach_name=approach_name,
        iteration=iteration,
        modality=modality,
        steering_mode=steering_mode,
        source=source,
        search_query=search_query,
        raw_payload=raw_payload or {},
        created_at=utcnow(),
    )
    db.session.add(event)
    db.session.commit()
    return event


def cluster_details(
    model_id: str, raw_adjustments: dict, *, cluster_map: Optional[dict] = None
) -> List[dict]:
    if not raw_adjustments:
        return []
    registry = load_semantic_clusters(model_id)
    clusters_by_id = {row["cluster_id"]: row for row in registry.get("clusters", [])}
    cluster_map = cluster_map or registry.get("cluster_map", {})
    expanded = expand_feature_adjustments(raw_adjustments, cluster_map)
    details = []
    for cluster_id, raw_value in raw_adjustments.items():
        cid = str(cluster_id)
        cluster = clusters_by_id.get(cid)
        if cluster is None:
            raise AuditContractError(f"Cluster {cid} missing from semantic registry")
        neuron_ids = [int(nid) for nid in cluster.get("neuron_ids", [])]
        details.append(
            {
                "cluster_id": cid,
                "label": cluster.get("label"),
                "description": cluster.get("description", ""),
                "raw_delta": float(raw_value),
                "neuron_ids": neuron_ids,
                "expanded_neuron_adjustments": {
                    str(nid): float(expanded[str(nid)])
                    for nid in neuron_ids
                    if str(nid) in expanded
                },
            }
        )
    return details


def record_elicitation_completed(
    selected_movies: Iterable[int],
    *,
    participation_id: int,
) -> SaeSteeringEvent:
    participation = _participation(participation_id)
    conf = _conf(participation)
    loader = load_ml_dataset(ml_variant=get_study_dataset_variant(conf))
    movies = [_movie_snapshot(loader, mid) for mid in selected_movies]
    return record_event(
        "elicitation-completed",
        participation_id=participation.id,
        raw_payload={"selected_movies": movies},
        allow_no_approach=True,
    )


def record_elicitation_pick(
    movie_id: Any, action: str, *, participation_id: int
) -> SaeElicitationPick:
    participation = _participation(participation_id)
    pick = SaeElicitationPick(
        study_run_id=None,
        participation_id=participation.id,
        user_study_id=participation.user_study_id,
        movie_id=int(movie_id),
        action=str(action),
        created_at=utcnow(),
    )
    db.session.add(pick)
    db.session.commit()
    return pick


def _resolve_text_match_direction(weight: Any) -> Optional[str]:
    if weight is None:
        return None
    try:
        w = float(weight)
    except (TypeError, ValueError):
        return None
    if w > 0:
        return "boost"
    if w < 0:
        return "suppress"
    return "neutral"


def record_text_steering(
    query: str,
    resolved,
    *,
    participation_id: int,
    approach_index: int,
    active_model: dict,
    iteration: int,
    composition_mode: str = "replace",
) -> SaeTextSteeringQuery:
    if not query:
        raise AuditContractError("Text steering query is required")
    features = resolved.features or []
    event = record_event(
        "text-steering-parsed",
        participation_id=participation_id,
        approach_index=approach_index,
        iteration=iteration,
        modality="text",
        steering_mode=active_model.get("steering_mode"),
        raw_payload={
            "query": query,
            "segments": resolved.metadata.get("segments", []),
            "cluster_adjustments": resolved.adjustments,
            "composition_mode": composition_mode,
        },
    )
    query_row = SaeTextSteeringQuery(
        study_run_id=event.study_run_id,
        approach_run_id=event.approach_run_id,
        participation_id=event.participation_id,
        event_id=event.id,
        iteration=int(iteration),
        query_text=query,
        length_chars=len(query),
        composition_mode=composition_mode,
        created_at=utcnow(),
    )
    db.session.add(query_row)
    db.session.flush()
    for row in features:
        cluster_id = row.get("id")
        if not cluster_id:
            raise AuditContractError("Resolved text cluster missing id")
        weight = row.get("weight")
        db.session.add(
            SaeTextSteeringMatch(
                query_id=query_row.id,
                cluster_id=str(cluster_id),
                label=row.get("label"),
                weight=float(weight) if weight is not None else 0.0,
                match_score=row.get("match_score"),
                direction=_resolve_text_match_direction(weight),
            )
        )
    db.session.commit()
    return query_row


def record_feature_search(
    query: str,
    hits: Iterable[dict],
    *,
    participation_id: int,
    approach_index: int,
    iteration: int,
    active_model: dict,
) -> SaeFeatureSearch:
    if not query:
        raise AuditContractError("Feature search query is required")
    hit_list = list(hits or [])
    event = record_event(
        "feature-search",
        participation_id=participation_id,
        approach_index=approach_index,
        iteration=iteration,
        modality="feature-search",
        steering_mode=active_model.get("steering_mode"),
        source="search",
        search_query=query,
        raw_payload={"query": query, "result_count": len(hit_list)},
    )
    search_row = SaeFeatureSearch(
        study_run_id=event.study_run_id,
        approach_run_id=event.approach_run_id,
        participation_id=event.participation_id,
        event_id=event.id,
        iteration=int(iteration),
        query_text=query,
        result_count=len(hit_list),
        created_at=utcnow(),
    )
    db.session.add(search_row)
    db.session.flush()
    for rank, hit in enumerate(hit_list, start=1):
        feature_id = hit.get("id") or hit.get("feature_id")
        if feature_id is None:
            raise AuditContractError("Feature search hit missing id")
        db.session.add(
            SaeFeatureSearchHit(
                search_id=search_row.id,
                feature_id=str(feature_id),
                label=hit.get("label"),
                match_score=hit.get("match_score") or hit.get("score"),
                rank=rank,
            )
        )
    db.session.commit()
    return search_row


def record_example_steering(
    *,
    participation_id: int,
    approach_index: int,
    iteration: int,
    active_model: dict,
    movies: Iterable[dict],
    example_strength: Optional[float] = None,
    example_top_k: Optional[int] = None,
) -> SaeExampleSteering:
    movie_list = list(movies or [])
    event = record_event(
        "example-steering-applied",
        participation_id=participation_id,
        approach_index=approach_index,
        iteration=iteration,
        modality="examples",
        steering_mode=active_model.get("steering_mode"),
        source="example",
        raw_payload={
            "movie_ids": [int(m.get("movie_id", m.get("id"))) for m in movie_list if m.get("movie_id") or m.get("id")],
            "example_strength": example_strength,
            "example_top_k": example_top_k,
        },
    )
    example_row = SaeExampleSteering(
        study_run_id=event.study_run_id,
        approach_run_id=event.approach_run_id,
        participation_id=event.participation_id,
        event_id=event.id,
        iteration=int(iteration),
        example_strength=example_strength,
        example_top_k=example_top_k,
        created_at=utcnow(),
    )
    db.session.add(example_row)
    db.session.flush()
    for rank, movie in enumerate(movie_list, start=1):
        movie_id = movie.get("movie_id") or movie.get("id")
        if movie_id is None:
            raise AuditContractError("Example steering movie missing id")
        db.session.add(
            SaeExampleSteeringMovie(
                example_id=example_row.id,
                movie_id=int(movie_id),
                title=movie.get("title"),
                rank=rank,
            )
        )
    db.session.commit()
    return example_row


def record_global_reset(
    *,
    participation_id: int,
    approach_index: int,
    iteration: int,
    trigger: Optional[str] = None,
    scope: str = "all-features",
    active_model: Optional[dict] = None,
) -> SaeResetAction:
    event = record_event(
        "global-reset",
        participation_id=participation_id,
        approach_index=approach_index,
        iteration=iteration,
        modality="reset",
        steering_mode=(active_model or {}).get("steering_mode"),
        source="reset",
        raw_payload={"trigger": trigger, "scope": scope},
    )
    reset_row = SaeResetAction(
        study_run_id=event.study_run_id,
        approach_run_id=event.approach_run_id,
        participation_id=event.participation_id,
        event_id=event.id,
        iteration=int(iteration),
        trigger=trigger,
        scope=scope,
        created_at=utcnow(),
    )
    db.session.add(reset_row)
    db.session.commit()
    return reset_row


def record_feature_adjustment(
    *,
    participation_id: int,
    approach_index: int,
    raw_adjustments: dict,
    recommendation_adjustments: dict,
    previous_adjustments: dict,
    resulting_adjustments: dict,
    active_model: dict,
    iteration: int,
    modality: str,
    search_context: dict,
    control_state: list,
    liked_movies: list,
    example_adjustments: dict,
    example_metadata: dict,
    cluster_map: Optional[dict] = None,
) -> SaeSteeringEvent:
    model_id = active_model.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    searched_feature_ids = {
        str(item.get("id"))
        for item in (search_context or {}).get("adjusted_features", [])
        if item.get("id") is not None
    }
    normalized_control_state: list[dict] = []
    search_queries: list[str] = []
    for item in control_state or []:
        raw_feature_id = item.get("id")
        if raw_feature_id is None:
            continue
        feature_id = str(raw_feature_id)
        item_search_query = item.get("search_query")
        if item_search_query and item_search_query not in search_queries:
            search_queries.append(str(item_search_query))
        normalized_control_state.append(
            {
                "id": feature_id,
                "label": item.get("label"),
                "description": item.get("description"),
                "before": float(item.get("before", 0) or 0),
                "after": float(item.get("after", 0) or 0),
                "delta": float(item.get("delta", 0) or 0),
                "source": item.get("source")
                or ("search" if feature_id in searched_feature_ids else "displayed"),
                "search_query": item.get("search_query"),
            }
        )
    for item in (search_context or {}).get("adjusted_features", []):
        query = item.get("query")
        if query and query not in search_queries:
            search_queries.append(str(query))
    if search_queries:
        event_source = "search"
    elif example_adjustments:
        event_source = "selected-movie-examples"
    else:
        event_source = "displayed-controls"
    event = record_event(
        "feature-adjustment",
        participation_id=participation_id,
        approach_index=approach_index,
        iteration=iteration,
        modality=modality,
        steering_mode=active_model.get("steering_mode"),
        source=event_source,
        search_query="; ".join(search_queries) if search_queries else None,
        raw_payload={
            "raw_cluster_adjustments": raw_adjustments,
            "previous_neuron_adjustments": previous_adjustments,
            "resulting_neuron_adjustments": resulting_adjustments,
            "recommendation_neuron_adjustments": recommendation_adjustments,
            "selected_movie_example_neuron_adjustments": example_adjustments,
            "selected_movie_example_metadata": example_metadata,
            "cluster_details": cluster_details(
                model_id, raw_adjustments, cluster_map=cluster_map or {}
            ),
            "liked_movie_ids": liked_movies,
            "search_context": search_context,
        },
    )
    now = utcnow()
    nonzero_rows = 0
    for entry in normalized_control_state:
        if abs(entry["delta"]) < 1e-9:
            continue
        nonzero_rows += 1
        db.session.add(
            SaeFeatureAdjustment(
                study_run_id=event.study_run_id,
                approach_run_id=event.approach_run_id,
                participation_id=event.participation_id,
                event_id=event.id,
                iteration=int(iteration),
                feature_id=entry["id"],
                cluster_label=entry["label"],
                before_value=entry["before"],
                after_value=entry["after"],
                delta=entry["delta"],
                applied_via=entry["source"],
                search_query=entry.get("search_query"),
                created_at=now,
            )
        )
    if nonzero_rows:
        if event.approach_run_id is not None:
            approach_run = db.session.get(SaeApproachRun, event.approach_run_id)
            if approach_run is not None:
                approach_run.total_slider_changes = (
                    int(approach_run.total_slider_changes or 0) + nonzero_rows
                )
    db.session.commit()
    return event


def record_recommendations_shown(
    recommendations: list,
    *,
    participation_id: int,
    approach_index: int,
    iteration: int,
    list_id: str,
    steering_mode: str,
    debug_payload: dict,
) -> SaeRecommendationSet:
    participation = _participation(participation_id)
    conf = _conf(participation)
    approach_run = ensure_approach_run(
        participation.id, approach_index=approach_index, conf=conf
    )
    study_run = ensure_study_run(participation.id)
    rec_set = SaeRecommendationSet(
        study_run_id=study_run.id,
        approach_run_id=approach_run.id,
        participation_id=participation.id,
        approach_index=approach_run.approach_index,
        approach_name=approach_run.approach_name,
        iteration=int(iteration),
        list_id=list_id,
        steering_mode=steering_mode,
        generated_at=utcnow(),
        debug_payload=debug_payload or {},
    )
    db.session.add(rec_set)
    db.session.flush()
    for rank, row in enumerate(recommendations or [], start=1):
        movie_id = row.get("movie_idx", row.get("movie_id"))
        title = row.get("title")
        genres = row.get("metadata", row.get("genres"))
        if movie_id is None or not title or genres is None:
            raise AuditContractError("Recommendation item missing movie_id/title/genres")
        db.session.add(
            SaeRecommendationItem(
                recommendation_set_id=rec_set.id,
                movie_id=int(movie_id),
                title=str(title),
                genres=str(genres),
                rank=rank,
                list_id=list_id,
                score=row.get("score"),
                cf_score=row.get("cf_score"),
                genre_score=row.get("genre_score"),
                steering_score=row.get("steering_score"),
                raw_payload=dict(row),
            )
        )
    db.session.commit()
    record_event(
        "recommendations-shown",
        participation_id=participation_id,
        approach_index=approach_run.approach_index,
        iteration=iteration,
        modality="recommendations",
        steering_mode=steering_mode,
        raw_payload={
            "recommendation_set_id": rec_set.id,
            "movie_count": len(recommendations or []),
            "list_id": list_id,
        },
    )
    return rec_set


def lookup_recommendation_set_id(
    *,
    participation_id: int,
    approach_index: int,
    list_id: str,
) -> Optional[int]:
    """Return the most recent recommendation set id for an approach+list_id pair."""
    rec_set = (
        SaeRecommendationSet.query.filter(
            SaeRecommendationSet.participation_id == participation_id,
            SaeRecommendationSet.approach_index == int(approach_index),
            SaeRecommendationSet.list_id == list_id,
        )
        .order_by(SaeRecommendationSet.generated_at.desc(), SaeRecommendationSet.id.desc())
        .first()
    )
    return rec_set.id if rec_set else None


def record_movie_feedback(
    data: dict,
    *,
    participation_id: int,
    approach_index: int,
    iteration: int,
) -> SaeMovieFeedback:
    participation = _participation(participation_id)
    conf = _conf(participation)
    approach_run = ensure_approach_run(
        participation.id, approach_index=approach_index, conf=conf
    )
    movie_id = data.get("movie_id")
    if movie_id is None:
        raise AuditContractError("Movie feedback missing movie_id")
    loader = load_ml_dataset(ml_variant=get_study_dataset_variant(conf))
    movie = _movie_snapshot(loader, movie_id)
    list_id = data.get("list_id")
    if not list_id:
        raise AuditContractError("Movie feedback missing list_id")
    rec_set_id = lookup_recommendation_set_id(
        participation_id=participation.id,
        approach_index=approach_index,
        list_id=list_id,
    )
    if rec_set_id is None:
        raise AuditContractError(
            f"Movie feedback cannot be linked to a recommendation set for "
            f"approach={approach_index}, list_id={list_id}"
        )
    final_iteration = int(data.get("iteration", iteration))
    action = data.get("action", "neutral")
    event = record_event(
        "movie-feedback",
        participation_id=participation_id,
        approach_index=approach_index,
        iteration=final_iteration,
        modality="movie-feedback",
        steering_mode=approach_run.steering_mode,
        raw_payload={
            "movie_id": int(movie["movie_id"]),
            "action": action,
            "rank": data.get("rank"),
            "list_id": list_id,
            "recommendation_set_id": rec_set_id,
        },
    )
    feedback = SaeMovieFeedback(
        study_run_id=event.study_run_id,
        approach_run_id=approach_run.id,
        recommendation_set_id=rec_set_id,
        participation_id=participation.id,
        event_id=event.id,
        approach_index=approach_index,
        approach_name=approach_run.approach_name,
        iteration=final_iteration,
        movie_id=movie["movie_id"],
        title=movie["title"],
        genres=movie["genres"],
        rank=data.get("rank"),
        list_id=list_id,
        action=action,
        created_at=utcnow(),
    )
    db.session.add(feedback)
    db.session.commit()
    return feedback


def complete_approach_run(
    approach_index: int, *, participation_id: int, summary: dict
) -> SaeApproachRun:
    participation = _participation(participation_id)
    approach_run = ensure_approach_run(
        participation.id, approach_index=approach_index, conf=_conf(participation)
    )
    approach_run.completed_at = utcnow()
    approach_run.status = "completed"
    approach_run.final_liked_count = summary.get("final_liked_count")
    approach_run.iterations_used = summary.get("iterations_used")
    if summary.get("total_slider_changes") is not None:
        approach_run.total_slider_changes = int(summary["total_slider_changes"])
    approach_run.summary = summary
    db.session.commit()
    record_event(
        "approach-complete",
        participation_id=participation_id,
        approach_index=approach_run.approach_index,
        steering_mode=approach_run.steering_mode,
        raw_payload=summary,
    )
    return approach_run


def record_questionnaire_response(
    response_type: str,
    answers: dict,
    *,
    participation_id: int,
    approach_index: Optional[int] = None,
    questionnaire_file: Optional[str] = None,
) -> SaeQuestionnaireResponse:
    if not answers:
        raise AuditContractError("Questionnaire answers are required")
    participation = _participation(participation_id)
    study_run = ensure_study_run(participation.id)
    approach_run = None
    if approach_index is not None:
        approach_run = ensure_approach_run(
            participation.id, approach_index=int(approach_index), conf=_conf(participation)
        )
    response = SaeQuestionnaireResponse(
        study_run_id=study_run.id,
        approach_run_id=approach_run.id if approach_run else None,
        participation_id=participation.id,
        response_type=response_type,
        approach_index=approach_run.approach_index if approach_run else None,
        approach_name=approach_run.approach_name if approach_run else None,
        questionnaire_file=questionnaire_file,
        answers=answers,
        submitted_at=utcnow(),
    )
    db.session.add(response)
    db.session.commit()
    record_event(
        f"{response_type}-questionnaire",
        participation_id=participation_id,
        approach_index=response.approach_index,
        modality="questionnaire",
        raw_payload={
            "response_id": response.id,
            "answer_count": len(answers),
            "questionnaire_file": questionnaire_file,
        },
        allow_no_approach=approach_run is None,
    )
    return response


def record_autosave_snapshot(
    data: dict, *, participation_id: int, iteration: int
) -> SaeSteeringEvent:
    return record_event(
        "autosave",
        participation_id=participation_id,
        iteration=iteration,
        modality="system",
        raw_payload={
            "trigger": data.get("trigger"),
            "liked_movies": data.get("liked_movies", []),
            "feature_adjustments": data.get("feature_adjustments", {}),
            "activity_snapshot": data.get("activity_snapshot", {}),
            "client_timestamp": data.get("timestamp"),
        },
        allow_no_approach=True,
    )


def complete_study_run(participation_id: int) -> None:
    run = ensure_study_run(participation_id)
    run.finished_at = utcnow()
    run.status = "completed"
    db.session.commit()
