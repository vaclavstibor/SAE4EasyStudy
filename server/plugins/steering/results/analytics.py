"""Results dashboard analytics built from typed SAE audit tables.

Every aggregate is computed from typed columns, never from JSON blobs.
The shape is intentionally generic so adding a new modality, questionnaire,
or chart does not require changing the analytics contract.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from sqlalchemy import func

from server.platform.persistence.base_models import Participation, UserStudy
from server.platform.persistence.db import db
from server.platform.shared.common import load_user_study_config_by_guid
from server.plugins.steering.persistence.models import (
    SaeApproachRun,
    SaeExampleSteering,
    SaeFeatureAdjustment,
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

from ..constants import PROLIFIC_BASE_URL
from ..study_config import approach_label, normalize_study_config

# Helpers


def safe_parse_json(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def build_prolific_block(extra_data, study_config=None):
    extra = safe_parse_json(extra_data) if not isinstance(extra_data, dict) else extra_data
    pid = (extra or {}).get("PROLIFIC_PID")
    study_id = (extra or {}).get("PROLIFIC_STUDY_ID")
    session_id = (extra or {}).get("PROLIFIC_SESSION_ID")
    completion_code = (
        (study_config or {}).get("prolific_code") if isinstance(study_config, dict) else None
    )
    completion_url = f"{PROLIFIC_BASE_URL}?cc={completion_code}" if completion_code else None
    return {
        "pid": pid,
        "study_id": study_id,
        "session_id": session_id,
        "completion_code": completion_code,
        "completion_url": completion_url,
    }


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    cleaned = [v for v in values if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def _round(value, digits=3):
    if value is None:
        return None
    return round(value, digits)


# Modality observations


def _selected_rank_distribution(user_study_id: int) -> dict:
    """Histogram of rank for liked movies, grouped by approach.

    The user_study_id filter keeps the query restricted to the current study.
    Buckets are inclusive ranks (1, 2, 3, ...). Missing ranks (legacy data)
    are dropped to keep the chart honest.
    """
    rows = (
        db.session.query(
            SaeMovieFeedback.approach_index,
            SaeMovieFeedback.approach_name,
            SaeMovieFeedback.rank,
            func.count(SaeMovieFeedback.id),
        )
        .join(SaeStudyRun, SaeMovieFeedback.study_run_id == SaeStudyRun.id)
        .filter(
            SaeStudyRun.user_study_id == user_study_id,
            SaeMovieFeedback.action == "like",
            SaeMovieFeedback.rank.isnot(None),
        )
        .group_by(
            SaeMovieFeedback.approach_index,
            SaeMovieFeedback.approach_name,
            SaeMovieFeedback.rank,
        )
        .all()
    )
    by_approach: dict[str, dict] = {}
    for approach_index, approach_name, rank, cnt in rows:
        bucket = by_approach.setdefault(
            str(approach_index),
            {"label": approach_name, "rank_counts": {}, "total": 0},
        )
        bucket["rank_counts"][str(int(rank))] = int(cnt)
        bucket["total"] += int(cnt)
    return by_approach


def _slider_movement_by_position(user_study_id: int) -> dict:
    """Mean absolute slider delta per cluster label, grouped by approach.

    "Position" here means the cluster that was moved (the most stable
    label-based grouping; ``cluster_label`` is captured by the audit row).
    Returns the top 20 most-moved clusters per approach so the chart stays
    legible.
    """
    rows = (
        db.session.query(
            SaeApproachRun.approach_index,
            SaeApproachRun.approach_name,
            SaeFeatureAdjustment.cluster_label,
            func.avg(func.abs(SaeFeatureAdjustment.delta)).label("mean_abs"),
            func.count(SaeFeatureAdjustment.id).label("n"),
        )
        .join(SaeApproachRun, SaeFeatureAdjustment.approach_run_id == SaeApproachRun.id)
        .join(SaeStudyRun, SaeApproachRun.study_run_id == SaeStudyRun.id)
        .filter(
            SaeStudyRun.user_study_id == user_study_id,
            SaeFeatureAdjustment.cluster_label.isnot(None),
        )
        .group_by(
            SaeApproachRun.approach_index,
            SaeApproachRun.approach_name,
            SaeFeatureAdjustment.cluster_label,
        )
        .all()
    )
    by_approach: dict[str, dict] = {}
    for approach_index, approach_name, label, mean_abs, n in rows:
        bucket = by_approach.setdefault(
            str(approach_index),
            {"label": approach_name, "clusters": []},
        )
        bucket["clusters"].append(
            {
                "cluster_label": label,
                "mean_abs_delta": _round(_to_float(mean_abs), 4),
                "n": int(n or 0),
            }
        )
    for bucket in by_approach.values():
        bucket["clusters"].sort(key=lambda r: r["mean_abs_delta"] or 0, reverse=True)
        bucket["clusters"] = bucket["clusters"][:20]
    return by_approach


def _text_prompt_cluster_mappings(user_study_id: int) -> list[dict]:
    """Return one row per (query_text, cluster_label) with aggregate weight.

    This is what answers the researcher question: "what prompt mapped to
    which cluster, with how much weight?" Each row contains the prompt
    text, the cluster label, the mean signed weight, and the number of
    times the pairing was observed.
    """
    rows = (
        db.session.query(
            SaeTextSteeringQuery.query_text,
            SaeTextSteeringMatch.cluster_id,
            SaeTextSteeringMatch.label,
            func.avg(SaeTextSteeringMatch.weight).label("mean_weight"),
            func.count(SaeTextSteeringMatch.id).label("n"),
        )
        .join(
            SaeTextSteeringMatch,
            SaeTextSteeringMatch.query_id == SaeTextSteeringQuery.id,
        )
        .join(SaeStudyRun, SaeTextSteeringQuery.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .group_by(
            SaeTextSteeringQuery.query_text,
            SaeTextSteeringMatch.cluster_id,
            SaeTextSteeringMatch.label,
        )
        .order_by(func.count(SaeTextSteeringMatch.id).desc())
        .limit(50)
        .all()
    )
    return [
        {
            "query_text": query,
            "cluster_id": cluster_id,
            "cluster_label": label,
            "mean_weight": _round(_to_float(mean_w), 3),
            "n": int(n or 0),
        }
        for query, cluster_id, label, mean_w, n in rows
    ]


# Approach and selection overview


def _approach_overview(user_study_id: int, config_models: list[dict]) -> dict:
    """One row per approach with the metrics shown in the overview table."""
    overview: dict[str, dict] = {}
    approach_rows = (
        SaeApproachRun.query.join(SaeStudyRun, SaeApproachRun.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .all()
    )
    grouped: dict[str, list[SaeApproachRun]] = defaultdict(list)
    for row in approach_rows:
        grouped[str(row.approach_index)].append(row)
    for key, rows in grouped.items():
        mean_abs = (
            db.session.query(func.avg(func.abs(SaeFeatureAdjustment.delta)))
            .filter(SaeFeatureAdjustment.approach_run_id.in_([r.id for r in rows]))
            .scalar()
        )
        nonzero = (
            db.session.query(func.count(SaeFeatureAdjustment.id))
            .filter(SaeFeatureAdjustment.approach_run_id.in_([r.id for r in rows]))
            .scalar()
        ) or 0
        iterations = [r.iterations_used for r in rows if r.iterations_used is not None]
        sliders = [int(r.total_slider_changes or 0) for r in rows]
        overview[key] = {
            "label": rows[0].approach_name,
            "participants": len(rows),
            "mean_iterations": _round(_mean(iterations), 3),
            "mean_abs_adjustment": _round(_to_float(mean_abs), 4),
            "mean_nonzero_adjustments": _round(nonzero / len(rows) if rows else None, 3),
            "mean_total_slider_changes": _round(_mean(sliders), 3),
        }
    for i, model in enumerate(config_models):
        overview.setdefault(
            str(i),
            {
                "label": model.get("name", approach_label(i)),
                "participants": 0,
                "mean_iterations": None,
                "mean_abs_adjustment": None,
                "mean_nonzero_adjustments": None,
                "mean_total_slider_changes": None,
            },
        )
    return overview


def _selection_dynamics(user_study_id: int) -> dict:
    """Likes/neutrals per approach used by the Selection Dynamics table."""
    dynamics: dict[str, dict] = {}
    rows = (
        db.session.query(
            SaeApproachRun.approach_index,
            SaeApproachRun.approach_name,
            SaeApproachRun.participation_id,
            SaeMovieFeedback.action,
            func.count(SaeMovieFeedback.id),
        )
        .join(SaeStudyRun, SaeApproachRun.study_run_id == SaeStudyRun.id)
        .outerjoin(SaeMovieFeedback, SaeMovieFeedback.approach_run_id == SaeApproachRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .group_by(
            SaeApproachRun.approach_index,
            SaeApproachRun.approach_name,
            SaeApproachRun.participation_id,
            SaeMovieFeedback.action,
        )
        .all()
    )
    grouped: dict[str, dict] = {}
    for approach_index, approach_name, participation_id, action, cnt in rows:
        key = str(approach_index)
        bucket = grouped.setdefault(
            key,
            {
                "label": approach_name,
                "participants_with_feedback": set(),
                "total_like_events": 0,
                "total_neutral_events": 0,
                "_likes_per_participant": defaultdict(int),
            },
        )
        if action is None:
            continue
        if action == "like":
            bucket["total_like_events"] += int(cnt)
            bucket["_likes_per_participant"][participation_id] += int(cnt)
            bucket["participants_with_feedback"].add(participation_id)
        elif action == "neutral":
            bucket["total_neutral_events"] += int(cnt)
            bucket["participants_with_feedback"].add(participation_id)
    for key, bucket in grouped.items():
        likes = list(bucket["_likes_per_participant"].values())
        dynamics[key] = {
            "label": bucket["label"],
            "participants_with_feedback": len(bucket["participants_with_feedback"]),
            "total_like_events": bucket["total_like_events"],
            "total_neutral_events": bucket["total_neutral_events"],
            "mean_like_events_per_participant": _round(_mean(likes), 3),
        }
    return dynamics


# Questionnaire monitor


def _infer_field_kind(values: list) -> str:
    """Best-effort field classification: likert, numeric, categorical, text."""
    if not values:
        return "empty"
    floats: list[float] = []
    for v in values:
        try:
            floats.append(float(v))
        except (TypeError, ValueError):
            floats = []
            break
    if floats:
        if all(1 <= x <= 7 and float(x).is_integer() for x in floats):
            return "likert"
        return "numeric"
    unique = {str(v).strip() for v in values}
    if len(unique) <= 12 and all(len(s) <= 64 for s in unique):
        return "categorical"
    return "text"


def _summarize_field(values: list) -> dict:
    """Build a kind-aware summary for one questionnaire field."""
    kind = _infer_field_kind(values)
    if kind == "likert" or kind == "numeric":
        floats = [float(v) for v in values]
        counts: Counter = Counter()
        for v in values:
            counts[str(v)] += 1
        return {
            "kind": kind,
            "n": len(floats),
            "mean": _round(sum(floats) / len(floats), 3) if floats else None,
            "min": min(floats) if floats else None,
            "max": max(floats) if floats else None,
            "counts": dict(counts),
        }
    if kind == "categorical":
        counts = Counter(str(v).strip() for v in values)
        return {"kind": kind, "n": sum(counts.values()), "counts": dict(counts)}
    samples = [str(v).strip() for v in values if str(v).strip()]
    return {
        "kind": "text",
        "n": len(samples),
        "samples": samples[:10],
    }


def _questionnaire_monitor(user_study_id: int) -> dict:
    """Auto-discover every questionnaire and aggregate every answer field.

    Adding a new questionnaire (template HTML + answers JSON) requires no
    changes here: each questionnaire file becomes its own group, and every
    field inside the JSON answers turns into a summary row whose chart is
    chosen from the inferred kind (likert/numeric/categorical/text).
    """
    rows = (
        SaeQuestionnaireResponse.query.join(
            SaeStudyRun, SaeQuestionnaireResponse.study_run_id == SaeStudyRun.id
        )
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .all()
    )
    groups: dict[str, dict] = {}
    for row in rows:
        answers = row.answers or {}
        if not isinstance(answers, dict):
            continue
        file_key = row.questionnaire_file or f"{row.response_type or 'unknown'} (no file)"
        group = groups.setdefault(
            file_key,
            {
                "questionnaire_file": row.questionnaire_file,
                "response_type": row.response_type,
                "approach_indices": set(),
                "responses": 0,
                "_fields": defaultdict(list),
            },
        )
        if row.approach_index is not None:
            group["approach_indices"].add(int(row.approach_index))
        group["responses"] += 1
        for field, value in answers.items():
            if value is None or value == "":
                continue
            group["_fields"][field].append(value)
    monitor: dict[str, dict] = {}
    for key, group in groups.items():
        fields = group.pop("_fields")
        monitor[key] = {
            "questionnaire_file": group["questionnaire_file"],
            "response_type": group["response_type"],
            "approach_indices": sorted(group["approach_indices"]),
            "responses": group["responses"],
            "fields": {name: _summarize_field(values) for name, values in fields.items()},
        }
    return monitor


# Main entry


def build_results_payload(guid):
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return {"error": "Study not found"}, 404

    study_config = normalize_study_config(load_user_study_config_by_guid(guid))
    config_models = list(study_config.get("models", []))
    completed_participations = Participation.query.filter(
        (Participation.time_finished.isnot(None)) & (Participation.user_study_id == user_study.id)
    ).all()
    all_participations = Participation.query.filter(
        Participation.user_study_id == user_study.id
    ).all()

    approach_overview = _approach_overview(user_study.id, config_models)
    selection_dynamics = _selection_dynamics(user_study.id)
    rank_distribution = _selected_rank_distribution(user_study.id)
    slider_movement = _slider_movement_by_position(user_study.id)
    text_mappings = _text_prompt_cluster_mappings(user_study.id)
    questionnaire_monitor = _questionnaire_monitor(user_study.id)

    modality_rows = (
        db.session.query(SaeSteeringEvent.modality, func.count(SaeSteeringEvent.id))
        .join(SaeStudyRun, SaeSteeringEvent.study_run_id == SaeStudyRun.id)
        .filter(
            SaeStudyRun.user_study_id == user_study.id,
            SaeSteeringEvent.modality.isnot(None),
        )
        .group_by(SaeSteeringEvent.modality)
        .all()
    )
    modality_counter = Counter()
    for modality, cnt in modality_rows:
        modality_counter[modality] = int(cnt or 0)

    total_resets = (
        db.session.query(func.count(SaeResetAction.id))
        .join(SaeStudyRun, SaeResetAction.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study.id)
        .scalar()
    ) or 0
    total_text_queries = (
        db.session.query(func.count(SaeTextSteeringQuery.id))
        .join(SaeStudyRun, SaeTextSteeringQuery.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study.id)
        .scalar()
    ) or 0
    total_example_events = (
        db.session.query(func.count(SaeExampleSteering.id))
        .join(SaeStudyRun, SaeExampleSteering.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study.id)
        .scalar()
    ) or 0
    total_recommendation_impressions = (
        db.session.query(func.count(SaeRecommendationItem.id))
        .join(
            SaeRecommendationSet,
            SaeRecommendationItem.recommendation_set_id == SaeRecommendationSet.id,
        )
        .join(SaeStudyRun, SaeRecommendationSet.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study.id)
        .scalar()
    ) or 0
    searched_then_adjusted = (
        db.session.query(func.count(SaeFeatureAdjustment.id))
        .join(SaeStudyRun, SaeFeatureAdjustment.study_run_id == SaeStudyRun.id)
        .filter(
            SaeStudyRun.user_study_id == user_study.id,
            SaeFeatureAdjustment.applied_via == "search",
        )
        .scalar()
    ) or 0

    participants_table = []
    for participant in all_participations:
        study_run = SaeStudyRun.query.filter(SaeStudyRun.participation_id == participant.id).first()
        duration_sec = None
        if participant.time_joined and participant.time_finished:
            duration_sec = int(
                (participant.time_finished - participant.time_joined).total_seconds()
            )
        questionnaire_count = (
            db.session.query(func.count(SaeQuestionnaireResponse.id))
            .filter(SaeQuestionnaireResponse.study_run_id == study_run.id)
            .scalar()
            if study_run
            else 0
        ) or 0
        participants_table.append(
            {
                "participation_id": participant.id,
                "uuid": participant.uuid,
                "email": participant.participant_email,
                "language": participant.language,
                "status": "completed" if participant.time_finished else "in_progress",
                "time_joined": (
                    participant.time_joined.isoformat() if participant.time_joined else None
                ),
                "time_finished": (
                    participant.time_finished.isoformat() if participant.time_finished else None
                ),
                "duration_sec": duration_sec,
                "prolific": build_prolific_block(participant.extra_data, study_config),
                "effective_order": study_run.effective_order if study_run else [],
                "questionnaire_responses": int(questionnaire_count),
            }
        )

    return {
        "study_guid": guid,
        "sample": {
            "participants_total": len(all_participations),
            "participants_completed": len(completed_participations),
            "participants_in_progress": (len(all_participations) - len(completed_participations)),
        },
        "approaches": {
            "labels": {
                str(i): model.get("name", approach_label(i))
                for i, model in enumerate(config_models)
            },
            "overview": approach_overview,
            "selection_dynamics": selection_dynamics,
            "rank_distribution": rank_distribution,
        },
        "modalities": {
            "modality_usage": dict(modality_counter),
            "slider_movement": slider_movement,
            "text_prompt_mappings": text_mappings,
        },
        "structured_events": {
            "reset_count": int(total_resets),
            "text_queries_count": int(total_text_queries),
            "example_event_count": int(total_example_events),
            "recommendation_impressions": int(total_recommendation_impressions),
            "searched_then_adjusted_count": int(searched_then_adjusted),
        },
        "questionnaires": questionnaire_monitor,
        "participants_table": participants_table,
        "prolific": build_prolific_block(None, study_config),
    }, 200
