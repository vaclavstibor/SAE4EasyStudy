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

# Canonical modality names that the dashboard knows how to render. The order
# determines display order inside each approach card. New modalities only
# need an entry in ``_MODALITY_LABELS`` + a branch in
# ``_modality_metrics_for_approach``; the frontend renders them
# data-drivenly.
_MODALITY_LABELS: dict[str, str] = {
    "sliders": "Slider steering",
    "toggles": "Toggle steering",
    "text": "Text steering",
    "examples": "Example movies",
    "reset": "Reset events",
}

# Placeholder cluster labels written when the steering source could not
# attach a human-readable LLM name (most commonly: features touched by the
# text pipeline). They are filtered out of *cluster*-based aggregates but
# still counted as raw adjustments.
_PLACEHOLDER_CLUSTER_LIKE_PATTERNS = (
    "feature\\_%",
    "cluster\\_%",
    "feature cluster\\_%",
)

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


def _selected_rank_distribution(user_study_id: int, config_models: list[dict]) -> dict:
    """Histogram of rank for liked movies, grouped by approach identity.

    Cross-participant aggregations MUST group by ``SaeApproachRun.approach_id``
    (stable identity from the study config), NOT by
    ``SaeMovieFeedback.approach_index`` (per-participant phase position),
    because the approach order is randomized per participant.

    The user_study_id filter keeps the query restricted to the current study.
    Buckets are inclusive ranks (1, 2, 3, ...). Missing ranks (legacy data)
    are dropped to keep the chart honest. The returned dict preserves the
    config-models order (Python dicts are insertion-ordered), so the
    dashboard renders approaches A / B / ... left-to-right consistently.
    """
    rows = (
        db.session.query(
            SaeApproachRun.approach_id,
            SaeApproachRun.approach_name,
            SaeMovieFeedback.rank,
            func.count(SaeMovieFeedback.id),
        )
        .join(SaeApproachRun, SaeMovieFeedback.approach_run_id == SaeApproachRun.id)
        .join(SaeStudyRun, SaeMovieFeedback.study_run_id == SaeStudyRun.id)
        .filter(
            SaeStudyRun.user_study_id == user_study_id,
            SaeMovieFeedback.action == "like",
            SaeMovieFeedback.rank.isnot(None),
        )
        .group_by(
            SaeApproachRun.approach_id,
            SaeApproachRun.approach_name,
            SaeMovieFeedback.rank,
        )
        .all()
    )
    by_approach: dict[str, dict] = {}
    for i, model in enumerate(config_models):
        approach_id = str(model.get("id") or f"approach_{i + 1}")
        by_approach[approach_id] = {
            "label": model.get("name") or approach_label(i),
            "rank_counts": {},
            "total": 0,
        }
    for approach_id, approach_name, rank, cnt in rows:
        bucket = by_approach.setdefault(
            str(approach_id),
            {"label": approach_name, "rank_counts": {}, "total": 0},
        )
        bucket["rank_counts"][str(int(rank))] = int(cnt)
        bucket["total"] += int(cnt)
    return by_approach


def _approach_run_ids_by_approach(user_study_id: int) -> dict[str, list[int]]:
    """Return ``{approach_id: [approach_run.id, ...]}`` for the given study.

    Used as the join key for every per-approach modality aggregate so each
    function gets a single pre-computed scope instead of re-running the
    same join. Returns an empty dict when no approach runs exist.
    """
    rows = (
        db.session.query(SaeApproachRun.approach_id, SaeApproachRun.id)
        .join(SaeStudyRun, SaeApproachRun.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .all()
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for approach_id, run_id in rows:
        grouped[str(approach_id)].append(int(run_id))
    return grouped


def _slider_movement_by_position(user_study_id: int, config_models: list[dict]) -> dict:
    """Mean absolute slider delta per cluster label, grouped by approach identity.

    Grouped by ``SaeApproachRun.approach_id`` so that cross-participant
    aggregates remain stable under randomized approach order.

    Only rows whose ``cluster_label`` is a non-empty human-readable cluster
    name pass through — bare ``feature_<n>`` / ``cluster_<n>`` placeholders
    are filtered out at the SQL layer because they are the noisy
    "unnamed-feature" entries that swamp the chart without conveying any
    interpretable signal. The dashboard advertises this as a *cluster*
    movement chart, not a feature movement chart.

    Returns the top 20 most-moved clusters per approach so the chart stays
    legible.
    """
    placeholder_filters = [
        ~SaeFeatureAdjustment.cluster_label.ilike(pattern, escape="\\")
        for pattern in _PLACEHOLDER_CLUSTER_LIKE_PATTERNS
    ]
    rows = (
        db.session.query(
            SaeApproachRun.approach_id,
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
            SaeFeatureAdjustment.cluster_label != "",
            *placeholder_filters,
        )
        .group_by(
            SaeApproachRun.approach_id,
            SaeApproachRun.approach_name,
            SaeFeatureAdjustment.cluster_label,
        )
        .all()
    )
    by_approach: dict[str, dict] = {}
    for i, model in enumerate(config_models):
        approach_id = str(model.get("id") or f"approach_{i + 1}")
        by_approach[approach_id] = {
            "label": model.get("name") or approach_label(i),
            "clusters": [],
        }
    for approach_id, approach_name, label, mean_abs, n in rows:
        bucket = by_approach.setdefault(
            str(approach_id),
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


def _approach_modality_breakdown(
    user_study_id: int, config_models: list[dict]
) -> dict:
    """Per-approach modality counts driven by each approach's enabled_modalities.

    Returns a dict keyed by ``approach_id`` (config-models order) with:

    ``label``: approach display name from config.
    ``steering_mode``: from config; surfaced in the section header.
    ``modalities``: dict keyed by canonical modality name. Each entry has
    ``label`` (display string) and ``metrics`` (list of
    ``{key, label, value, fmt?}`` rows). Only modalities listed in the
    approach's ``enabled_modalities`` appear here — an approach using only
    text never gets a "Slider steering" card, even if the audit table
    happens to contain text-driven feature adjustments (those land in the
    text modality's metrics, not the slider's).

    Adding a new modality requires (a) listing it in ``_MODALITY_LABELS``
    and (b) adding a branch in this function. The frontend renders the
    result generically and does not need to know about modality names.
    """
    runs_by_approach = _approach_run_ids_by_approach(user_study_id)

    placeholder_filters = [
        ~SaeFeatureAdjustment.cluster_label.ilike(pattern, escape="\\")
        for pattern in _PLACEHOLDER_CLUSTER_LIKE_PATTERNS
    ]

    def metric(key: str, label: str, value, fmt: str = "int") -> dict:
        return {"key": key, "label": label, "value": value, "fmt": fmt}

    def _slider_metrics(run_ids: list[int]) -> list[dict]:
        if not run_ids:
            return [
                metric("adjustments", "Adjustments", 0),
                metric("distinct_clusters", "Distinct clusters", 0),
                metric("mean_abs_delta", "Mean |Δ|", None, fmt="decimal"),
            ]
        total = (
            db.session.query(func.count(SaeFeatureAdjustment.id))
            .filter(SaeFeatureAdjustment.approach_run_id.in_(run_ids))
            .scalar()
        ) or 0
        distinct = (
            db.session.query(func.count(func.distinct(SaeFeatureAdjustment.cluster_label)))
            .filter(
                SaeFeatureAdjustment.approach_run_id.in_(run_ids),
                SaeFeatureAdjustment.cluster_label.isnot(None),
                SaeFeatureAdjustment.cluster_label != "",
                *placeholder_filters,
            )
            .scalar()
        ) or 0
        mean_abs = (
            db.session.query(func.avg(func.abs(SaeFeatureAdjustment.delta)))
            .filter(SaeFeatureAdjustment.approach_run_id.in_(run_ids))
            .scalar()
        )
        return [
            metric("adjustments", "Adjustments", int(total)),
            metric("distinct_clusters", "Distinct named clusters", int(distinct)),
            metric("mean_abs_delta", "Mean |Δ|", _round(_to_float(mean_abs), 4), fmt="decimal"),
        ]

    def _toggle_metrics(run_ids: list[int]) -> list[dict]:
        if not run_ids:
            return [
                metric("toggle_events", "Toggle events", 0),
                metric("distinct_features", "Distinct features toggled", 0),
            ]
        total = (
            db.session.query(func.count(SaeFeatureAdjustment.id))
            .filter(
                SaeFeatureAdjustment.approach_run_id.in_(run_ids),
                SaeFeatureAdjustment.applied_via == "toggle",
            )
            .scalar()
        ) or 0
        distinct = (
            db.session.query(func.count(func.distinct(SaeFeatureAdjustment.feature_id)))
            .filter(
                SaeFeatureAdjustment.approach_run_id.in_(run_ids),
                SaeFeatureAdjustment.applied_via == "toggle",
            )
            .scalar()
        ) or 0
        return [
            metric("toggle_events", "Toggle events", int(total)),
            metric("distinct_features", "Distinct features toggled", int(distinct)),
        ]

    def _text_metrics(run_ids: list[int]) -> list[dict]:
        if not run_ids:
            return [
                metric("queries", "Prompts submitted", 0),
                metric("distinct_prompts", "Distinct prompts", 0),
                metric("cluster_mappings", "Prompt→cluster matches", 0),
            ]
        queries = (
            db.session.query(func.count(SaeTextSteeringQuery.id))
            .filter(SaeTextSteeringQuery.approach_run_id.in_(run_ids))
            .scalar()
        ) or 0
        distinct = (
            db.session.query(func.count(func.distinct(SaeTextSteeringQuery.query_text)))
            .filter(SaeTextSteeringQuery.approach_run_id.in_(run_ids))
            .scalar()
        ) or 0
        matches = (
            db.session.query(func.count(SaeTextSteeringMatch.id))
            .join(
                SaeTextSteeringQuery,
                SaeTextSteeringMatch.query_id == SaeTextSteeringQuery.id,
            )
            .filter(SaeTextSteeringQuery.approach_run_id.in_(run_ids))
            .scalar()
        ) or 0
        return [
            metric("queries", "Prompts submitted", int(queries)),
            metric("distinct_prompts", "Distinct prompts", int(distinct)),
            metric("cluster_mappings", "Prompt→cluster matches", int(matches)),
        ]

    def _examples_metrics(run_ids: list[int]) -> list[dict]:
        if not run_ids:
            return [metric("example_events", "Example events", 0)]
        total = (
            db.session.query(func.count(SaeExampleSteering.id))
            .filter(SaeExampleSteering.approach_run_id.in_(run_ids))
            .scalar()
        ) or 0
        return [metric("example_events", "Example events", int(total))]

    def _reset_metrics(run_ids: list[int]) -> list[dict]:
        if not run_ids:
            return [metric("reset_count", "Reset events", 0)]
        total = (
            db.session.query(func.count(SaeResetAction.id))
            .filter(SaeResetAction.approach_run_id.in_(run_ids))
            .scalar()
        ) or 0
        return [metric("reset_count", "Reset events", int(total))]

    metric_fns = {
        "sliders": _slider_metrics,
        "toggles": _toggle_metrics,
        "text": _text_metrics,
        "examples": _examples_metrics,
        "reset": _reset_metrics,
    }

    breakdown: dict[str, dict] = {}
    for i, model in enumerate(config_models):
        approach_id = str(model.get("id") or f"approach_{i + 1}")
        run_ids = runs_by_approach.get(approach_id, [])
        enabled = list(model.get("enabled_modalities") or [])
        modalities: dict[str, dict] = {}
        # Preserve _MODALITY_LABELS order regardless of how enabled_modalities
        # was authored, so the frontend renders modalities consistently.
        for canonical_name in _MODALITY_LABELS:
            if canonical_name not in enabled:
                continue
            metric_fn = metric_fns.get(canonical_name)
            if metric_fn is None:
                continue
            modalities[canonical_name] = {
                "label": _MODALITY_LABELS[canonical_name],
                "metrics": metric_fn(run_ids),
            }
        breakdown[approach_id] = {
            "label": model.get("name") or approach_label(i),
            "steering_mode": model.get("steering_mode"),
            "modalities_enabled": enabled,
            "modalities": modalities,
            "participations": len(run_ids),
        }
    return breakdown


def _text_prompt_cluster_mappings(user_study_id: int) -> list[dict]:
    """Return one row per (approach_id, query_text, cluster_label) with aggregate weight.

    The ``approach_id`` is the stable identifier from the study config; the
    dashboard groups the table by approach so participants who only saw a
    text-enabled approach are not mixed with slider-only data.
    """
    rows = (
        db.session.query(
            SaeApproachRun.approach_id,
            SaeApproachRun.approach_name,
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
        .join(SaeApproachRun, SaeTextSteeringQuery.approach_run_id == SaeApproachRun.id)
        .join(SaeStudyRun, SaeTextSteeringQuery.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .group_by(
            SaeApproachRun.approach_id,
            SaeApproachRun.approach_name,
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
            "approach_id": approach_id,
            "approach_label": approach_name,
            "query_text": query,
            "cluster_id": cluster_id,
            "cluster_label": label,
            "mean_weight": _round(_to_float(mean_w), 3),
            "n": int(n or 0),
        }
        for approach_id, approach_name, query, cluster_id, label, mean_w, n in rows
    ]


# Approach and selection overview


def _approach_overview(user_study_id: int, config_models: list[dict]) -> dict:
    """One row per approach (stable identity) with overview metrics.

    Grouped by ``SaeApproachRun.approach_id`` so randomized per-participant
    approach order does not collapse two approaches into a single phase
    bucket. The result dict is keyed by ``approach_id`` and populated in
    config order so the dashboard renders A / B / ... left-to-right.
    """
    overview: dict[str, dict] = {}
    approach_rows = (
        SaeApproachRun.query.join(SaeStudyRun, SaeApproachRun.study_run_id == SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .all()
    )
    grouped: dict[str, list[SaeApproachRun]] = defaultdict(list)
    for row in approach_rows:
        grouped[str(row.approach_id)].append(row)
    # Seed in config order so the rendering order is deterministic.
    for i, model in enumerate(config_models):
        approach_id = str(model.get("id") or f"approach_{i + 1}")
        rows = grouped.get(approach_id, [])
        if rows:
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
            overview[approach_id] = {
                "label": model.get("name") or rows[0].approach_name or approach_label(i),
                "participants": len(rows),
                "mean_iterations": _round(_mean(iterations), 3),
                "mean_abs_adjustment": _round(_to_float(mean_abs), 4),
                "mean_nonzero_adjustments": _round(nonzero / len(rows) if rows else None, 3),
                "mean_total_slider_changes": _round(_mean(sliders), 3),
            }
        else:
            overview[approach_id] = {
                "label": model.get("name", approach_label(i)),
                "participants": 0,
                "mean_iterations": None,
                "mean_abs_adjustment": None,
                "mean_nonzero_adjustments": None,
                "mean_total_slider_changes": None,
            }
    # Surface any approach_id we have data for that is no longer in the
    # config (stale data after a settings edit) — at the end, in observed
    # order, so the researcher can still see the rows.
    for approach_id, rows in grouped.items():
        if approach_id in overview:
            continue
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
        overview[approach_id] = {
            "label": rows[0].approach_name or approach_id,
            "participants": len(rows),
            "mean_iterations": _round(_mean(iterations), 3),
            "mean_abs_adjustment": _round(_to_float(mean_abs), 4),
            "mean_nonzero_adjustments": _round(nonzero / len(rows) if rows else None, 3),
            "mean_total_slider_changes": _round(_mean(sliders), 3),
        }
    return overview


def _selection_dynamics(user_study_id: int, config_models: list[dict]) -> dict:
    """Likes/neutrals per approach (stable identity) for the Selection Dynamics table.

    Grouped by ``SaeApproachRun.approach_id`` so randomized per-participant
    order does not produce two rows for the same approach. Returned in
    config-models order so the table is stable across reloads.
    """
    dynamics: dict[str, dict] = {}
    rows = (
        db.session.query(
            SaeApproachRun.approach_id,
            SaeApproachRun.approach_name,
            SaeApproachRun.participation_id,
            SaeMovieFeedback.action,
            func.count(SaeMovieFeedback.id),
        )
        .join(SaeStudyRun, SaeApproachRun.study_run_id == SaeStudyRun.id)
        .outerjoin(SaeMovieFeedback, SaeMovieFeedback.approach_run_id == SaeApproachRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .group_by(
            SaeApproachRun.approach_id,
            SaeApproachRun.approach_name,
            SaeApproachRun.participation_id,
            SaeMovieFeedback.action,
        )
        .all()
    )
    grouped: dict[str, dict] = {}
    for i, model in enumerate(config_models):
        approach_id = str(model.get("id") or f"approach_{i + 1}")
        grouped[approach_id] = {
            "label": model.get("name") or approach_label(i),
            "participants_with_feedback": set(),
            "total_like_events": 0,
            "total_neutral_events": 0,
            "_likes_per_participant": defaultdict(int),
        }
    for approach_id, approach_name, participation_id, action, cnt in rows:
        key = str(approach_id)
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
    """Auto-discover every questionnaire submission identity and aggregate answers.

    Each group is keyed by ``(questionnaire_file, approach_index)``. A study
    where two approaches both use the same per-approach questionnaire file
    produces two separate groups; the final questionnaire (approach_index is
    None) is its own group. Within each group every field in the JSON answers
    turns into a summary row whose chart is chosen from the inferred kind
    (likert / numeric / categorical / text). The frontend renders one section
    per group.
    """
    rows = (
        SaeQuestionnaireResponse.query.join(
            SaeStudyRun, SaeQuestionnaireResponse.study_run_id == SaeStudyRun.id
        )
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .all()
    )
    groups: dict[tuple[str, object], dict] = {}
    for row in rows:
        answers = row.answers or {}
        if not isinstance(answers, dict):
            continue
        file_label = row.questionnaire_file or f"{row.response_type or 'unknown'} (no file)"
        approach_key: object = (
            int(row.approach_index) if row.approach_index is not None else None
        )
        group_key = (file_label, approach_key)
        group = groups.setdefault(
            group_key,
            {
                "questionnaire_file": row.questionnaire_file,
                "response_type": row.response_type,
                "approach_index": approach_key,
                "approach_name": row.approach_name,
                "responses": 0,
                "_fields": defaultdict(list),
            },
        )
        # First-seen approach_name wins; downstream display prefers approach_index.
        if not group["approach_name"] and row.approach_name:
            group["approach_name"] = row.approach_name
        group["responses"] += 1
        for field, value in answers.items():
            if value is None or value == "":
                continue
            group["_fields"][field].append(value)

    # Stable display order: per-approach groups first (sorted by approach_index),
    # then the final questionnaire group (approach_index is None) at the bottom.
    def _sort_key(item: tuple) -> tuple:
        (file_label, approach_key), _ = item
        if approach_key is None:
            return (1, 0, file_label)
        return (0, int(approach_key), file_label)

    monitor: dict[str, dict] = {}
    for (file_label, approach_key), group in sorted(groups.items(), key=_sort_key):
        fields = group.pop("_fields")
        if approach_key is None:
            display_key = f"{file_label} · Final"
        else:
            display_key = f"{file_label} · Approach #{int(approach_key) + 1}"
        monitor[display_key] = {
            "questionnaire_file": group["questionnaire_file"],
            "response_type": group["response_type"],
            "approach_index": group["approach_index"],
            "approach_name": group["approach_name"],
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
    selection_dynamics = _selection_dynamics(user_study.id, config_models)
    rank_distribution = _selected_rank_distribution(user_study.id, config_models)
    slider_movement = _slider_movement_by_position(user_study.id, config_models)
    text_mappings = _text_prompt_cluster_mappings(user_study.id)
    modality_breakdown = _approach_modality_breakdown(user_study.id, config_models)
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
        # Aggregate attention-check pass/total across every questionnaire this
        # participant submitted that *declared* a spec. Submissions whose
        # questionnaire file ships no spec contribute neither to the pass
        # count nor to the total (``attention_check_passed`` is NULL for
        # those rows).
        attn_passed = 0
        attn_total = 0
        if study_run:
            attn_rows = (
                db.session.query(SaeQuestionnaireResponse.attention_check_passed)
                .filter(SaeQuestionnaireResponse.study_run_id == study_run.id)
                .filter(SaeQuestionnaireResponse.attention_check_passed.isnot(None))
                .all()
            )
            attn_total = len(attn_rows)
            attn_passed = sum(1 for (passed,) in attn_rows if passed)
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
                "attention_checks": {
                    "passed": attn_passed,
                    "total": attn_total,
                },
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
            "modality_breakdown": modality_breakdown,
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
