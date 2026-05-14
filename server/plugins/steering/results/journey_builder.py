"""Offline journey reconstruction from a ``/export-raw/<guid>`` JSON dump.

This is the pure-function adapter that ``scripts/reconstruct_journey.py``
imports so researchers can replay a participant's chronological journey from
the v1 typed-audit JSON export without needing live database access. It
mirrors the DB-backed
:func:`server.plugins.steering.routes.results.journey.participant_journey`
view but consumes the export shape produced by
:func:`server.plugins.steering.routes.results.views.export_raw_data` instead.

The "schema":"sae-typed-audit.v1" contract used by the export is treated as
authoritative; any new typed-audit table must be lifted into both
``export_raw_data`` and this module if it should show up in offline journey
reconstructions.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any


_NOISE_EVENT_TYPES = {"autosave"}


def _fmt_time_short(iso_ts: str | None) -> str:
    if not iso_ts:
        return "-"
    try:
        return datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except (ValueError, AttributeError):
        return iso_ts[:19] if isinstance(iso_ts, str) else "-"


def _section(event_type: str | None) -> str:
    et = event_type or ""
    if et.startswith("elicitation"):
        return "ELICITATION"
    if et == "approach-complete":
        return "APPROACH"
    if "questionnaire" in et:
        return "QUESTIONNAIRE"
    if et == "autosave":
        return "SYSTEM"
    if et == "study-ended":
        return "FINISH"
    return "STEERING"


def _summarize_event(
    event: dict,
    *,
    feature_adj_by_event: dict[int | None, list[dict]],
    text_query_by_event: dict[int | None, dict],
    feature_search_by_event: dict[int | None, dict],
    movie_feedback_by_event: dict[int | None, dict],
    example_by_event: dict[int | None, dict],
    reset_by_event: dict[int | None, dict],
    rec_set_by_id: dict[int | None, dict],
) -> str:
    raw = event.get("raw_payload") or {}
    event_type = event.get("event_type") or "unknown"
    event_id = event.get("id")
    approach_index = event.get("approach_index")
    approach_label = (
        f"Approach {(approach_index or 0) + 1} ({event.get('approach_name') or '?'})"
        if approach_index is not None
        else "Elicitation"
    )

    if event_type == "elicitation-search":
        return (
            f'Elicitation search: "{raw.get("query") or event.get("search_query") or "?"}" '
            f"-> {raw.get('result_count', '?')} results"
        )
    if event_type == "elicitation-completed":
        movies = raw.get("selected_movies") or []
        titles = ", ".join(row.get("title") or str(row.get("movie_id")) for row in movies)
        return f"Elicitation completed: {len(movies)} selected movies ({titles})"

    if event_type == "feature-adjustment":
        adjustments = feature_adj_by_event.get(event_id, [])
        modality = event.get("modality") or "?"
        return (
            f"{approach_label} iter {event.get('iteration', '?')}: "
            f"{len(adjustments)} feature adjustment(s) via {modality}"
        )
    if event_type == "text-steering-parsed":
        query_row = text_query_by_event.get(event_id) or {}
        query = query_row.get("query") or raw.get("query") or ""
        matches = len(query_row.get("matches") or [])
        return f'{approach_label}: text steering "{query}" -> {matches} match(es)'
    if event_type == "feature-search":
        search_row = feature_search_by_event.get(event_id) or {}
        query = search_row.get("query") or event.get("search_query") or "?"
        return (
            f"{approach_label} iter {event.get('iteration', '?')}: "
            f"feature search '{query}' -> {search_row.get('result_count', '?')} hits"
        )
    if event_type == "movie-feedback":
        feedback = movie_feedback_by_event.get(event_id) or {}
        movie_label = feedback.get("title") or f"movie {feedback.get('movie_id', '?')}"
        return (
            f"{approach_label} iter {event.get('iteration', '?')}: "
            f"{feedback.get('action', '?')} '{movie_label}'"
        )
    if event_type == "recommendations-shown":
        rec_set_id = raw.get("recommendation_set_id")
        rec_set = rec_set_by_id.get(rec_set_id) or {}
        return (
            f"{approach_label} iter {event.get('iteration', '?')}: "
            f"{len(rec_set.get('items') or [])} recommendations shown"
        )
    if event_type == "global-reset":
        reset = reset_by_event.get(event_id) or {}
        return f"{approach_label}: reset ({reset.get('scope', '?')}, {reset.get('trigger', '?')})"
    if event_type == "example-steering-applied":
        example_row = example_by_event.get(event_id) or {}
        movie_count = len(example_row.get("movies") or [])
        return (
            f"{approach_label}: example steering, {movie_count} movies, "
            f"strength={example_row.get('example_strength', '?')}, "
            f"top_k={example_row.get('example_top_k', '?')}"
        )
    if event_type == "approach-order-assigned":
        return (
            f"Approach order assigned: "
            f"{raw.get('effective_order') or raw.get('approach_order') or '?'}"
        )
    if event_type == "phase-complete":
        return f"{approach_label}: phase complete"
    if event_type == "study-ended":
        return "Study ended"
    return event_type


def build_journey(participant: dict, include_noise: bool = False) -> dict[str, Any]:
    """Build a chronological timeline + summary for one participant export row.

    Args:
        participant: One element of ``export["participants"]`` from
            ``/export-raw/<guid>``.
        include_noise: When ``False`` (default), noise-class events
            (``autosave``) are filtered out and counted in
            ``summary["noise_hidden"]``.

    Returns:
        Dict with two keys:

        - ``timeline``: list of ``{ts, ts_short, section, summary, type}``
          ordered by timestamp.
        - ``summary``: aggregate counts and per-phase breakdown matching the
          shape consumed by ``scripts/reconstruct_journey.py``.
    """
    events: list[dict] = participant.get("steering_events") or []
    feature_adjustments: list[dict] = participant.get("feature_adjustments") or []
    feature_searches: list[dict] = participant.get("feature_searches") or []
    text_queries: list[dict] = participant.get("text_steering_queries") or []
    movie_feedback: list[dict] = participant.get("movie_feedback") or []
    example_steerings: list[dict] = participant.get("example_steerings") or []
    reset_actions: list[dict] = participant.get("reset_actions") or []
    recommendation_sets: list[dict] = participant.get("recommendation_sets") or []
    approach_runs: list[dict] = participant.get("approach_runs") or []

    feature_adj_by_event: dict[int | None, list[dict]] = {}
    for adj in feature_adjustments:
        feature_adj_by_event.setdefault(adj.get("event_id"), []).append(adj)

    text_query_by_event = {row.get("event_id"): row for row in text_queries}
    feature_search_by_event = {row.get("event_id"): row for row in feature_searches}
    movie_feedback_by_event = {row.get("event_id"): row for row in movie_feedback}
    example_by_event = {row.get("event_id"): row for row in example_steerings}
    reset_by_event = {row.get("event_id"): row for row in reset_actions}
    rec_set_by_id = {row.get("id"): row for row in recommendation_sets}

    timeline: list[dict] = []
    type_counts: Counter[str] = Counter()
    noise_hidden = 0

    for event in events:
        event_type = event.get("event_type") or "unknown"
        if not include_noise and event_type in _NOISE_EVENT_TYPES:
            noise_hidden += 1
            continue
        timeline.append(
            {
                "ts": event.get("created_at"),
                "ts_short": _fmt_time_short(event.get("created_at")),
                "section": _section(event_type),
                "type": event_type,
                "summary": _summarize_event(
                    event,
                    feature_adj_by_event=feature_adj_by_event,
                    text_query_by_event=text_query_by_event,
                    feature_search_by_event=feature_search_by_event,
                    movie_feedback_by_event=movie_feedback_by_event,
                    example_by_event=example_by_event,
                    reset_by_event=reset_by_event,
                    rec_set_by_id=rec_set_by_id,
                ),
            }
        )
        type_counts[event_type] += 1

    timeline.sort(key=lambda entry: entry["ts"] or "")

    phases = []
    for approach in approach_runs:
        approach_index = approach.get("approach_index")
        approach_events_ids = {
            event.get("id") for event in events if event.get("approach_index") == approach_index
        }
        likes = sum(
            1
            for feedback in movie_feedback
            if feedback.get("approach_index") == approach_index and feedback.get("action") == "like"
        )
        dislikes = sum(
            1
            for feedback in movie_feedback
            if feedback.get("approach_index") == approach_index
            and feedback.get("action") == "dislike"
        )
        slider_adjustments = sum(
            1 for adj in feature_adjustments if adj.get("event_id") in approach_events_ids
        )
        searches = sum(
            1
            for search in feature_searches
            if search.get("event_id") in approach_events_ids
        )
        approach_name = approach.get("approach_name") or f"approach_{approach_index}"
        approach_summary = approach.get("summary") or {}
        iterations_used = approach_summary.get("iterations_used")
        if iterations_used is None:
            iterations_used = max(
                (
                    event.get("iteration") or 0
                    for event in events
                    if event.get("approach_index") == approach_index
                ),
                default=0,
            )
        phases.append(
            {
                "phase": approach_index,
                "models": [approach_name],
                "iterations": iterations_used,
                "likes": likes,
                "dislikes": dislikes,
                "slider_adjustments": slider_adjustments,
                "searches": searches,
            }
        )

    return {
        "timeline": timeline,
        "summary": {
            "total_interactions": len(timeline),
            "type_counts": dict(type_counts),
            "phases": phases,
            "noise_hidden": noise_hidden,
        },
    }
