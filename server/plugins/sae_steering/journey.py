"""Shared journey/interaction utilities.

This module is the single source of truth for:

  * NOISE_TYPES   - high-frequency interaction types that should be hidden
                    from human-readable journey reconstructions.
  * PHASE_LABELS  - mapping of interaction type -> human-readable phase label.
  * describe_interaction(ix) - build a one-line summary for an interaction.
  * scrub_interaction(ix)    - drop bulky payload pieces (e.g. viewport `items[]`)
                               from raw export rows so historical data also comes
                               out clean.
  * build_journey(rows)      - turn a list of interaction rows into a structured
                               timeline + per-phase summary used by both the
                               admin UI and the standalone CLI script.

The CLI script ``scripts/reconstruct_journey.py`` re-imports from this module so
the format stays identical between server and offline analysis.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


NOISE_TYPES = {"on-input", "changed-viewport"}
NOISE_INPUT_TYPES = {"mouse-enter", "mouse-leave"}


PHASE_LABELS: Dict[str, str] = {
    "feature-adjustment": "STEERING",
    "recommendations-shown": "STEERING",
    "preferences-approved": "STEERING",
    "movie-feedback": "STEERING",
    "search-slider-adjusted": "STEERING",
    "slider-adjusted": "STEERING",
    "slider-restored-from-history": "STEERING",
    "phase-complete": "PHASE",
    "phase-questionnaire": "QUESTIONNAIRE",
    "final-questionnaire": "QUESTIONNAIRE",
    "elicitation-completed": "ELICITATION",
    "elicitation-search": "ELICITATION",
    "selected-item": "ELICITATION",
    "deselected-item": "ELICITATION",
    "loaded-page": "NAVIGATION",
    "study-ended": "FINISH",
    "approach-order-assigned": "SETUP",
    "autosave": "SYSTEM",
    "ui-event": "UI",
}


def fmt_time(ts: Optional[str]) -> str:
    """Parse ISO timestamp and return readable HH:MM:SS, leaving the input
    string alone if it cannot be parsed."""
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ts[:19] if isinstance(ts, str) else "-"


def scrub_interaction(ix: Dict[str, Any]) -> Dict[str, Any]:
    """Strip the bulky `items[]` array out of `data.context.extra` and
    drop hover noise from `on-input`.  Returns the same dict (mutated)
    with no-op semantics if nothing matches.

    Used both at request time (to keep the DB lean) and at export time
    (so historical rows also come out clean)."""
    if not isinstance(ix, dict):
        return ix
    data = ix.get("data")
    if not isinstance(data, dict):
        return ix
    ctx = data.get("context")
    if isinstance(ctx, dict):
        extra = ctx.get("extra")
        if isinstance(extra, dict) and isinstance(extra.get("items"), list):
            extra["items_count"] = len(extra["items"])
            extra.pop("items", None)
    return ix


def is_noise(ix: Dict[str, Any]) -> bool:
    """Return True for interactions that are pure hover/viewport noise and
    should be dropped from the raw JSON export by default.

    Conservative: only hides mouse-enter / mouse-leave ``on-input`` events.
    Clicks, keyboard input, and ``changed-viewport`` rows are kept (the
    latter still gets its bulky ``items[]`` stripped by
    :func:`scrub_interaction`)."""
    if ix.get("type") != "on-input":
        return False
    data = ix.get("data") or {}
    return isinstance(data, dict) and data.get("input_type") in NOISE_INPUT_TYPES


def is_timeline_noise(ix: Dict[str, Any]) -> bool:
    """Stricter variant used by the admin journey timeline — drops every
    ``on-input`` and ``changed-viewport`` row because the chronological
    view is otherwise unreadable.  The user's Likert clicks / slider drags
    come through dedicated event types, so we don't lose signal here."""
    if ix.get("type") in NOISE_TYPES:
        return True
    return is_noise(ix)


def describe_interaction(ix: Dict[str, Any]) -> str:
    """Return a human-readable one-liner for an interaction row."""
    t = ix.get("type", "?")
    d = ix.get("data") or {}
    if not isinstance(d, dict):
        d = {}

    if t == "loaded-page":
        return f"Opened page: {d.get('page', '?')}"

    if t == "selected-item":
        title = d.get("item", {}).get("title") if isinstance(d.get("item"), dict) else None
        title = title or d.get("title", "?")
        action = d.get("action", "select")
        return f"Elicitation: {action} \"{title}\""

    if t == "deselected-item":
        title = d.get("item", {}).get("title") if isinstance(d.get("item"), dict) else None
        title = title or d.get("title", "?")
        return f"Elicitation: deselect \"{title}\""

    if t == "elicitation-search":
        return f"Elicitation search: \"{d.get('query', '?')}\" -> {d.get('result_count', '?')} results"

    if t == "elicitation-completed":
        liked = d.get("liked_movies") or d.get("selected_movies") or []
        return f"Elicitation finished ({len(liked)} movies liked)"

    if t == "feature-adjustment":
        adj_count = len(d.get("adjustments", {}) or {})
        search_hist = (d.get("search_context", {}) or {}).get("search_history", [])
        searches = ""
        if search_hist:
            queries = [s.get("query", "") for s in search_hist if isinstance(s, dict)]
            searches = f" (searched: {queries})"
        return (
            f"Feature adjustment: phase={d.get('phase', '?')} "
            f"iter={d.get('iteration', '?')} "
            f"model={d.get('model_id', '?')}, "
            f"{adj_count} sliders changed{searches}"
        )

    if t == "recommendations-shown":
        cnt = len(d.get("movie_ids", []) or [])
        return (
            f"Recommendations shown: phase={d.get('phase', '?')} "
            f"iter={d.get('iteration', '?')}, {cnt} movies, "
            f"model={d.get('model_id', '?')}"
        )

    if t == "movie-feedback":
        title = d.get("movie_title", d.get("movie_id", "?"))
        return (
            f"Movie feedback: {d.get('action', '?')} \"{title}\" "
            f"(phase={d.get('phase', '?')} iter={d.get('iteration', '?')})"
        )

    if t == "preferences-approved":
        liked = len(d.get("liked_movies", []) or [])
        tag = " [FINAL CONFIRMATION]" if d.get("is_final_confirmation") else ""
        return (
            f"Approved preferences: phase={d.get('phase', '?')} "
            f"iter={d.get('iteration', '?')}, {liked} liked{tag}"
        )

    if t == "phase-complete":
        return (
            f"Phase {d.get('phase', '?')} complete: "
            f"model={d.get('model', '?')}, "
            f"{d.get('iterations_used', '?')} iterations, "
            f"{d.get('total_liked', '?')} liked"
        )

    if t == "phase-questionnaire":
        keys = [k for k in d.keys() if k != "phase"]
        return f"Phase {d.get('phase', '?')} questionnaire submitted ({len(keys)} answers)"

    if t == "final-questionnaire":
        return f"Final questionnaire submitted ({len(d)} answers)"

    if t == "search-slider-adjusted":
        return (
            f"Search slider adjusted: \"{d.get('feature_label', '?')}\" "
            f"-> {d.get('new_value', '?')} (query: \"{d.get('search_query', '')}\")"
        )

    if t == "slider-adjusted":
        return f"Slider adjusted: \"{d.get('feature_label', '?')}\" -> {d.get('new_value', '?')}"

    if t == "slider-restored-from-history":
        return f"Slider restored from history: \"{d.get('feature_label', '?')}\""

    if t == "study-ended":
        return "Study ended"

    if t == "approach-order-assigned":
        order = d.get("effective_order", d.get("approach_order", "?"))
        return f"Approach order assigned: {order}"

    if t == "autosave":
        return f"Autosave ({d.get('trigger', '?')})"

    return t


def _parse_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def build_journey(
    interactions: Iterable[Dict[str, Any]],
    *,
    include_noise: bool = False,
) -> Dict[str, Any]:
    """Turn a list of interaction rows into a structured timeline + summary.

    Each row is expected to be a dict like:

        {"type": str, "time": iso-str, "data": dict, "id": int}

    Returns a dict with ``timeline`` (chronological list of compact entries)
    and ``summary`` (per-phase aggregates + total counters).
    """
    rows: List[Dict[str, Any]] = sorted(
        list(interactions),
        key=lambda r: r.get("time") or "",
    )

    timeline: List[Dict[str, Any]] = []
    type_counts: Counter = Counter()
    phase_events: Dict[int, List[Dict[str, Any]]] = {}

    for ix in rows:
        t = ix.get("type", "?")
        type_counts[t] += 1
        if not include_noise and is_timeline_noise(ix):
            continue
        section = PHASE_LABELS.get(t, "OTHER")
        timeline.append(
            {
                "section": section,
                "type": t,
                "ts": ix.get("time"),
                "ts_short": fmt_time(ix.get("time")),
                "summary": describe_interaction(ix),
            }
        )
        d = ix.get("data") or {}
        if isinstance(d, dict):
            phase = d.get("phase")
            if phase is not None:
                try:
                    phase_events.setdefault(int(phase), []).append(ix)
                except (TypeError, ValueError):
                    pass

    phases = []
    for phase_idx in sorted(phase_events.keys()):
        evts = phase_events[phase_idx]
        model_ids = set()
        iterations_seen = set()
        likes = 0
        dislikes = 0
        slider_adjustments = 0
        searches: List[str] = []
        for e in evts:
            d = e.get("data") or {}
            if not isinstance(d, dict):
                continue
            if d.get("model_id"):
                model_ids.add(str(d["model_id"]))
            if d.get("iteration") is not None:
                try:
                    iterations_seen.add(int(d["iteration"]))
                except (TypeError, ValueError):
                    pass
            if e.get("type") == "movie-feedback":
                if d.get("action") == "like":
                    likes += 1
                elif d.get("action") == "dislike":
                    dislikes += 1
            if e.get("type") == "feature-adjustment":
                slider_adjustments += len(d.get("adjustments", {}) or {})
                for s in (d.get("search_context", {}) or {}).get("search_history", []) or []:
                    if isinstance(s, dict):
                        searches.append(s.get("query", ""))
        phases.append(
            {
                "phase": phase_idx,
                "models": sorted(model_ids),
                "iterations": sorted(iterations_seen),
                "likes": likes,
                "dislikes": dislikes,
                "slider_adjustments": slider_adjustments,
                "searches": searches,
            }
        )

    first_ts = _parse_dt(rows[0].get("time")) if rows else None
    last_ts = _parse_dt(rows[-1].get("time")) if rows else None
    duration_sec = (
        int((last_ts - first_ts).total_seconds()) if first_ts and last_ts else None
    )

    total_clicks = sum(
        1
        for ix in rows
        if ix.get("type") in {"selected-item", "deselected-item", "movie-feedback", "feature-adjustment", "slider-adjusted", "search-slider-adjusted", "preferences-approved"}
    )

    return {
        "timeline": timeline,
        "summary": {
            "phases": phases,
            "duration_sec": duration_sec,
            "total_interactions": len(rows),
            "total_clicks": total_clicks,
            "type_counts": dict(type_counts.most_common()),
            "noise_hidden": (
                sum(type_counts[t] for t in NOISE_TYPES) if not include_noise else 0
            ),
        },
    }
