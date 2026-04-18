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


def _extract_item_title(obj: Any) -> Optional[str]:
    """Pick a display label from the various item-shaped payloads the client sends.

    The elicitation page ships ``{movie: {idx, url, ...}, movieName: "…"}`` while
    other flows ship a flatter ``{title: …}`` — we tolerate both plus a few
    near-synonyms so the journey timeline never falls back to ``"?"`` when
    metadata is actually present.
    """
    if not isinstance(obj, dict):
        return None
    for key in ("movieName", "movie_name", "title", "name", "label"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v
    movie = obj.get("movie")
    if isinstance(movie, dict):
        for key in ("title", "name", "movieName", "idx", "id"):
            v = movie.get(key)
            if v not in (None, ""):
                return str(v)
    return None


def _fmt_value(v: Any, digits: int = 2) -> str:
    """Format a numeric slider value compactly, falling back to str()."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return str(v)
    if v in (None, ""):
        return "?"
    return str(v)


def describe_interaction(ix: Dict[str, Any]) -> str:
    """Return a human-readable one-liner for an interaction row."""
    t = ix.get("type", "?")
    d = ix.get("data") or {}
    if not isinstance(d, dict):
        d = {}

    if t == "loaded-page":
        return f"Opened page: {d.get('page', '?')}"

    if t in ("selected-item", "deselected-item"):
        # Client sends either {selected_item: {...}} or {item: {...}}; handle both.
        src = (
            d.get("selected_item")
            or d.get("deselected_item")
            or d.get("item")
            or d
        )
        title = _extract_item_title(src) or "?"
        verb = "select" if t == "selected-item" else "deselect"
        return f"Elicitation: {verb} \"{title}\""

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
        model = d.get("approach_name") or d.get("model") or d.get("model_id") or "?"
        return (
            f"Feature adjustment: phase={d.get('phase', '?')} "
            f"iter={d.get('iteration', '?')} "
            f"model={model}, "
            f"{adj_count} sliders changed{searches}"
        )

    if t == "recommendations-shown":
        # Server uses `movies`/`model` (current) or `movie_ids`/`model_id` (legacy);
        # also handle the A/B comparison variant (`model_a`/`model_b`).
        movies = d.get("movies")
        if movies is None:
            movies = d.get("movie_ids")
        if movies is None and (d.get("model_a") is not None or d.get("model_b") is not None):
            a = d.get("model_a") or []
            b = d.get("model_b") or []
            return (
                f"Recommendations shown (A/B): iter={d.get('iteration', '?')}, "
                f"A={len(a)} movies, B={len(b)} movies"
            )
        cnt = len(movies or [])
        model = d.get("approach_name") or d.get("model") or d.get("model_id") or "?"
        return (
            f"Recommendations shown: phase={d.get('phase', '?')} "
            f"iter={d.get('iteration', '?')}, {cnt} movies, "
            f"model={model}"
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
        model = d.get("approach_name") or d.get("model") or d.get("model_id") or "?"
        parts = [f"Phase {d.get('phase', '?')} complete", f"model={model}"]
        if d.get("iterations_used") is not None:
            parts.append(f"{d.get('iterations_used')} iterations")
        if d.get("total_liked") is not None:
            parts.append(f"{d.get('total_liked')} liked")
        if d.get("total_slider_changes") is not None:
            parts.append(f"{d.get('total_slider_changes')} slider changes")
        return ": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]

    if t == "phase-questionnaire":
        keys = [k for k in d.keys() if k != "phase"]
        return f"Phase {d.get('phase', '?')} questionnaire submitted ({len(keys)} answers)"

    if t == "final-questionnaire":
        return f"Final questionnaire submitted ({len(d)} answers)"

    if t == "search-slider-adjusted":
        # Client ships `label`/`value`/`found_via_query`; older rows used
        # `feature_label`/`new_value`/`search_query`.
        label = d.get("label") or d.get("feature_label") or d.get("feature_id") or "?"
        value = _fmt_value(d.get("value", d.get("new_value")))
        query = d.get("found_via_query") or d.get("search_query") or ""
        return f"Search slider adjusted: \"{label}\" -> {value} (query: \"{query}\")"

    if t == "slider-adjusted":
        label = d.get("label") or d.get("feature_label") or d.get("feature_id") or "?"
        value = _fmt_value(d.get("value", d.get("new_value")))
        return f"Slider adjusted: \"{label}\" -> {value}"

    if t == "slider-restored-from-history":
        label = d.get("label") or d.get("feature_label") or d.get("feature_id") or "?"
        return f"Slider restored from history: \"{label}\""

    if t == "study-ended":
        return "Study ended"

    if t == "approach-order-assigned":
        # Prefer the human-readable names we now store alongside the numeric indices.
        names = d.get("effective_order")
        if isinstance(names, list) and names:
            return "Approach order assigned: " + " → ".join(f"Phase {i + 1}: {n}" for i, n in enumerate(names))
        order = d.get("approach_order")
        if isinstance(order, list):
            return f"Approach order assigned (indices): {order}"
        return "Approach order assigned: ?"

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
        sae_model_ids: set = set()
        approach_names: set = set()
        iterations_seen = set()
        likes = 0
        dislikes = 0
        slider_adjustments = 0
        searches: List[str] = []
        for e in evts:
            d = e.get("data") or {}
            if not isinstance(d, dict):
                continue
            # The SAE checkpoint (e.g. "TopKSAE-1024") is the same across arms in
            # most studies, so we also track the human-readable approach name
            # coming from recommendations-shown / feature-adjustment / phase-complete.
            if d.get("model_id"):
                sae_model_ids.add(str(d["model_id"]))
            for key in ("approach_name", "model"):
                v = d.get(key)
                if isinstance(v, str) and v.strip():
                    approach_names.add(v.strip())
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
        # `models` stays for backward compatibility (used by older UI code); it now
        # prefers the approach name over the raw SAE checkpoint so the journey
        # summary reads "Approach without Explicit Feedback" instead of
        # "TopKSAE-1024" for both phases.
        models_display = sorted(approach_names) if approach_names else sorted(sae_model_ids)
        phases.append(
            {
                "phase": phase_idx,
                "models": models_display,
                "approach_names": sorted(approach_names),
                "sae_model_ids": sorted(sae_model_ids),
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
