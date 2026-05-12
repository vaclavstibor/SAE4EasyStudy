"""Participant journey route backed by typed SAE audit tables."""

from flask import jsonify
from flask_login import login_required

from server.platform.persistence.base_models import Participation, UserStudy
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

from ...plugin import bp
from ...results.analytics import build_prolific_block, safe_parse_json


def _fmt_time(ts):
    return ts.strftime("%H:%M:%S") if ts else "-"


def _event_summary(event, ctx):
    raw = event.raw_payload or {}
    approach_label = (
        f"Approach {(event.approach_index or 0) + 1} ({event.approach_name})"
        if event.approach_index is not None
        else "Elicitation"
    )

    if event.event_type == "elicitation-search":
        return (
            f"Elicitation search: \"{raw.get('query') or event.search_query}\" "
            f"-> {raw.get('result_count')} results"
        )
    if event.event_type == "elicitation-completed":
        movies = raw.get("selected_movies") or []
        titles = ", ".join(row.get("title") or str(row.get("movie_id")) for row in movies)
        return f"Elicitation completed: {len(movies)} selected movies ({titles})"
    if event.event_type == "text-steering-parsed":
        query_row = ctx["text_queries_by_event"].get(event.id)
        if query_row:
            mapped = ", ".join(
                f"{m.label} ({m.cluster_id}, weight={m.weight:+.2f})"
                for m in ctx["text_matches_by_query"].get(query_row.id, [])
            )
            return f"Text steering: \"{query_row.query_text}\" -> {mapped}"
        return f"Text steering: \"{event.search_query}\""
    if event.event_type == "feature-search":
        search_row = ctx["feature_search_by_event"].get(event.id)
        result_count = search_row.result_count if search_row else raw.get("result_count")
        query = (search_row.query_text if search_row else None) or raw.get("query") or event.search_query
        return f"Feature search: \"{query}\" -> {result_count} results"
    if event.event_type == "feature-adjustment":
        adjustments = ctx["feature_adjustments_by_event"].get(event.id, [])
        changed = [
            f"{row.cluster_label or row.feature_id}: {row.before_value:+.2f} -> {row.after_value:+.2f}"
            for row in adjustments[:5]
        ]
        search_queries = sorted(
            {row.search_query for row in adjustments if row.search_query}
        )
        search_note = f", search: {', '.join(search_queries)}" if search_queries else ""
        change_note = f", controls [{'; '.join(changed)}]" if changed else ""
        return (
            f"Feature adjustment: {approach_label}, iteration {event.iteration}, "
            f"{len(adjustments)} feature(s) adjusted{search_note}{change_note}"
        )
    if event.event_type == "recommendations-shown":
        rec_set_id = raw.get("recommendation_set_id")
        rec_set = ctx["recommendation_sets"].get(rec_set_id)
        count = len(rec_set["items"]) if rec_set else raw.get("movie_count")
        top_titles = ", ".join(item["title"] for item in (rec_set or {}).get("items", [])[:5])
        return (
            f"Recommendations shown: {approach_label}, iteration {event.iteration}, "
            f"{count} movies stored with ranks"
            + (f" (top: {top_titles})" if top_titles else "")
        )
    if event.event_type == "movie-feedback":
        feedback = ctx["movie_feedback_by_event"].get(event.id)
        if feedback is None:
            return f"Movie feedback: {approach_label}, iteration {event.iteration}"
        return (
            f"Movie feedback: {feedback.action} \"{feedback.title}\" at rank {feedback.rank} "
            f"in {approach_label}, iteration {event.iteration}"
        )
    if event.event_type == "preferences-approved":
        liked = raw.get("liked_movies", [])
        final = " final confirmation" if raw.get("is_final_confirmation") else ""
        return (
            f"Preferences approved:{final} {approach_label}, iteration {event.iteration}, "
            f"{len(liked)} liked"
        )
    if event.event_type == "approach-complete":
        return (
            f"Approach {(event.approach_index or 0) + 1} complete: {event.approach_name}, "
            f"{raw.get('iterations_used')} iterations, {raw.get('final_liked_count')} liked, "
            f"{raw.get('total_slider_changes')} feature changes"
        )
    if event.event_type == "approach-questionnaire":
        return f"Approach questionnaire submitted: {raw.get('answer_count')} answers"
    if event.event_type == "final-questionnaire":
        return f"Final questionnaire submitted: {raw.get('answer_count')} answers"
    if event.event_type == "autosave":
        return f"Autosave snapshot: {raw.get('trigger')}"
    if event.event_type == "global-reset":
        reset_row = ctx["resets_by_event"].get(event.id)
        scope = reset_row.scope if reset_row is not None else raw.get("scope")
        trigger = (reset_row.trigger if reset_row is not None else raw.get("trigger")) or ""
        trigger_note = f" via {trigger}" if trigger else ""
        return f"Global reset (scope: {scope}{trigger_note})"
    if event.event_type == "example-steering-applied":
        example_row = ctx["examples_by_event"].get(event.id)
        movie_count = ctx["example_movie_counts"].get(example_row.id, 0) if example_row else 0
        strength = example_row.example_strength if example_row else raw.get("example_strength")
        top_k = example_row.example_top_k if example_row else raw.get("example_top_k")
        return (
            f"Example steering applied: {movie_count} movies, "
            f"strength={strength}, top_k={top_k}"
        )
    if event.event_type == "approach-order-assigned":
        return f"Approach order assigned: {raw.get('effective_order') or raw.get('approach_order')}"
    return event.event_type


def _section(event_type):
    if event_type.startswith("elicitation"):
        return "ELICITATION"
    if event_type in {"approach-complete"}:
        return "APPROACH"
    if "questionnaire" in event_type:
        return "QUESTIONNAIRE"
    if event_type == "autosave":
        return "SYSTEM"
    if event_type == "study-ended":
        return "FINISH"
    return "STEERING"


@bp.route("/journey/<int:participation_id>")
@login_required
def participant_journey(participation_id):
    participation = Participation.query.filter(Participation.id == participation_id).first()
    if not participation:
        return jsonify({"error": "Participation not found"}), 404

    user_study = UserStudy.query.filter(UserStudy.id == participation.user_study_id).first()
    study_config = safe_parse_json(user_study.settings) if user_study else {}
    study_run = SaeStudyRun.query.filter(SaeStudyRun.participation_id == participation.id).first()
    if not study_run:
        return jsonify({"error": "No typed SAE audit run found for this participant"}), 404

    approach_runs = (
        SaeApproachRun.query.filter(SaeApproachRun.study_run_id == study_run.id)
        .order_by(SaeApproachRun.approach_index.asc())
        .all()
    )

    rec_sets = (
        SaeRecommendationSet.query.filter(SaeRecommendationSet.study_run_id == study_run.id)
        .order_by(SaeRecommendationSet.generated_at.asc())
        .all()
    )
    recommendation_sets = {}
    for rec_set in rec_sets:
        items = (
            SaeRecommendationItem.query.filter(
                SaeRecommendationItem.recommendation_set_id == rec_set.id
            )
            .order_by(SaeRecommendationItem.rank.asc())
            .all()
        )
        recommendation_sets[rec_set.id] = {
            "id": rec_set.id,
            "items": [
                {
                    "movie_id": item.movie_id,
                    "title": item.title,
                    "genres": item.genres,
                    "rank": item.rank,
                }
                for item in items
            ],
        }

    events = (
        SaeSteeringEvent.query.filter(SaeSteeringEvent.study_run_id == study_run.id)
        .order_by(SaeSteeringEvent.created_at.asc(), SaeSteeringEvent.id.asc())
        .all()
    )

    feature_adjustments_by_event: dict[int, list[SaeFeatureAdjustment]] = {}
    for adj in SaeFeatureAdjustment.query.filter(
        SaeFeatureAdjustment.study_run_id == study_run.id
    ).order_by(SaeFeatureAdjustment.created_at.asc()).all():
        feature_adjustments_by_event.setdefault(adj.event_id, []).append(adj)

    text_queries = SaeTextSteeringQuery.query.filter(
        SaeTextSteeringQuery.study_run_id == study_run.id
    ).all()
    text_queries_by_event = {row.event_id: row for row in text_queries}
    text_matches_by_query: dict[int, list[SaeTextSteeringMatch]] = {}
    for match in SaeTextSteeringMatch.query.filter(
        SaeTextSteeringMatch.query_id.in_([row.id for row in text_queries])
    ).all() if text_queries else []:
        text_matches_by_query.setdefault(match.query_id, []).append(match)

    feature_searches = SaeFeatureSearch.query.filter(
        SaeFeatureSearch.study_run_id == study_run.id
    ).all()
    feature_search_by_event = {row.event_id: row for row in feature_searches}

    movie_feedback_by_event = {
        row.event_id: row
        for row in SaeMovieFeedback.query.filter(
            SaeMovieFeedback.study_run_id == study_run.id
        ).all()
    }

    elicitation_picks = (
        SaeElicitationPick.query.filter(
            SaeElicitationPick.participation_id == participation.id
        )
        .order_by(SaeElicitationPick.created_at.asc(), SaeElicitationPick.id.asc())
        .all()
    )

    resets_by_event = {
        row.event_id: row
        for row in SaeResetAction.query.filter(
            SaeResetAction.study_run_id == study_run.id
        ).all()
    }

    example_rows = SaeExampleSteering.query.filter(
        SaeExampleSteering.study_run_id == study_run.id
    ).all()
    examples_by_event = {row.event_id: row for row in example_rows}
    example_movie_counts: dict[int, int] = {}
    if example_rows:
        example_ids = [row.id for row in example_rows]
        for movie in SaeExampleSteeringMovie.query.filter(
            SaeExampleSteeringMovie.example_id.in_(example_ids)
        ).all():
            example_movie_counts[movie.example_id] = (
                example_movie_counts.get(movie.example_id, 0) + 1
            )

    ctx = {
        "recommendation_sets": recommendation_sets,
        "feature_adjustments_by_event": feature_adjustments_by_event,
        "text_queries_by_event": text_queries_by_event,
        "text_matches_by_query": text_matches_by_query,
        "feature_search_by_event": feature_search_by_event,
        "movie_feedback_by_event": movie_feedback_by_event,
        "elicitation_picks": elicitation_picks,
        "resets_by_event": resets_by_event,
        "examples_by_event": examples_by_event,
        "example_movie_counts": example_movie_counts,
    }

    timeline = [
        {
            "section": _section(event.event_type),
            "type": event.event_type,
            "ts": event.created_at.isoformat() if event.created_at else None,
            "ts_short": _fmt_time(event.created_at),
            "summary": _event_summary(event, ctx),
            "source": event.source,
            "search_query": event.search_query,
            "modality": event.modality,
            "raw_payload": event.raw_payload,
        }
        for event in events
    ]

    questionnaire_rows = SaeQuestionnaireResponse.query.filter(
        SaeQuestionnaireResponse.study_run_id == study_run.id
    ).all()
    questionnaire_responses = [
        {
            "response_type": row.response_type,
            "approach_index": row.approach_index,
            "approach_name": row.approach_name,
            "questionnaire_file": row.questionnaire_file,
            "submitted_at": row.submitted_at.isoformat() if row.submitted_at else None,
            "answers": row.answers or {},
        }
        for row in questionnaire_rows
    ]

    first_ts = events[0].created_at if events else study_run.started_at
    last_ts = (study_run.finished_at or events[-1].created_at) if events else study_run.finished_at
    duration_sec = int((last_ts - first_ts).total_seconds()) if first_ts and last_ts else None
    approaches = [
        {
            "approach_index": row.approach_index,
            "approach_name": row.approach_name,
            "steering_mode": row.steering_mode,
            "composition_mode": row.composition_mode,
            "reranking_strategy": row.reranking_strategy,
            "iterations": list(range(1, (row.iterations_used or 0) + 1)),
            "likes": row.final_liked_count or 0,
            "slider_adjustments": int(row.total_slider_changes or 0),
        }
        for row in approach_runs
    ]

    return jsonify(
        {
            "participation_id": participation.id,
            "study_guid": user_study.guid if user_study else None,
            "participant": {
                "uuid": participation.uuid,
                "email": participation.participant_email,
                "language": participation.language,
                "prolific": build_prolific_block(participation.extra_data, study_config),
                "time_joined": (
                    participation.time_joined.isoformat() if participation.time_joined else None
                ),
                "time_finished": (
                    participation.time_finished.isoformat() if participation.time_finished else None
                ),
                "approach_order": study_run.approach_order,
                "effective_order": study_run.effective_order,
            },
            "questionnaire_responses": questionnaire_responses,
            "timeline": timeline,
            "summary": {
                "approaches": approaches,
                "duration_sec": duration_sec,
                "total_interactions": len(events),
                "total_clicks": sum(
                    1
                    for event in events
                    if event.event_type
                    in {"movie-feedback", "feature-adjustment", "preferences-approved"}
                ),
                "type_counts": {
                    event_type: sum(1 for event in events if event.event_type == event_type)
                    for event_type in sorted({event.event_type for event in events})
                },
            },
        }
    )
