"""Results dashboard, export, and cleanup routes."""

import csv
import datetime
import io
import os
import shutil
import zipfile

from flask import Response, jsonify, render_template, request, url_for
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
from ...constants import PLUGIN_NAME
from ...paths import get_cache_path
from ...results.analytics import build_prolific_block, build_results_payload, safe_parse_json


@bp.route("/results")
@login_required
def results():
    guid = request.args.get("guid")
    return render_template(
        "sae_steering_results.html",
        guid=guid,
        fetch_results_url=url_for(f"{PLUGIN_NAME}.fetch_results", guid=guid),
        journey_url_base=url_for(f"{PLUGIN_NAME}.participant_journey", participation_id=0).rstrip("0"),
        export_raw_url=url_for(f"{PLUGIN_NAME}.export_raw_data", guid=guid),
        export_csv_url=url_for(f"{PLUGIN_NAME}.export_csv_data", guid=guid),
    )


@bp.route("/fetch-results/<guid>")
@login_required
def fetch_results(guid):
    payload, status = build_results_payload(guid)
    return jsonify(payload), status


@bp.route("/export-raw/<guid>")
@login_required
def export_raw_data(guid):
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return jsonify({"error": "Study not found"}), 404
    study_config = safe_parse_json(user_study.settings)
    all_participations = Participation.query.filter(
        Participation.user_study_id == user_study.id
    ).order_by(Participation.time_joined.asc()).all()

    participants_data = []
    for participant in all_participations:
        study_run = SaeStudyRun.query.filter(SaeStudyRun.participation_id == participant.id).first()
        approach_runs = []
        steering_events = []
        recommendation_sets = []
        movie_feedback = []
        questionnaire_responses = []
        feature_adjustments = []
        feature_searches = []
        text_steering_queries = []
        example_steerings = []
        reset_actions = []
        elicitation_picks = []
        if study_run:
            for approach in SaeApproachRun.query.filter(SaeApproachRun.study_run_id == study_run.id).order_by(SaeApproachRun.approach_index.asc()).all():
                approach_runs.append({
                    "id": approach.id,
                    "approach_index": approach.approach_index,
                    "approach_id": approach.approach_id,
                    "approach_name": approach.approach_name,
                    "steering_mode": approach.steering_mode,
                    "enabled_modalities": approach.enabled_modalities,
                    "sae_model_id": approach.sae_model_id,
                    "base_model_id": approach.base_model_id,
                    "started_at": approach.started_at.isoformat() if approach.started_at else None,
                    "completed_at": approach.completed_at.isoformat() if approach.completed_at else None,
                    "status": approach.status,
                    "summary": approach.summary,
                })
            for event in SaeSteeringEvent.query.filter(SaeSteeringEvent.study_run_id == study_run.id).order_by(SaeSteeringEvent.created_at.asc()).all():
                steering_events.append({
                    "id": event.id,
                    "event_type": event.event_type,
                    "approach_index": event.approach_index,
                    "approach_name": event.approach_name,
                    "iteration": event.iteration,
                    "modality": event.modality,
                    "steering_mode": event.steering_mode,
                    "source": event.source,
                    "search_query": event.search_query,
                    "raw_payload": event.raw_payload,
                    "created_at": event.created_at.isoformat() if event.created_at else None,
                })
            for rec_set in SaeRecommendationSet.query.filter(SaeRecommendationSet.study_run_id == study_run.id).order_by(SaeRecommendationSet.generated_at.asc()).all():
                items = SaeRecommendationItem.query.filter(SaeRecommendationItem.recommendation_set_id == rec_set.id).order_by(SaeRecommendationItem.rank.asc()).all()
                recommendation_sets.append({
                    "id": rec_set.id,
                    "approach_index": rec_set.approach_index,
                    "approach_name": rec_set.approach_name,
                    "iteration": rec_set.iteration,
                    "list_id": rec_set.list_id,
                    "steering_mode": rec_set.steering_mode,
                    "generated_at": rec_set.generated_at.isoformat() if rec_set.generated_at else None,
                    "debug_payload": rec_set.debug_payload,
                    "items": [
                        {
                            "rank": item.rank,
                            "movie_id": item.movie_id,
                            "title": item.title,
                            "genres": item.genres,
                            "score": item.score,
                            "cf_score": item.cf_score,
                            "genre_score": item.genre_score,
                            "steering_score": item.steering_score,
                        }
                        for item in items
                    ],
                })
            for feedback in SaeMovieFeedback.query.filter(SaeMovieFeedback.study_run_id == study_run.id).order_by(SaeMovieFeedback.created_at.asc()).all():
                movie_feedback.append({
                    "id": feedback.id,
                    "event_id": feedback.event_id,
                    "recommendation_set_id": feedback.recommendation_set_id,
                    "approach_index": feedback.approach_index,
                    "approach_name": feedback.approach_name,
                    "iteration": feedback.iteration,
                    "movie_id": feedback.movie_id,
                    "title": feedback.title,
                    "genres": feedback.genres,
                    "rank": feedback.rank,
                    "list_id": feedback.list_id,
                    "action": feedback.action,
                    "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
                })
            for response in SaeQuestionnaireResponse.query.filter(SaeQuestionnaireResponse.study_run_id == study_run.id).order_by(SaeQuestionnaireResponse.submitted_at.asc()).all():
                questionnaire_responses.append({
                    "id": response.id,
                    "response_type": response.response_type,
                    "approach_index": response.approach_index,
                    "approach_name": response.approach_name,
                    "questionnaire_file": response.questionnaire_file,
                    "answers": response.answers,
                    "submitted_at": response.submitted_at.isoformat() if response.submitted_at else None,
                })
            for adj in SaeFeatureAdjustment.query.filter(SaeFeatureAdjustment.study_run_id == study_run.id).order_by(SaeFeatureAdjustment.created_at.asc()).all():
                feature_adjustments.append({
                    "id": adj.id,
                    "event_id": adj.event_id,
                    "approach_run_id": adj.approach_run_id,
                    "iteration": adj.iteration,
                    "feature_id": adj.feature_id,
                    "cluster_label": adj.cluster_label,
                    "before_value": adj.before_value,
                    "after_value": adj.after_value,
                    "delta": adj.delta,
                    "applied_via": adj.applied_via,
                    "search_query": adj.search_query,
                    "created_at": adj.created_at.isoformat() if adj.created_at else None,
                })
            for search in SaeFeatureSearch.query.filter(SaeFeatureSearch.study_run_id == study_run.id).order_by(SaeFeatureSearch.created_at.asc()).all():
                hits = SaeFeatureSearchHit.query.filter(SaeFeatureSearchHit.search_id == search.id).order_by(SaeFeatureSearchHit.rank.asc()).all()
                feature_searches.append({
                    "id": search.id,
                    "event_id": search.event_id,
                    "approach_run_id": search.approach_run_id,
                    "iteration": search.iteration,
                    "query": search.query_text,
                    "result_count": search.result_count,
                    "created_at": search.created_at.isoformat() if search.created_at else None,
                    "hits": [
                        {
                            "rank": hit.rank,
                            "feature_id": hit.feature_id,
                            "label": hit.label,
                            "match_score": hit.match_score,
                        }
                        for hit in hits
                    ],
                })
            for query in SaeTextSteeringQuery.query.filter(SaeTextSteeringQuery.study_run_id == study_run.id).order_by(SaeTextSteeringQuery.created_at.asc()).all():
                matches = SaeTextSteeringMatch.query.filter(SaeTextSteeringMatch.query_id == query.id).all()
                text_steering_queries.append({
                    "id": query.id,
                    "event_id": query.event_id,
                    "approach_run_id": query.approach_run_id,
                    "iteration": query.iteration,
                    "query": query.query_text,
                    "length_chars": query.length_chars,
                    "composition_mode": query.composition_mode,
                    "created_at": query.created_at.isoformat() if query.created_at else None,
                    "matches": [
                        {
                            "cluster_id": match.cluster_id,
                            "label": match.label,
                            "weight": match.weight,
                            "match_score": match.match_score,
                            "direction": match.direction,
                        }
                        for match in matches
                    ],
                })
            for example in SaeExampleSteering.query.filter(SaeExampleSteering.study_run_id == study_run.id).order_by(SaeExampleSteering.created_at.asc()).all():
                example_movies = SaeExampleSteeringMovie.query.filter(SaeExampleSteeringMovie.example_id == example.id).order_by(SaeExampleSteeringMovie.rank.asc()).all()
                example_steerings.append({
                    "id": example.id,
                    "event_id": example.event_id,
                    "approach_run_id": example.approach_run_id,
                    "iteration": example.iteration,
                    "example_strength": example.example_strength,
                    "example_top_k": example.example_top_k,
                    "created_at": example.created_at.isoformat() if example.created_at else None,
                    "movies": [
                        {
                            "rank": movie.rank,
                            "movie_id": movie.movie_id,
                            "title": movie.title,
                        }
                        for movie in example_movies
                    ],
                })
            for reset in SaeResetAction.query.filter(SaeResetAction.study_run_id == study_run.id).order_by(SaeResetAction.created_at.asc()).all():
                reset_actions.append({
                    "id": reset.id,
                    "event_id": reset.event_id,
                    "approach_run_id": reset.approach_run_id,
                    "iteration": reset.iteration,
                    "trigger": reset.trigger,
                    "scope": reset.scope,
                    "created_at": reset.created_at.isoformat() if reset.created_at else None,
                })
        for pick in SaeElicitationPick.query.filter(SaeElicitationPick.participation_id == participant.id).order_by(SaeElicitationPick.created_at.asc()).all():
            elicitation_picks.append({
                "id": pick.id,
                "movie_id": pick.movie_id,
                "action": pick.action,
                "created_at": pick.created_at.isoformat() if pick.created_at else None,
            })
        participants_data.append(
            {
                "participation_id": participant.id,
                "uuid": participant.uuid,
                "email": participant.participant_email,
                "language": participant.language,
                "time_joined": participant.time_joined.isoformat() if participant.time_joined else None,
                "time_finished": participant.time_finished.isoformat() if participant.time_finished else None,
                "prolific": build_prolific_block(participant.extra_data, study_config),
                "extra_data": safe_parse_json(participant.extra_data),
                "sae_study_run": {
                    "id": study_run.id,
                    "schema_version": study_run.schema_version,
                    "study_guid": study_run.study_guid,
                    "approach_order": study_run.approach_order,
                    "effective_order": study_run.effective_order,
                    "started_at": study_run.started_at.isoformat() if study_run.started_at else None,
                    "finished_at": study_run.finished_at.isoformat() if study_run.finished_at else None,
                    "status": study_run.status,
                    "config_snapshot": study_run.config_snapshot,
                } if study_run else None,
                "approach_runs": approach_runs,
                "steering_events": steering_events,
                "recommendation_sets": recommendation_sets,
                "movie_feedback": movie_feedback,
                "questionnaire_responses": questionnaire_responses,
                "feature_adjustments": feature_adjustments,
                "feature_searches": feature_searches,
                "text_steering_queries": text_steering_queries,
                "example_steerings": example_steerings,
                "reset_actions": reset_actions,
                "elicitation_picks": elicitation_picks,
            }
        )

    export = {
        "study_guid": guid,
        "study_id": user_study.id,
        "study_config": study_config,
        "exported_at": datetime.datetime.utcnow().isoformat(),
        "schema": "sae-typed-audit.v1",
        "participants_total": len(participants_data),
        "participants_completed": sum(1 for participant in participants_data if participant["time_finished"]),
        "participants": participants_data,
    }
    response = jsonify(export)
    response.headers["Content-Disposition"] = f"attachment; filename=study_{guid}_export.json"
    return response


def _participation_ids_for_study(user_study_id: int) -> list[int]:
    return [
        pid
        for (pid,) in Participation.query.with_entities(Participation.id)
        .filter(Participation.user_study_id == user_study_id)
        .all()
    ]


def _csv_response(study_run_ids: list[int], participation_ids: list[int]) -> dict:
    """Build a dict of {csv_filename: csv_bytes} for every typed table."""
    def write_rows(headers, rows):
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        writer.writerows(rows)
        return buf.getvalue().encode("utf-8")

    files: dict[str, bytes] = {}

    files["sae_study_run.csv"] = write_rows(
        [
            "id", "participation_id", "user_study_id", "study_guid",
            "approach_order", "effective_order", "started_at", "finished_at", "status",
        ],
        [
            [
                r.id, r.participation_id, r.user_study_id, r.study_guid,
                r.approach_order, r.effective_order,
                r.started_at.isoformat() if r.started_at else None,
                r.finished_at.isoformat() if r.finished_at else None,
                r.status,
            ]
            for r in SaeStudyRun.query.filter(SaeStudyRun.id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_approach_run.csv"] = write_rows(
        [
            "id", "study_run_id", "participation_id", "approach_index", "approach_id",
            "approach_name", "steering_mode", "enabled_modalities", "sae_model_id",
            "base_model_id", "composition_mode", "reranking_strategy",
            "started_at", "completed_at", "status",
            "iterations_used", "final_liked_count", "total_slider_changes",
        ],
        [
            [
                r.id, r.study_run_id, r.participation_id, r.approach_index, r.approach_id,
                r.approach_name, r.steering_mode, r.enabled_modalities, r.sae_model_id,
                r.base_model_id, r.composition_mode, r.reranking_strategy,
                r.started_at.isoformat() if r.started_at else None,
                r.completed_at.isoformat() if r.completed_at else None,
                r.status, r.iterations_used, r.final_liked_count, r.total_slider_changes,
            ]
            for r in SaeApproachRun.query.filter(SaeApproachRun.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_steering_event.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id", "event_type",
            "approach_index", "approach_name", "iteration", "modality", "steering_mode",
            "source", "search_query", "created_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.event_type,
                r.approach_index, r.approach_name, r.iteration, r.modality, r.steering_mode,
                r.source, r.search_query,
                r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeSteeringEvent.query.filter(SaeSteeringEvent.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_feature_adjustment.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id", "event_id",
            "iteration", "feature_id", "cluster_label", "before_value", "after_value",
            "delta", "applied_via", "search_query", "created_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.event_id,
                r.iteration, r.feature_id, r.cluster_label, r.before_value, r.after_value,
                r.delta, r.applied_via, r.search_query,
                r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeFeatureAdjustment.query.filter(SaeFeatureAdjustment.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_feature_search.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id", "event_id",
            "iteration", "query_text", "result_count", "created_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.event_id,
                r.iteration, r.query_text, r.result_count,
                r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeFeatureSearch.query.filter(SaeFeatureSearch.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_feature_search_hit.csv"] = write_rows(
        ["id", "search_id", "feature_id", "label", "match_score", "rank"],
        [
            [r.id, r.search_id, r.feature_id, r.label, r.match_score, r.rank]
            for r in (
                SaeFeatureSearchHit.query.join(
                    SaeFeatureSearch, SaeFeatureSearch.id == SaeFeatureSearchHit.search_id
                ).filter(SaeFeatureSearch.study_run_id.in_(study_run_ids)).all()
                if study_run_ids
                else []
            )
        ],
    )
    files["sae_text_steering_query.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id", "event_id",
            "iteration", "query_text", "length_chars", "composition_mode", "created_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.event_id,
                r.iteration, r.query_text, r.length_chars, r.composition_mode,
                r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeTextSteeringQuery.query.filter(SaeTextSteeringQuery.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_text_steering_match.csv"] = write_rows(
        ["id", "query_id", "cluster_id", "label", "weight", "match_score", "direction"],
        [
            [r.id, r.query_id, r.cluster_id, r.label, r.weight, r.match_score, r.direction]
            for r in (
                SaeTextSteeringMatch.query.join(
                    SaeTextSteeringQuery, SaeTextSteeringQuery.id == SaeTextSteeringMatch.query_id
                ).filter(SaeTextSteeringQuery.study_run_id.in_(study_run_ids)).all()
                if study_run_ids
                else []
            )
        ],
    )
    files["sae_example_steering.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id", "event_id",
            "iteration", "example_strength", "example_top_k", "created_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.event_id,
                r.iteration, r.example_strength, r.example_top_k,
                r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeExampleSteering.query.filter(SaeExampleSteering.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_example_steering_movie.csv"] = write_rows(
        ["id", "example_id", "movie_id", "title", "rank"],
        [
            [r.id, r.example_id, r.movie_id, r.title, r.rank]
            for r in (
                SaeExampleSteeringMovie.query.join(
                    SaeExampleSteering, SaeExampleSteering.id == SaeExampleSteeringMovie.example_id
                ).filter(SaeExampleSteering.study_run_id.in_(study_run_ids)).all()
                if study_run_ids
                else []
            )
        ],
    )
    files["sae_reset_action.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id", "event_id",
            "iteration", "trigger", "scope", "created_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.event_id,
                r.iteration, r.trigger, r.scope,
                r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeResetAction.query.filter(SaeResetAction.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_recommendation_set.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id", "approach_index",
            "approach_name", "iteration", "list_id", "steering_mode", "generated_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.approach_index,
                r.approach_name, r.iteration, r.list_id, r.steering_mode,
                r.generated_at.isoformat() if r.generated_at else None,
            ]
            for r in SaeRecommendationSet.query.filter(SaeRecommendationSet.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_recommendation_item.csv"] = write_rows(
        [
            "id", "recommendation_set_id", "movie_id", "title", "genres", "rank",
            "list_id", "score", "cf_score", "genre_score", "steering_score",
        ],
        [
            [
                r.id, r.recommendation_set_id, r.movie_id, r.title, r.genres, r.rank,
                r.list_id, r.score, r.cf_score, r.genre_score, r.steering_score,
            ]
            for r in (
                SaeRecommendationItem.query.join(
                    SaeRecommendationSet,
                    SaeRecommendationSet.id == SaeRecommendationItem.recommendation_set_id,
                ).filter(SaeRecommendationSet.study_run_id.in_(study_run_ids)).all()
                if study_run_ids
                else []
            )
        ],
    )
    files["sae_movie_feedback.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "recommendation_set_id",
            "participation_id", "event_id", "approach_index", "approach_name",
            "iteration", "movie_id", "title", "genres", "rank", "list_id",
            "action", "created_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.recommendation_set_id,
                r.participation_id, r.event_id, r.approach_index, r.approach_name,
                r.iteration, r.movie_id, r.title, r.genres, r.rank, r.list_id,
                r.action, r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeMovieFeedback.query.filter(SaeMovieFeedback.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_questionnaire_response.csv"] = write_rows(
        [
            "id", "study_run_id", "approach_run_id", "participation_id",
            "response_type", "approach_index", "approach_name", "questionnaire_file",
            "answers_json", "submitted_at",
        ],
        [
            [
                r.id, r.study_run_id, r.approach_run_id, r.participation_id,
                r.response_type, r.approach_index, r.approach_name, r.questionnaire_file,
                r.answers, r.submitted_at.isoformat() if r.submitted_at else None,
            ]
            for r in SaeQuestionnaireResponse.query.filter(SaeQuestionnaireResponse.study_run_id.in_(study_run_ids)).all()
        ] if study_run_ids else [],
    )
    files["sae_elicitation_pick.csv"] = write_rows(
        ["id", "participation_id", "study_run_id", "user_study_id", "movie_id", "action", "created_at"],
        [
            [
                r.id, r.participation_id, r.study_run_id, r.user_study_id,
                r.movie_id, r.action,
                r.created_at.isoformat() if r.created_at else None,
            ]
            for r in SaeElicitationPick.query.filter(
                SaeElicitationPick.participation_id.in_(participation_ids)
            ).all()
        ] if participation_ids else [],
    )
    return files


@bp.route("/export-csv/<guid>")
@login_required
def export_csv_data(guid):
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return jsonify({"error": "Study not found"}), 404
    participation_ids = _participation_ids_for_study(user_study.id)
    study_run_ids = [
        sid
        for (sid,) in SaeStudyRun.query.with_entities(SaeStudyRun.id)
        .filter(SaeStudyRun.user_study_id == user_study.id)
        .all()
    ]
    files = _csv_response(study_run_ids, participation_ids)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    response = Response(zip_buffer.getvalue(), mimetype="application/zip")
    response.headers["Content-Disposition"] = (
        f"attachment; filename=study_{guid}_csv_export.zip"
    )
    return response


@bp.route("/dispose", methods=["DELETE"])
@login_required
def dispose():
    guid = request.args.get("guid")
    cache_path = get_cache_path(guid)
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    return "OK"
