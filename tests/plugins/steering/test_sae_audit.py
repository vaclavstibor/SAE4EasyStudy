import json
from datetime import datetime


def _seed_participation(db):
    from server.platform.persistence.base_models import Participation, User, UserStudy

    user = User(email="admin@example.com", password="x", authenticated=True, admin=True)
    study = UserStudy(
        creator=user.email,
        guid="study-guid",
        parent_plugin="sae_steering",
        settings=json.dumps(
            {
                "dataset": "ml-32m-filtered",
                "models": [
                    {
                        "id": "approach_a",
                        "name": "Approach A",
                        "base": "elsa",
                        "sae": "TopKSAE-1024",
                        "steering_mode": "both",
                        "enabled_modalities": ["sliders", "text", "reset"],
                    },
                    {
                        "id": "approach_b",
                        "name": "Approach B",
                        "base": "elsa",
                        "sae": "TopKSAE-1024",
                        "steering_mode": "toggles",
                        "enabled_modalities": ["toggles", "reset"],
                    },
                ],
                "comparison_mode": "sequential",
                "randomize_approach_order": False,
            }
        ),
        time_created=datetime.utcnow(),
        active=True,
        initialized=True,
    )
    db.session.add(user)
    db.session.add(study)
    db.session.flush()
    participation = Participation(
        participant_email="participant@example.com",
        user_study_id=study.id,
        time_joined=datetime.utcnow(),
        uuid="participant-uuid",
        language="en",
    )
    db.session.add(participation)
    db.session.commit()
    return study, participation


def test_typed_sae_tables_exist(app_ctx):
    _, db = app_ctx
    from sqlalchemy import inspect

    tables = set(inspect(db.engine).get_table_names())
    assert {
        "sae_study_run",
        "sae_approach_run",
        "sae_steering_event",
        "sae_recommendation_set",
        "sae_recommendation_item",
        "sae_movie_feedback",
        "sae_questionnaire_response",
    }.issubset(tables)


def test_audit_service_creates_strict_study_and_approach_runs(app_ctx):
    app, db = app_ctx
    study, participation = _seed_participation(db)

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.persistence.models import SaeApproachRun, SaeStudyRun
        from server.plugins.steering.service import audit

        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["approach_order"] = [0, 1]
        session["current_phase"] = 0

        study_run = audit.ensure_study_run(participation.id)
        approach_run = audit.ensure_approach_run(participation.id, approach_index=0)

        assert SaeStudyRun.query.count() == 1
        assert SaeApproachRun.query.count() == 1
        assert study_run.effective_order == ["Approach A", "Approach B"]
        assert approach_run.approach_name == "Approach A"
        assert approach_run.steering_mode == "both"


def test_text_steering_audit_writes_typed_query_and_matches(app_ctx):
    app, db = app_ctx
    study, participation = _seed_participation(db)

    class Resolved:
        features = [
            {
                "id": "cluster_55",
                "label": "Marvel Cinematic Universe",
                "description": "Marvel superhero films.",
                "weight": 0.9,
                "match_score": 4.0,
                "direction": "boost",
                "member_ids": [66, 290],
            }
        ]
        adjustments = {"cluster_55": 0.9}
        metadata = {"segments": [{"text": "I like Marvel", "direction": 1, "tokens": ["marvel"]}]}

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.persistence.models import (
            SaeSteeringEvent,
            SaeTextSteeringMatch,
            SaeTextSteeringQuery,
        )
        from server.plugins.steering.service import audit

        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["approach_order"] = [0, 1]
        session["current_phase"] = 0

        audit.record_text_steering(
            "I like Marvel",
            Resolved(),
            participation_id=participation.id,
            approach_index=0,
            active_model={"steering_mode": "both"},
            iteration=1,
            composition_mode="replace",
        )

        event = SaeSteeringEvent.query.filter_by(event_type="text-steering-parsed").one()
        assert event.raw_payload["query"] == "I like Marvel"
        assert event.raw_payload["composition_mode"] == "replace"

        query_row = db.session.query(SaeTextSteeringQuery).one()
        assert query_row.query_text == "I like Marvel"
        assert query_row.length_chars == len("I like Marvel")
        assert query_row.composition_mode == "replace"
        assert query_row.event_id == event.id

        match = SaeTextSteeringMatch.query.filter_by(query_id=query_row.id).one()
        assert match.cluster_id == "cluster_55"
        assert match.weight == 0.9
        assert match.direction == "boost"


def test_enabled_modalities_are_authoritative_for_steering_mode(app_ctx):
    from server.plugins.steering.study_config import normalize_study_config

    conf = normalize_study_config(
        {
            "models": [
                {
                    "name": "No conflict",
                    "steering_mode": "none",
                    "enabled_modalities": ["toggles", "text", "reset"],
                }
            ]
        }
    )

    model = conf["models"][0]
    assert model["steering_mode"] == "toggles"
    assert model["enabled_modalities"] == ["toggles", "text", "reset"]
    assert model["selection_signal_weight"] == 0.25


def test_selection_signal_weight_defaults_and_override(app_ctx):
    from server.plugins.steering.study_config import normalize_study_config

    conf = normalize_study_config(
        {
            "models": [
                {"name": "Implicit", "enabled_modalities": []},
                {"name": "Slider", "enabled_modalities": ["sliders"]},
                {"name": "Custom", "enabled_modalities": ["text"], "selection_signal_weight": 0.7},
            ]
        }
    )

    assert conf["models"][0]["selection_signal_weight"] == 0.5
    assert conf["models"][1]["selection_signal_weight"] == 0.25
    assert conf["models"][2]["selection_signal_weight"] == 0.7


def test_search_source_is_typed_on_steering_event(app_ctx):
    app, db = app_ctx
    study, participation = _seed_participation(db)

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.persistence.models import SaeSteeringEvent
        from server.plugins.steering.service import audit

        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["approach_order"] = [0, 1]
        session["current_phase"] = 0

        audit.record_event(
            "feature-search",
            participation_id=participation.id,
            approach_index=0,
            iteration=1,
            modality="feature-search",
            source="search",
            search_query="marvel origin",
            raw_payload={"query": "marvel origin", "result_count": 3},
        )

        event = SaeSteeringEvent.query.filter_by(event_type="feature-search").one()
        assert event.source == "search"
        assert event.search_query == "marvel origin"
        assert event.raw_payload["query"] == "marvel origin"


def test_finish_user_study_redirects_to_explicit_final_questionnaire(
    app_ctx,
    monkeypatch,
    tmp_path,
):
    app, db = app_ctx
    study, participation = _seed_participation(db)
    study.settings = json.dumps(
        {
            "dataset": "ml-32m-filtered",
            "questionnaire_file": "final.html",
            "models": [
                {
                    "id": "approach_a",
                    "name": "Approach A",
                    "base": "elsa",
                    "sae": "TopKSAE-1024",
                    "steering_mode": "both",
                    "enabled_modalities": ["sliders", "text", "reset"],
                }
            ],
        }
    )
    db.session.commit()

    questionnaire_path = tmp_path / "final.html"
    questionnaire_path.write_text("<html>final</html>", encoding="utf-8")
    monkeypatch.setattr(
        "server.plugins.steering.paths.get_cache_path",
        lambda guid, name="": str(questionnaire_path if name else questionnaire_path.parent),
    )

    client = app.test_client()
    with client.session_transaction() as session:
        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["user_study_guid"] = study.guid

    response = client.get("/sae_steering/finish-user-study")

    assert response.status_code == 302
    location = response.headers["Location"]
    assert "/utils/final-questionnaire" in location
    assert "questionnaire_file=final.html" in location
    assert "continuation_url=/sae_steering/_complete-study" in location


def test_finish_user_study_materializes_bundled_questionnaire_when_cache_missing(
    app_ctx,
    monkeypatch,
    tmp_path,
):
    app, db = app_ctx
    study, participation = _seed_participation(db)
    study.settings = json.dumps(
        {
            "dataset": "ml-32m-filtered",
            "questionnaire_file": "final.html",
            "models": [
                {
                    "id": "approach_a",
                    "name": "Approach A",
                    "base": "elsa",
                    "sae": "TopKSAE-1024",
                    "steering_mode": "both",
                    "enabled_modalities": ["sliders", "text", "reset"],
                }
            ],
        }
    )
    db.session.commit()

    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    (bundled_dir / "final.html").write_text("<html>bundled final</html>", encoding="utf-8")
    cache_root = tmp_path / "cache" / study.guid
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir()

    monkeypatch.setattr(
        "server.plugins.steering.paths.get_bundled_questionnaire_path",
        lambda name="": str(bundled_dir / name if name else bundled_dir),
    )
    monkeypatch.setattr(
        "server.plugins.steering.paths.get_uploads_path",
        lambda name="": str(uploads_dir / name if name else uploads_dir),
    )
    monkeypatch.setattr(
        "server.plugins.steering.paths.get_cache_path",
        lambda guid, name="": str(cache_root / name if name else cache_root),
    )

    client = app.test_client()
    with client.session_transaction() as session:
        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["user_study_guid"] = study.guid

    response = client.get("/sae_steering/finish-user-study")

    assert response.status_code == 302
    assert "/utils/final-questionnaire" in response.headers["Location"]
    assert (cache_root / "final.html").exists()


def test_complete_study_records_final_questionnaire_and_completes_run(app_ctx):
    app, db = app_ctx
    app.config["WTF_CSRF_ENABLED"] = False
    study, participation = _seed_participation(db)
    study.settings = json.dumps(
        {
            "dataset": "ml-32m-filtered",
            "questionnaire_file": "final.html",
            "models": [
                {
                    "id": "approach_a",
                    "name": "Approach A",
                    "base": "elsa",
                    "sae": "TopKSAE-1024",
                    "steering_mode": "both",
                    "enabled_modalities": ["sliders", "text", "reset"],
                }
            ],
        }
    )
    db.session.commit()

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.persistence.models import (
            SaeQuestionnaireResponse,
            SaeStudyRun,
        )
        from server.plugins.steering.service import audit

        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["user_study_guid"] = study.guid
        session["approach_order"] = [0]
        session["current_phase"] = 0

        audit.ensure_study_run(participation.id)

    client = app.test_client()
    with client.session_transaction() as session:
        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["user_study_guid"] = study.guid
        session["approach_order"] = [0]
        session["current_phase"] = 0

    response = client.post(
        "/sae_steering/_complete-study",
        data={
            "final_questionnaire_data": "final_questionnaire_data",
            "f1_preference": "approach_a",
        },
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/utils/finish")

    with app.app_context():
        questionnaire = SaeQuestionnaireResponse.query.one()
        study_run = SaeStudyRun.query.one()
        assert questionnaire.response_type == "final"
        assert questionnaire.questionnaire_file == "final.html"
        assert questionnaire.answers["f1_preference"] == "approach_a"
        assert study_run.status == "completed"
        assert study_run.finished_at is not None
