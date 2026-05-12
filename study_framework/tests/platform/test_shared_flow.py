import json
from datetime import datetime
from io import BytesIO


def _seed_steering_study(db):
    from server.platform.persistence.base_models import Participation, User, UserStudy

    user = User(email="admin@example.com", password="x", authenticated=True, admin=True)
    study = UserStudy(
        creator=user.email,
        guid="shared-flow-guid",
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


def test_utils_blueprint_and_interaction_routes_registered(test_app):
    app, _ = test_app
    endpoints = set(app.url_map._rules_by_endpoint)

    assert {
        "utils.results",
        "utils.join",
        "utils.preference_elicitation",
        "utils.finish",
        "utils.final_questionnaire",
        "utils.upload",
        "utils.changed_viewport",
        "utils.selected_item",
        "utils.deselected_item",
        "utils.loaded_page",
        "utils.on_input",
        "utils.on_message",
    }.issubset(endpoints)


def test_study_intro_renders_with_utils_continuation(app_ctx):
    app, db = app_ctx
    study, participation = _seed_steering_study(db)
    client = app.test_client()

    with client.session_transaction() as session:
        session["user_study_id"] = study.id
        session["participation_id"] = participation.id
        session["approach_order"] = [0, 1]
        session["current_phase"] = 0

    response = client.get("/sae_steering/study-intro")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "/utils/preference-elicitation" in body


def test_utils_upload_route_accepts_file(app_ctx):
    app, _ = app_ctx
    app.config["WTF_CSRF_ENABLED"] = False
    client = app.test_client()

    response = client.post(
        "/utils/upload",
        data={
            "plugin_name": "sae_steering",
            "file": (BytesIO(b"hello"), "questionnaire.txt"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["upload_name"].endswith("_questionnaire.txt")
