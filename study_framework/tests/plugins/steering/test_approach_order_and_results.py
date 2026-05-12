import json
from datetime import datetime


def _three_approach_config(*, randomize=True):
    return {
        "enable_comparison": True,
        "comparison_mode": "sequential",
        "randomize_approach_order": randomize,
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
            {
                "id": "approach_c",
                "name": "Approach C",
                "base": "elsa",
                "sae": "TopKSAE-1024",
                "steering_mode": "text",
                "enabled_modalities": ["text", "reset"],
            },
        ],
    }


def _seed_three_approach_participation(db, *, randomize=True):
    from server.platform.persistence.base_models import Participation, User, UserStudy

    user = User(email="admin@example.com", password="x", authenticated=True, admin=True)
    study = UserStudy(
        creator=user.email,
        guid="study-guid-3",
        parent_plugin="sae_steering",
        settings=json.dumps(_three_approach_config(randomize=randomize)),
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
        uuid="participant-uuid-3",
        language="en",
    )
    db.session.add(participation)
    db.session.commit()
    return study, participation


def test_get_effective_models_keeps_fixed_order_for_n_approaches(app_ctx):
    app, _ = app_ctx

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.service.participation import get_effective_models

        models = get_effective_models(_three_approach_config(randomize=False))

        assert [model["name"] for model in models] == ["Approach A", "Approach B", "Approach C"]
        assert session["approach_order"] == [0, 1, 2]


def test_get_effective_models_randomizes_n_approaches_once(app_ctx, monkeypatch):
    app, _ = app_ctx

    class FakeRandom:
        def shuffle(self, values):
            values[:] = [2, 0, 1]

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.service import participation

        monkeypatch.setattr(participation.secrets, "SystemRandom", lambda: FakeRandom())

        models = participation.get_effective_models(_three_approach_config(randomize=True))
        models_again = participation.get_effective_models(_three_approach_config(randomize=True))

        assert [model["name"] for model in models] == ["Approach C", "Approach A", "Approach B"]
        assert [model["name"] for model in models_again] == ["Approach C", "Approach A", "Approach B"]
        assert session["approach_order"] == [2, 0, 1]


def test_audit_study_run_uses_canonical_order_mapping_for_randomized_n_approaches(app_ctx):
    app, db = app_ctx
    study, participation = _seed_three_approach_participation(db, randomize=True)

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.service import audit

        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["approach_order"] = [2, 0, 1]
        session["current_phase"] = 0

        study_run = audit.ensure_study_run(participation.id, approach_order=[2, 0, 1])

        assert study_run.approach_order == [2, 0, 1]
        assert study_run.effective_order == ["Approach C", "Approach A", "Approach B"]


