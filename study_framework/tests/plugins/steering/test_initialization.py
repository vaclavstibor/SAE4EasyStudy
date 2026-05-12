import json
from datetime import datetime


def _seed_uninitialized_steering_study(db):
    from server.platform.persistence.base_models import User, UserStudy

    user = User(email="admin@example.com", password="x", authenticated=True, admin=True)
    study = UserStudy(
        creator=user.email,
        guid="steering-init-guid",
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
                    }
                ],
            }
        ),
        time_created=datetime.utcnow(),
        active=False,
        initialized=False,
        initialization_error=None,
    )
    db.session.add(user)
    db.session.add(study)
    db.session.commit()
    return study


def test_initialize_route_marks_study_initialized_and_active(app_ctx):
    app, db = app_ctx
    study = _seed_uninitialized_steering_study(db)

    from server.platform.persistence.base_models import UserStudy
    from server.plugins.steering.routes.admin import initialize

    with app.test_request_context(
        f"/sae_steering/initialize?guid={study.guid}&continuation_url=/administration"
    ):
        response = initialize.__wrapped__()

    db.session.remove()
    refreshed = UserStudy.query.filter(UserStudy.guid == study.guid).first()
    assert response.status_code == 302
    assert response.location.endswith("/administration")
    assert refreshed is not None
    assert refreshed.initialized is True
    assert refreshed.active is True
    assert refreshed.initialization_error is None
