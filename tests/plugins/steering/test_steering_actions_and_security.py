"""End-to-end coverage for the new steering endpoints + auth guards.

Covers
------
- POST /sae_steering/reset (FR-12)              writes exactly one SaeResetAction +
                                                one SaeSteeringEvent envelope and clears
                                                the in-session adjustment state.
- POST /sae_steering/parse-text-steering        200-char enforcement (FR-09) and the
                                                NFR-12 graceful "no-match" path.
- _compose_text_adjustments                     replace / add / intersect modes (FR-09).
- GET  /sae_steering/export-csv/<guid>          FR-17 ZIP export requires admin login.
- Researcher admin routes (/loaded-plugins,
  /existing-user-studies, /user-study,
  /results/<plugin>/<guid>)                     unauthenticated callers must be bounced
                                                to /login (security hardening).
"""

import io
import json
import zipfile
from datetime import datetime

import pytest

from server.plugins.steering.routes.steering.actions import _compose_text_adjustments

# Seed helpers


def _seed(db, *, admin_password: str = "hashed-pw"):
    """Create one admin user, one initialised study, one participation."""
    from server.platform.persistence.base_models import Participation, User, UserStudy

    admin = User(
        email="admin@example.com",
        password=admin_password,
        authenticated=True,
        admin=True,
    )
    study = UserStudy(
        creator=admin.email,
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
    db.session.add(admin)
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
    return admin, study, participation


def _login_session(client, email):
    """Inject a logged-in flask-login session for an existing User row."""
    with client.session_transaction() as sess:
        sess["_user_id"] = email
        sess["_fresh"] = True


# Pure unit: text composition modes (FR-09)


class TestComposeTextAdjustments:
    def test_replace_drops_previous(self):
        result = _compose_text_adjustments(
            "replace",
            previous={"cluster_a": 0.4},
            current={"cluster_b": 0.6},
        )
        assert result == {"cluster_b": 0.6}

    def test_replace_with_no_previous(self):
        result = _compose_text_adjustments("replace", previous={}, current={"x": 0.3})
        assert result == {"x": 0.3}

    def test_add_merges_and_clamps(self):
        result = _compose_text_adjustments(
            "add",
            previous={"shared": 0.5, "old": -0.2},
            current={"shared": 0.6, "new": 0.4},
        )
        assert result["new"] == 0.4
        assert result["old"] == -0.2
        # 0.5 + 0.6 saturates to the +0.95 cap.
        assert result["shared"] == pytest.approx(0.95)

    def test_add_clamps_negative_sum(self):
        result = _compose_text_adjustments(
            "add",
            previous={"x": -0.7},
            current={"x": -0.7},
        )
        assert result["x"] == pytest.approx(-0.95)

    def test_intersect_keeps_only_overlap_and_uses_current_weight(self):
        result = _compose_text_adjustments(
            "intersect",
            previous={"shared": 0.5, "only_prev": 0.4},
            current={"shared": 0.7, "only_curr": 0.3},
        )
        assert result == {"shared": 0.7}

    def test_unknown_mode_falls_back_to_default_replace(self):
        result = _compose_text_adjustments(
            "garbage",
            previous={"a": 0.5},
            current={"b": 0.3},
        )
        assert result == {"b": 0.3}


# /reset (FR-12)


def test_reset_writes_one_audit_row_and_clears_session(app_ctx):
    app, db = app_ctx
    app.config["WTF_CSRF_ENABLED"] = False
    _, study, participation = _seed(db)

    from server.plugins.steering.persistence.models import (
        SaeResetAction,
        SaeSteeringEvent,
    )

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["participation_id"] = participation.id
        sess["user_study_id"] = study.id
        sess["user_study_guid"] = study.guid
        sess["approach_order"] = [0, 1]
        sess["current_phase"] = 0
        sess["iteration"] = 2
        sess["cumulative_adjustments"] = {"feat_1": 0.5}
        sess["feature_adjustments"] = {"feat_1": 0.5}
        sess["user_touched_features"] = ["feat_1"]
        sess["last_text_steering"] = {"query": "old query", "adjustments": {"feat_1": 0.5}}

    response = client.post(
        "/sae_steering/reset",
        data=json.dumps({"scope": "all-features", "trigger": "manual-ui-reset"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body == {"status": "ok", "scope": "all-features"}

    with app.app_context():
        reset_rows = SaeResetAction.query.all()
        assert len(reset_rows) == 1
        reset = reset_rows[0]
        assert reset.participation_id == participation.id
        assert reset.scope == "all-features"
        assert reset.trigger == "manual-ui-reset"

        envelopes = SaeSteeringEvent.query.filter_by(event_type="global-reset").all()
        assert len(envelopes) == 1
        envelope = envelopes[0]
        assert envelope.modality == "reset"
        assert envelope.source == "reset"
        assert envelope.raw_payload["scope"] == "all-features"
        assert reset.event_id == envelope.id

    with client.session_transaction() as sess:
        assert sess["cumulative_adjustments"] == {}
        assert sess["feature_adjustments"] == {}
        assert sess["user_touched_features"] == []
        assert sess["last_text_steering"] == {}


def test_reset_without_participation_returns_ok_and_writes_no_rows(app_ctx):
    """No participation in session = no audit row, but UI state is still cleared."""
    app, db = app_ctx
    app.config["WTF_CSRF_ENABLED"] = False
    _seed(db)

    from server.plugins.steering.persistence.models import SaeResetAction

    client = app.test_client()
    response = client.post(
        "/sae_steering/reset",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
    with app.app_context():
        assert SaeResetAction.query.count() == 0


# /parse-text-steering (FR-09 + NFR-12)


def test_parse_text_steering_rejects_oversize_query_with_400(app_ctx):
    app, db = app_ctx
    app.config["WTF_CSRF_ENABLED"] = False
    _, study, participation = _seed(db)

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["participation_id"] = participation.id
        sess["user_study_id"] = study.id
        sess["approach_order"] = [0, 1]
        sess["current_phase"] = 0
        sess["iteration"] = 1

    long_query = "marvel " * 60  # 420 chars, well above the 200-char cap.
    response = client.post(
        "/sae_steering/parse-text-steering",
        data=json.dumps({"query": long_query}),
        content_type="application/json",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert "max" in body["message"].lower()
    assert body["max_chars"] == 200


def test_parse_text_steering_no_match_returns_graceful_message(app_ctx, monkeypatch):
    """NFR-12: text that resolves to zero clusters must NOT crash."""
    app, db = app_ctx
    app.config["WTF_CSRF_ENABLED"] = False
    _, study, participation = _seed(db)

    monkeypatch.setattr(
        "server.plugins.steering.modalities.text.load_semantic_clusters",
        lambda _sae_id: {"clusters": []},
    )

    client = app.test_client()
    with client.session_transaction() as sess:
        sess["participation_id"] = participation.id
        sess["user_study_id"] = study.id
        sess["approach_order"] = [0, 1]
        sess["current_phase"] = 0
        sess["iteration"] = 1

    response = client.post(
        "/sae_steering/parse-text-steering",
        data=json.dumps({"query": "completely unmatchable gibberish xyzzy"}),
        content_type="application/json",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "no-match"
    assert body["matched"] == 0
    assert body["adjustments"] == {}
    assert "could not match" in body["message"].lower()


# CSV export (FR-17) + admin authentication


def test_export_csv_rejects_unauthenticated_callers(app_ctx):
    app, db = app_ctx
    _, study, _ = _seed(db)

    client = app.test_client()
    response = client.get(f"/sae_steering/export-csv/{study.guid}", follow_redirects=False)
    assert response.status_code in (302, 401)
    if response.status_code == 302:
        assert "/login" in response.headers["Location"]


def test_export_csv_authed_returns_zip_with_expected_files(app_ctx):
    app, db = app_ctx
    admin, study, _ = _seed(db)

    client = app.test_client()
    _login_session(client, admin.email)

    response = client.get(f"/sae_steering/export-csv/{study.guid}")
    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert f"study_{study.guid}_csv_export.zip" in response.headers["Content-Disposition"]

    archive = zipfile.ZipFile(io.BytesIO(response.data))
    names = set(archive.namelist())
    expected = {
        "sae_study_run.csv",
        "sae_approach_run.csv",
        "sae_steering_event.csv",
        "sae_feature_adjustment.csv",
        "sae_feature_search.csv",
        "sae_feature_search_hit.csv",
        "sae_text_steering_query.csv",
        "sae_text_steering_match.csv",
        "sae_example_steering.csv",
        "sae_example_steering_movie.csv",
        "sae_reset_action.csv",
        "sae_movie_feedback.csv",
        "sae_recommendation_set.csv",
        "sae_recommendation_item.csv",
        "sae_questionnaire_response.csv",
        "sae_elicitation_pick.csv",
    }
    assert expected.issubset(names), f"missing CSVs: {expected - names}"
    # Every CSV must have a non-empty header row even if it has no data rows.
    for name in expected:
        first_line = archive.read(name).decode("utf-8").splitlines()[0]
        assert first_line, f"{name} has no header"


def test_export_csv_unknown_guid_returns_404(app_ctx):
    app, db = app_ctx
    admin, _, _ = _seed(db)

    client = app.test_client()
    _login_session(client, admin.email)

    response = client.get("/sae_steering/export-csv/no-such-guid")
    assert response.status_code == 404


# Researcher routes: @login_required regression net


RESEARCHER_GET_ROUTES = [
    "/loaded-plugins",
    "/existing-user-studies",
    "/user-study?user_study_id=1",
    "/user-study-participants?user_study_id=1",
    "/user-participated-user-studies?user_email=admin@example.com",
    "/results/sae_steering/study-guid",
]


@pytest.mark.parametrize("path", RESEARCHER_GET_ROUTES)
def test_researcher_routes_require_login(app_ctx, path):
    app, db = app_ctx
    _seed(db)
    client = app.test_client()
    response = client.get(path, follow_redirects=False)
    assert response.status_code in (302, 401), (
        f"{path} returned {response.status_code} for an unauthenticated caller"
    )
    if response.status_code == 302:
        assert "/login" in response.headers["Location"]
