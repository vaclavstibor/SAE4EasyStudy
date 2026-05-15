"""Smoke tests for the upstream FastCompare plugin contract and routes."""

from server.plugins.fastcompare import get_plugin


def test_fastcompare_contract_metadata():
    plugin = get_plugin()
    assert plugin.metadata.name == "fastcompare"
    assert plugin.blueprint.url_prefix == "/fastcompare"


def test_fastcompare_health_route(test_app):
    app, _ = test_app
    client = app.test_client()

    response = client.get("/fastcompare/health")

    assert response.status_code == 200
    assert response.get_json()["plugin_id"] == "fastcompare"


def test_fastcompare_lifecycle_routes(app_ctx):
    app, _ = app_ctx
    client = app.test_client()

    initialize_response = client.get("/fastcompare/initialize?continuation_url=/administration")
    join_response = client.get("/fastcompare/join")
    results_response = client.get("/fastcompare/results")

    assert initialize_response.status_code == 302
    assert initialize_response.headers["Location"].endswith("/administration")
    assert join_response.status_code == 200
    assert results_response.status_code == 200


def test_fastcompare_initialize_activates_user_study(app_ctx):
    """`/fastcompare/initialize?guid=...` must flip the matching UserStudy row
    to initialized + active so the admin "Initialize" button completes the
    create → initialize → join flow.
    """
    import datetime

    app, db = app_ctx
    from server.platform.persistence.base_models import User, UserStudy

    db.session.add(
        User(email="admin@example.com", password="x", authenticated=True, admin=True)
    )
    db.session.add(
        UserStudy(
            creator="admin@example.com",
            guid="fastcompare-demo-guid",
            parent_plugin="fastcompare",
            settings="{}",
            time_created=datetime.datetime.utcnow(),
            active=False,
            initialized=False,
            initialization_error=None,
        )
    )
    db.session.commit()

    client = app.test_client()
    response = client.get(
        "/fastcompare/initialize?guid=fastcompare-demo-guid&continuation_url=/administration"
    )

    db.session.remove()
    refreshed = UserStudy.query.filter_by(guid="fastcompare-demo-guid").first()

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/administration")
    assert refreshed.initialized is True
    assert refreshed.active is True

    join_response = client.get("/fastcompare/join?guid=fastcompare-demo-guid")
    assert join_response.status_code == 200
