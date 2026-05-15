"""Smoke tests for the LayoutShuffling plugin contract and demo flow.

The upstream EasyStudy version of this plugin required a `utils` blueprint
(preference elicitation, search, etc.) that is not part of this repository.
The remaining routes in :mod:`server.plugins.layoutshuffling` are kept as a
source-level illustration; only `create`, `initialize`, and `join` are wired
end-to-end so the admin can prove the plugin contract works.
"""

import datetime

from server.plugins.layoutshuffling import get_plugin


def test_layoutshuffling_contract_metadata():
    plugin = get_plugin()
    assert plugin.metadata.name == "layoutshuffling"
    assert plugin.blueprint.url_prefix == "/layoutshuffling"


def test_layoutshuffling_initialize_activates_study_and_join_is_reachable(app_ctx):
    """`/layoutshuffling/initialize?guid=...` must flip the UserStudy row to
    initialized + active synchronously, and `/layoutshuffling/join` must
    render a public-facing page so the admin → create → initialize → join
    flow works end-to-end.
    """
    app, db = app_ctx
    from server.platform.persistence.base_models import User, UserStudy

    db.session.add(
        User(email="admin@example.com", password="x", authenticated=True, admin=True)
    )
    db.session.add(
        UserStudy(
            creator="admin@example.com",
            guid="layoutshuffling-demo-guid",
            parent_plugin="layoutshuffling",
            settings="{}",
            time_created=datetime.datetime.utcnow(),
            active=False,
            initialized=False,
            initialization_error=None,
        )
    )
    db.session.commit()

    client = app.test_client()
    init_response = client.get(
        "/layoutshuffling/initialize"
        "?guid=layoutshuffling-demo-guid&continuation_url=/administration"
    )

    db.session.remove()
    refreshed = UserStudy.query.filter_by(
        guid="layoutshuffling-demo-guid"
    ).first()

    assert init_response.status_code == 302
    assert init_response.headers["Location"].endswith("/administration")
    assert refreshed.initialized is True
    assert refreshed.active is True

    join_response = client.get(
        "/layoutshuffling/join?guid=layoutshuffling-demo-guid"
    )
    assert join_response.status_code == 200
    assert b"layoutshuffling-demo-guid" in join_response.data
