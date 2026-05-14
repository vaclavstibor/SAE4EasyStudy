"""Tests for the SAE steering study initialization endpoint."""

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


def test_critical_steering_endpoints_are_registered(app_ctx):
    """Regression guard: every route module under ``routes/steering`` and
    ``routes/results`` must be imported in ``plugin.py`` so its ``@bp.route``
    decorators register. The participant happy-path (preference elicitation →
    show-features → steering) silently breaks otherwise.
    """
    app, _ = app_ctx
    registered = {rule.endpoint for rule in app.url_map.iter_rules()}
    required = {
        "sae_steering.steering",
        "sae_steering.steering_interface",
        "sae_steering.show_features",
        "sae_steering.adjust_features",
        "sae_steering.reset_steering",
        "sae_steering.parse_text_steering",
        "sae_steering.apply_example_steering",
        "sae_steering.initialize",
    }
    missing = required - registered
    assert not missing, f"steering endpoints not registered: {sorted(missing)}"


def test_all_canonical_plugins_load_and_register(app_ctx):
    """Every entry in ``CANONICAL_PLUGIN_MODULES`` must load its plugin contract
    and register at least one route. Catches the case where a new plugin is
    added to the registry but its module crashes at import or its blueprint is
    never wired into Flask.
    """
    app, _ = app_ctx
    from server.platform.runtime.plugin_registry import (
        CANONICAL_PLUGIN_MODULES,
        load_canonical_plugin_contracts,
    )

    contracts = load_canonical_plugin_contracts()
    assert len(contracts) == len(CANONICAL_PLUGIN_MODULES)
    contract_names = {c.metadata.name for c in contracts}
    expected_names = {"sae_steering", "fastcompare", "emptytemplate", "layoutshuffling", "vae"}
    assert contract_names == expected_names, (
        f"plugin contract names mismatch: got {contract_names}, expected {expected_names}"
    )

    rules = list(app.url_map.iter_rules())
    blueprints_with_routes = {rule.endpoint.split(".", 1)[0] for rule in rules}
    for name in expected_names:
        assert name in blueprints_with_routes, f"plugin '{name}' registered no routes"


def test_admin_available_templates_excludes_hidden_plugins(app_ctx):
    """The admin "Available templates" picker (``get_loaded_plugins``) must
    expose study-type plugins only. ``emptytemplate`` (developer scaffold) and
    ``vae`` (algorithm wrapper consumed by ``fastcompare``) opt out via
    ``PluginMetadata.hidden_from_admin`` and must NOT show up.
    """
    app, _ = app_ctx
    from server.platform.admin.routes import get_loaded_plugins

    with app.app_context():
        listed = {entry["plugin_name"] for entry in get_loaded_plugins()}

    assert "emptytemplate" not in listed, (
        "emptytemplate must be hidden from /loaded-plugins (developer scaffold)"
    )
    assert "vae" not in listed, (
        "vae must be hidden from /loaded-plugins (algorithm wrapper, not a study type)"
    )
    assert {"sae_steering", "fastcompare", "layoutshuffling"}.issubset(listed), (
        f"researcher-facing plugins missing from /loaded-plugins: got {listed}"
    )
