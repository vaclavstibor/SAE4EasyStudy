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


def test_fastcompare_lifecycle_routes(test_app):
    app, _ = test_app
    client = app.test_client()

    initialize_response = client.get(
        "/fastcompare/initialize?continuation_url=/administration"
    )
    join_response = client.get("/fastcompare/join")
    results_response = client.get("/fastcompare/results")

    assert initialize_response.status_code == 302
    assert initialize_response.headers["Location"].endswith("/administration")
    assert join_response.status_code == 200
    assert results_response.status_code == 200
