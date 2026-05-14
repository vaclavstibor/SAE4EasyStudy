"""Smoke test for the ``/healthz`` liveness probe."""


def test_healthz_endpoint(test_app):
    app, _ = test_app
    client = app.test_client()

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "ok"
