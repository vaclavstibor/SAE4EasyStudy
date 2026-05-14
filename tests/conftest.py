"""Shared pytest fixtures (test Flask app + in-memory SQLite database)."""

import os

import pytest


@pytest.fixture(scope="session")
def test_app(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["APP_SECRET_KEY"] = "test-secret"

    from server.platform.app import create_app
    from server.platform.persistence.db import db

    return create_app(), db


@pytest.fixture()
def app_ctx(test_app):
    app, db = test_app
    with app.app_context():
        db.drop_all()
        db.create_all()
        yield app, db
        db.session.remove()
        db.drop_all()


@pytest.fixture(autouse=True)
def _stable_env(monkeypatch, test_app):
    app, db = test_app
    with app.app_context():
        monkeypatch.setenv("DATABASE_URL", str(db.engine.url))
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret")
