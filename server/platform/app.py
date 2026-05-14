"""Flask application factory and platform composition root.

Wires together the platform blueprints (auth, admin, participant flow), loads the
canonical study plugins, and configures persistence, sessions, and CSRF protection.
"""

import os
import random
import time
from importlib import import_module
from pathlib import Path

import flask
import numpy as np
from flask_login import LoginManager
from sqlalchemy import event
from sqlalchemy.engine import Engine

from server.platform.admin import main as main_blueprint
from server.platform.auth import auth as auth_blueprint
from server.platform.participant_flow import bp as participant_flow_blueprint
from server.platform.participant_flow import emit_assets
from server.platform.persistence.base_models import User
from server.platform.persistence.db import csrf, db, resolve_database_url, sess
from server.platform.runtime.plugin_registry import load_canonical_plugin_contracts

try:
    import tensorflow as tf
except Exception:
    tf = None


PLATFORM_ROOT = Path(__file__).resolve().parent
SERVER_ROOT = PLATFORM_ROOT.parent


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    driver = type(dbapi_connection).__module__ or ""
    if "sqlite" not in driver.lower():
        return
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def create_app() -> flask.Flask:
    """Build and return the configured Flask application.

    Returns:
        The application instance with all blueprints, persistence, and plugin
        contracts registered.
    """
    app = flask.Flask(
        "server",
        template_folder=str(PLATFORM_ROOT / "web" / "templates"),
        static_folder=str(SERVER_ROOT / "static"),
    )

    @app.get("/healthz")
    def healthz():
        return "ok", 200

    app.config["SECRET_KEY"] = os.environ.get("APP_SECRET_KEY", "8bf29bd88d0bfb94509f5fb0")
    db_url = resolve_database_url()
    app.config["SQLALCHEMY_DATABASE_URI"] = db_url

    backend = (
        "postgres"
        if db_url.startswith("postgresql")
        else ("sqlite" if db_url.startswith("sqlite") else db_url.split(":", 1)[0])
    )
    print(f"[startup] SQLAlchemy backend: {backend}", flush=True)
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }
    app.config["SESSION_COOKIE_NAME"] = os.environ.get("SESSION_COOKIE_NAME", "something")
    app.config["SESSION_TYPE"] = "sqlalchemy"
    app.config["SESSION_SQLALCHEMY"] = db

    db.init_app(app)
    sess.init_app(app)
    csrf.init_app(app)

    login_manager = LoginManager(app)
    app.jinja_env.globals["emit_assets"] = emit_assets

    @login_manager.user_loader
    def user_loader(user_id):
        return User.query.get(user_id)

    app.register_blueprint(main_blueprint)
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(participant_flow_blueprint)

    plugin_contracts = load_canonical_plugin_contracts()
    app.extensions["study_plugin_contracts"] = plugin_contracts
    for plugin in plugin_contracts:
        app.register_blueprint(plugin.blueprint)
        # Eagerly import the plugin's models module (declared via persistence_hooks)
        # so SQLAlchemy sees its tables before db.create_all(). Keeps the platform
        # free of hard-coded references to specific plugin packages.
        models_module = plugin.persistence_hooks.get("models_module")
        if models_module:
            import_module(models_module)

    with app.app_context():
        db.create_all()

    time_int = int(time.time())
    seed = os.getpid() + time_int
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
    print(f"Seeding with: {seed} ({time_int}, {os.getpid()})", flush=True)

    return app
