import time
import flask
from flask_pluginkit import PluginManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from flask_session import Session

import os
import random
import sys
from urllib.parse import urlparse
import numpy as np
try:
    import tensorflow as tf
except Exception:
    tf = None

import redis

from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine

naming_convention = {
    "ix": 'ix_%(column_0_label)s',
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

db = SQLAlchemy(metadata = MetaData(naming_convention=naming_convention))
migrate = Migrate()
pm = PluginManager(plugins_folder="plugins")
csrf = CSRFProtect()

sess = Session()

def _create_redis_client():
    # Railway provides REDIS_URL environment variable, but we also want to support separate variables for host, port, db and password for easier local development and testing
    redis_url = os.environ.get("REDIS_URL", "").strip()
    if redis_url:
        parsed = urlparse(redis_url)
        db_index = int(parsed.path.lstrip("/") or os.environ.get("REDIS_DB", "0"))
        return redis.Redis(
            host=parsed.hostname or os.environ.get("REDIS_HOST", "localhost"),
            port=parsed.port or int(os.environ.get("REDIS_PORT", "6379")),
            db=db_index,
            username=parsed.username,
            password=parsed.password,
        )

    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        password=os.environ.get("REDIS_PASSWORD", "") or None,
    )


rds = _create_redis_client()

from models import *

# This is needed to ensure foreign keys and corresponding cascade deletion work as
# expected when SQLite is used as backend for SQLAlchemy.  Postgres enforces FKs
# natively and does not understand PRAGMA, so guard on driver type — otherwise
# every new Postgres connection throws `syntax error at or near "PRAGMA"` and
# app boot fails before Gunicorn comes up.
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

# Insert/set all values that have to be set once (e.g. insert interaction types into DB)
def initialize_db_tables():
    pass
    # from models import InteractionType

    # # If it has not been inserted yet, insert selected-item interaction type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='selected-item').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "selected-item"
    #     db.session.add(x)

    # # If it has not been inserted yet, insert deselected-item interaction type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='deselected-item').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "deselected-item"
    #     db.session.add(x)

    # # If it has not been inserted yet, insert changed-viewport type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='changed-viewport').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "changed-viewport"
    #     db.session.add(x)

    # # If it has not been inserted yet, insert clicked-button type
    # if db.session.query(
    #     db.session.query(InteractionType).filter_by(name='clicked-button').exists()
    # ).scalar():
    #     x = InteractionType()
    #     x.name = "clicked-button"
    #     db.session.add(x)

def create_app():
    app = flask.Flask(__name__)
    #app.wsgi_app = ProfilerMiddleware(app.wsgi_app)

    @app.get("/healthz")
    def healthz():
        return "ok", 200

    app.config['SECRET_KEY'] = os.environ.get('APP_SECRET_KEY', '8bf29bd88d0bfb94509f5fb0')
    # Railway / Heroku-style hosted Postgres add-ons hand back URLs prefixed
    # with the legacy `postgres://` scheme.  SQLAlchemy 1.4+ requires
    # `postgresql://`, so normalise here before it reaches the engine factory.
    _db_url = os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite')
    if _db_url.startswith('postgres://'):
        _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
    app.config['SESSION_COOKIE_NAME'] = os.environ.get('SESSION_COOKIE_NAME', "something")
    app.config["SESSION_TYPE"] = "sqlalchemy"
    app.config["SESSION_SQLALCHEMY"] = db

    db.init_app(app)

    sess.init_app(app)

    migrate.init_app(app, db, render_as_batch=True)

    csrf.init_app(app)

    login_manager = LoginManager(app)
    
    pm.init_app(app)


    @login_manager.user_loader
    def user_loader(user_id):
        """Given *user_id*, return the associated User object.

        :param unicode user_id: user_id (email) user to retrieve

        """
        return User.query.get(user_id)

    from main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint)

    with app.app_context():
        db.create_all()
        initialize_db_tables()

    # Seed setting in the case we use --preload with multiple workers and want to improve randomization on the first iteration
    # Otherwise we can just assume that this will be random enough given that users are distributed to workers randomly
    time_int = int(time.time())
    seed = os.getpid() + time_int
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
    print(f"Seeding with: {seed} ({time_int}, {os.getpid()})", file=sys.stderr)

    return app