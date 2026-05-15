"""SQLAlchemy, session, and CSRF singletons plus database URL resolution."""

import os
from pathlib import Path

from flask_session import Session
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from sqlalchemy import MetaData

naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

DEFAULT_DATABASE_URL = "sqlite:///db.sqlite"
SERVER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE_PATH = SERVER_ROOT / "instance"


def resolve_database_url() -> str:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    # Connector for some cases (e.g. Railway deployment): https://docs.railway.app/databases/postgresql
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    # Connector for local/docker deployment
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        sqlite_path = url.removeprefix("sqlite:///")
        if sqlite_path and sqlite_path != ":memory:":
            DEFAULT_INSTANCE_PATH.mkdir(parents=True, exist_ok=True)
            absolute_path = (DEFAULT_INSTANCE_PATH / sqlite_path).resolve()
            url = f"sqlite:///{absolute_path}"
    return url


db = SQLAlchemy(metadata=MetaData(naming_convention=naming_convention))
csrf = CSRFProtect()
sess = Session()

__all__ = ["csrf", "db", "naming_convention", "resolve_database_url", "sess"]
