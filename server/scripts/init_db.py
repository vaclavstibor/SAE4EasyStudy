"""Idempotent database bootstrap for any SQLAlchemy backend.

Running ``flask db upgrade`` on a fresh Postgres database is a dead-end
on this project because the existing Alembic revisions carry SQLite-only
idioms (``CREATE TABLE … INTEGER PRIMARY KEY AUTOINCREMENT`` + ``DROP``/
``RENAME`` trick) that Postgres rejects with ``syntax error at or near
'AUTOINCREMENT'``.

The models themselves define everything the production schema needs —
``ondelete='CASCADE'`` included — so we let SQLAlchemy build the schema
natively on first boot and then stamp Alembic at ``head`` so future
``flask db upgrade`` calls stay fast no-ops. If the application schema
already exists (either because a prior deploy bootstrapped it or because
we're running against the legacy SQLite file), we defer to
``flask db upgrade`` for any pending migrations.

The script prints one line to stdout per decision so the deploy log
stays scrutable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


APPLICATION_TABLES = ("userstudy", "participation", "interaction", "message", "user")


def _run_flask(cmd: list[str]) -> int:
    env = os.environ.copy()
    env.setdefault("FLASK_APP", "app:create_app")
    proc = subprocess.run(["flask", *cmd], env=env, check=False)
    return proc.returncode


def main() -> int:
    # Import after logging env so we get consistent output even when the
    # import itself prints diagnostics (SAE bootstrap, redis probing, ...).
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here))
    from app import create_app  # noqa: E402
    from models import db  # noqa: E402
    from sqlalchemy import inspect  # noqa: E402

    app = create_app()
    with app.app_context():
        insp = inspect(db.engine)
        existing = set(insp.get_table_names())
        has_app_schema = any(t in existing for t in APPLICATION_TABLES)

        if has_app_schema:
            print(
                "[init-db] application tables already present "
                f"({sorted(t for t in APPLICATION_TABLES if t in existing)}); "
                "running flask db upgrade for any pending migrations"
            )
            rc = _run_flask(["db", "upgrade"])
            if rc != 0:
                print(
                    "[init-db] flask db upgrade returned non-zero "
                    f"({rc}); check logs — app will still boot"
                )
            return rc

        print(
            f"[init-db] empty database detected at {db.engine.url!r}; "
            "creating schema via db.create_all()"
        )
        db.create_all()
        print("[init-db] schema created; stamping Alembic head to keep future upgrades no-op")

    rc = _run_flask(["db", "stamp", "head"])
    if rc != 0:
        print(
            f"[init-db] flask db stamp head returned non-zero ({rc}); "
            "migrations will run on next boot — usually harmless"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
