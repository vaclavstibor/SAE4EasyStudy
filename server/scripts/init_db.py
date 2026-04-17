"""Idempotent database bootstrap for any SQLAlchemy backend.

Why this exists
---------------
``server/app.py::create_app`` already runs ``db.create_all()`` on every
boot, so the application schema always matches ``models.py`` after the
app is imported.  The legacy Alembic revisions in ``migrations/versions``
were authored for an old SQLite file and use SQLite-only DDL
(``CREATE TABLE … INTEGER PRIMARY KEY AUTOINCREMENT`` plus a
drop/rename dance).  On Postgres they fail with
``syntax error at or near "AUTOINCREMENT"`` which previously blocked the
deploy and left the ``alembic_version`` table empty.

What this script does
---------------------
1. Imports ``create_app`` (which runs ``db.create_all()`` internally and
   therefore leaves the schema in sync with ``models.py``).
2. Looks at the ``alembic_version`` table:
   * missing or empty → stamps head so the historical SQLite migrations
     are recorded as "already applied" and never execute against a
     fresh Postgres database,
   * already populated → defers to ``flask db upgrade``, which is a
     safe no-op when the DB is on the current head and applies any
     real pending migrations otherwise (the existing revisions are
     SQLite-batch-friendly, so upgrades on legacy SQLite files keep
     working).

The script is idempotent – repeating it after a successful boot logs
"alembic head present" and exits 0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_flask(cmd: list[str]) -> int:
    env = os.environ.copy()
    env.setdefault("FLASK_APP", "app:create_app")
    return subprocess.run(["flask", *cmd], env=env, check=False).returncode


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(here))

    # Import create_app first; it runs db.create_all() internally and
    # initialises Flask-Migrate, giving us a fully configured engine.
    from app import create_app  # noqa: E402
    from models import db  # noqa: E402
    from sqlalchemy import inspect, text  # noqa: E402

    app = create_app()

    with app.app_context():
        insp = inspect(db.engine)
        app_tables = {"userstudy", "participation", "interaction", "message", "user"}
        missing = sorted(app_tables - set(insp.get_table_names()))
        if missing:
            # create_app() should have produced these; shout loudly if not.
            print(
                "[init-db] WARNING: application tables missing after "
                f"db.create_all(): {missing}"
            )

        has_version_row = False
        if insp.has_table("alembic_version"):
            with db.engine.connect() as conn:
                has_version_row = (
                    conn.execute(
                        text("SELECT version_num FROM alembic_version LIMIT 1")
                    ).fetchone()
                    is not None
                )

    if not has_version_row:
        print(
            "[init-db] alembic_version empty — schema owned by db.create_all(); "
            "stamping head so legacy SQLite migrations stay skipped"
        )
        return _run_flask(["db", "stamp", "head"])

    print("[init-db] alembic_version populated; running flask db upgrade (no-op if already at head)")
    rc = _run_flask(["db", "upgrade"])
    if rc != 0:
        print(
            f"[init-db] flask db upgrade returned non-zero ({rc}); "
            "app will still boot — inspect migrations and DB manually"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
