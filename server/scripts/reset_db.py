"""Destructive: drop every table and rebuild the schema from models.

Use this when ``models.py`` changes and you do not need the existing
data. Aborts unless ``--yes`` is passed so it cannot be invoked by
accident.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if "--yes" not in argv:
        print(
            "[reset-db] refusing to run without --yes; this destroys ALL data.\n"
            "           rerun as: python server/scripts/reset_db.py --yes"
        )
        return 1

    repo_root = Path.cwd()
    sys.path.insert(0, str(repo_root))

    from sqlalchemy import inspect

    from server.platform.app import create_app
    from server.platform.persistence.db import db

    app = create_app()
    with app.app_context():
        db.drop_all()
        db.create_all()
        tables = sorted(inspect(db.engine).get_table_names())
    print(f"[reset-db] schema rebuilt ({len(tables)} tables): {', '.join(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
