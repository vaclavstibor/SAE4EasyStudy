"""Idempotent database bootstrap.

Models in ``server.platform.persistence.base_models`` and each plugin's
``persistence.models`` module are the single source of truth for the
schema. ``create_app()`` already runs ``db.create_all()`` on every boot,
so this script simply forces that to happen explicitly and reports the
final table set.

For a clean slate (drop + recreate) use ``scripts/reset-db.sh``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path.cwd()
    sys.path.insert(0, str(repo_root))

    from sqlalchemy import inspect

    from server.platform.app import create_app
    from server.platform.persistence.db import db

    app = create_app()
    with app.app_context():
        db.create_all()
        tables = sorted(inspect(db.engine).get_table_names())
    print(f"[init-db] schema ready ({len(tables)} tables): {', '.join(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
