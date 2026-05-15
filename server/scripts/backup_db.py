#!/usr/bin/env python3
"""Database backup helper for EasyStudy.

Two entry points share the same core implementation:

* :func:`create_backup_now` — used by the admin download endpoint
  ``/administration/db-backup``. The admin clicks the button, a fresh
  snapshot is written, pruned, and streamed back as a file.
* :func:`main` (this script's ``__main__``) — same logic as a CLI for
  ad-hoc or externally-scheduled use:

      python server/scripts/backup_db.py

Behaviour:

* If ``DATABASE_URL`` points at Postgres (``postgres://`` or
  ``postgresql://``), shell out to ``pg_dump`` and gzip the output.
* Otherwise falls back to copying the SQLite file at the URL path
  (sensible default for local Docker dev).
* Writes ``db_<UTC>.sql.gz`` (or ``.sqlite.gz``) into the backup
  directory resolved by
  :func:`server.platform.shared.common.resolve_backup_dir`. By default
  this is ``<repo_root>/backups`` (which is ``/app/backups`` inside the
  Docker image; on Railway the entrypoint symlinks that to
  ``${DATA_ROOT}/backups`` so backups land on the persistent volume).
* Keeps the most recent ``KEEP_LAST`` (default 14) backups, deletes the rest.

Env vars:

* ``DATABASE_URL``   - same URL the Flask app uses.
* ``BACKUP_DIR``     - explicit override for the destination directory.
* ``KEEP_LAST``      - rolling retention count (default: 14).

The CLI exits non-zero on failure so an external scheduler can flag it.
"""

from __future__ import annotations

import datetime
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

# When invoked as a script (``python server/scripts/backup_db.py``) the
# Python launcher only adds ``server/scripts`` to ``sys.path``, which does
# not let us import the ``server`` package. Mirror what ``init_db.py`` does
# and put the repo root on the path before the absolute imports below.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from server.platform.persistence.db import resolve_database_url  # noqa: E402
from server.platform.shared.common import resolve_backup_dir  # noqa: E402


class BackupError(RuntimeError):
    """Raised when the backup pipeline cannot produce a snapshot."""


def _ensure_backup_dir() -> Path:
    backup_dir = resolve_backup_dir()
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def _prune(backup_dir: Path, keep: int) -> list[Path]:
    """Delete all but the ``keep`` most recent ``db_*.gz`` files.

    Returns the list of files that were deleted (best-effort).
    """
    files = sorted(
        [p for p in backup_dir.glob("db_*.gz") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    pruned: list[Path] = []
    for stale in files[keep:]:
        try:
            stale.unlink()
            pruned.append(stale)
        except OSError as exc:
            print(f"[backup] could not prune {stale}: {exc}", file=sys.stderr)
    return pruned


def _dump_postgres(url: str, dest: Path) -> None:
    if shutil.which("pg_dump") is None:
        raise BackupError(
            "pg_dump not installed in this image — install postgresql-client or "
            "switch to the sqlite fallback for local development."
        )
    with gzip.open(dest, "wb") as gz:
        proc = subprocess.run(
            ["pg_dump", "--no-owner", "--no-privileges", url],
            stdout=subprocess.PIPE,
            check=True,
        )
        gz.write(proc.stdout)


def _copy_sqlite(url: str, dest: Path) -> None:
    parsed = urlparse(url)
    # SQLAlchemy URLs are "sqlite:///relative/path" or "sqlite:////abs/path".
    sqlite_path = parsed.path
    if url.startswith("sqlite:///") and not sqlite_path.startswith("/"):
        sqlite_path = "/" + sqlite_path
    sqlite_path = sqlite_path.lstrip("/") if not os.path.isabs(sqlite_path) else sqlite_path
    src = Path(sqlite_path)
    if not src.exists():
        # Fall back to the repo's `server/` directory for relative paths
        # (matches the resolution rules used elsewhere when the URL is
        # `sqlite:///instance/db.sqlite` and CWD is the repo root).
        repo_candidate = Path(__file__).resolve().parents[1] / sqlite_path
        if repo_candidate.exists():
            src = repo_candidate
        else:
            raise BackupError(f"SQLite database not found at {src}")
    with open(src, "rb") as fsrc, gzip.open(dest, "wb") as gz:
        shutil.copyfileobj(fsrc, gz)


def create_backup_now() -> Path:
    """Create a fresh DB snapshot, prune old ones, return the new file path.

    Raises ``BackupError`` for any expected failure (unsupported URL scheme,
    missing ``pg_dump``, missing SQLite file). Lets unexpected exceptions
    propagate so the caller can decide how to surface them.
    """
    url = resolve_database_url()
    backup_dir = _ensure_backup_dir()
    keep = int(os.environ.get("KEEP_LAST", "14"))
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if url.startswith("postgresql://"):
        dest = backup_dir / f"db_{stamp}.sql.gz"
        try:
            _dump_postgres(url, dest)
        except subprocess.CalledProcessError as exc:
            # Surface a clean message instead of a raw CalledProcessError.
            raise BackupError(f"pg_dump failed (exit={exc.returncode})") from exc
    elif url.startswith("sqlite"):
        dest = backup_dir / f"db_{stamp}.sqlite.gz"
        _copy_sqlite(url, dest)
    else:
        raise BackupError(f"unsupported DATABASE_URL scheme: {url}")

    _prune(backup_dir, keep)
    return dest


def main() -> int:
    try:
        dest = create_backup_now()
    except BackupError as exc:
        print(f"[backup] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[backup] unexpected failure: {exc}", file=sys.stderr)
        return 1

    print(f"[backup] wrote {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
