#!/usr/bin/env python3
"""Database backup helper for EasyStudy.

Designed to run both locally and as a daily Railway cron job:

    python server/scripts/backup_db.py

Behaviour:

  * If ``DATABASE_URL`` points at Postgres (``postgres://`` or
    ``postgresql://``), shell out to ``pg_dump`` and gzip the output.
  * Otherwise falls back to copying the SQLite file at the URL path
    (sensible default for local Docker dev).
  * Writes ``db_<UTC>.sql.gz`` (or ``.sqlite.gz``) into ``BACKUP_DIR``
    (defaults to ``/app/backups`` for Railway, override for local).
  * Keeps the most recent ``KEEP_LAST`` (default 14) backups, deletes the rest.

Env vars:

  * ``DATABASE_URL``   - same URL the Flask app uses.
  * ``BACKUP_DIR``     - destination directory (default: /app/backups).
  * ``KEEP_LAST``      - rolling retention count (default: 14).

Exits non-zero on failure so Railway flags the job.
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


def _resolve_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite:///instance/db.sqlite")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def _ensure_backup_dir() -> Path:
    backup_dir = Path(os.environ.get("BACKUP_DIR", "/app/backups"))
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def _prune(backup_dir: Path, keep: int) -> None:
    files = sorted(
        [p for p in backup_dir.glob("db_*.gz") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[keep:]:
        try:
            stale.unlink()
            print(f"[backup] pruned {stale.name}")
        except OSError as exc:
            print(f"[backup] could not prune {stale}: {exc}", file=sys.stderr)


def _dump_postgres(url: str, dest: Path) -> None:
    if shutil.which("pg_dump") is None:
        raise RuntimeError(
            "pg_dump not installed in this image — install postgresql-client or "
            "switch to the sqlite fallback for local development."
        )
    print(f"[backup] dumping Postgres -> {dest.name}")
    with gzip.open(dest, "wb") as gz:
        proc = subprocess.run(
            ["pg_dump", "--no-owner", "--no-privileges", url],
            stdout=subprocess.PIPE,
            check=True,
        )
        gz.write(proc.stdout)


def _copy_sqlite(url: str, dest: Path) -> None:
    parsed = urlparse(url)
    # SQLAlchemy URLs are "sqlite:///relative/path" or "sqlite:////abs/path"
    sqlite_path = parsed.path
    if url.startswith("sqlite:///") and not sqlite_path.startswith("/"):
        sqlite_path = "/" + sqlite_path
    sqlite_path = sqlite_path.lstrip("/") if not os.path.isabs(sqlite_path) else sqlite_path
    src = Path(sqlite_path)
    if not src.exists():
        # try relative to repo's server/ directory
        repo_candidate = Path(__file__).resolve().parents[1] / sqlite_path
        if repo_candidate.exists():
            src = repo_candidate
        else:
            raise FileNotFoundError(f"SQLite database not found at {src}")
    print(f"[backup] copying SQLite {src} -> {dest.name}")
    with open(src, "rb") as fsrc, gzip.open(dest, "wb") as gz:
        shutil.copyfileobj(fsrc, gz)


def main() -> int:
    url = _resolve_db_url()
    backup_dir = _ensure_backup_dir()
    keep = int(os.environ.get("KEEP_LAST", "14"))
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    try:
        if url.startswith("postgresql://"):
            dest = backup_dir / f"db_{stamp}.sql.gz"
            _dump_postgres(url, dest)
        elif url.startswith("sqlite"):
            dest = backup_dir / f"db_{stamp}.sqlite.gz"
            _copy_sqlite(url, dest)
        else:
            print(f"[backup] unsupported DATABASE_URL scheme: {url}", file=sys.stderr)
            return 2
    except subprocess.CalledProcessError as exc:
        print(f"[backup] pg_dump failed (exit={exc.returncode})", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:  # noqa: BLE001
        print(f"[backup] failed: {exc}", file=sys.stderr)
        return 1

    print(f"[backup] wrote {dest} ({dest.stat().st_size} bytes)")
    _prune(backup_dir, keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
