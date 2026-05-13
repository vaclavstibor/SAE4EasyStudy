# Root Scripts

Small repository-level entrypoints live here so the project can be driven from the
repository root instead of from inside `server/`.

- `run-dev.sh` starts the application with the canonical `server.platform.app:create_app()`
  entrypoint.
- `init-db.sh` runs the idempotent database bootstrap.
- `test.sh` runs the canonical repository test suite.
- `lint.sh` runs Ruff on the maintained source and test trees.

All runtime-facing scripts use the project-local `server/.venv39` interpreter.
