# Study Framework

Plugin-first modular monolith for EasyStudy-style experiments.

The repository now has one canonical runtime story:

- run from the repository root,
- load the app through `server.platform.app:create_app()`,
- treat `server/platform/` as the framework kernel,
- treat `server/plugins/steering/` as the canonical SAE study plugin.

## Project structure

```text

├── Dockerfile
├── docker-compose.yml
├── TEMP_REFACTOR_NOTES.md
├── pyproject.toml
├── justfile
├── docker/
├── deployment/
├── docs/
├── scripts/
├── tests/
└── server/
    ├── docker-entrypoint.sh
    ├── platform/
    │   ├── participant_flow/
    │   └── web/
    └── plugins/
        ├── steering/
        ├── fastcompare/
        └── empty_template/
```

## Manual local asset placement

| Location | Files |
|----------|--------|
| `server/static/datasets/ml-32m-filtered/` | `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`, `plots.csv`; optional `img/*.jpg` |
| `server/plugins/steering/models/` | `TopKSAE-1024.ckpt` or `TopKSAE-1024.pt` |
| `server/plugins/steering/data/` | `item_embeddings.pt`, `item_sae_features_TopKSAE-1024.pt`, `llm_labels_TopKSAE-1024_llm.json`, `semantic_merged_TopKSAE-1024.json` |

Where to obtain assets (bootstrap, legacy repo paths): see `server/plugins/steering/data/README.md`.

## Run with Docker

From repository root:

```bash
docker compose up --build
```

The canonical container path uses:

- `Dockerfile`
- `docker-compose.yml`
- `server/docker-entrypoint.sh`
- `server/scripts/init_db.py`

Defaults assume manual assets (`DATASET_BOOTSTRAP=0`, `SAE_BOOTSTRAP_MODEL=0`).
See `docker/compose.env.example` and `deployment/app.env.example` for the small
set of supported overrides.

The persistent cache volume must target `server/cache`, because dataset/model
cache paths are resolved relative to the `server/` package root.

## Run locally without Docker

Python `3.9` is the supported baseline.

```bash
python3.9 -m venv server/.venv39
./server/.venv39/bin/python -m pip install -r server/pip_requirements.txt pytest ruff
./scripts/init-db.sh
./scripts/run-dev.sh
```

Then open `http://localhost:5000`.

## Tests and lint

The canonical test collection path is the root `tests/` tree.

```bash
./scripts/test.sh
./scripts/lint.sh
```

You can also use:

```bash
just test
just lint
just run
```

## Database

The schema is defined by SQLAlchemy models only; there is no migration system.

- `server/platform/persistence/base_models.py` and each plugin's
  `persistence/models.py` are the single source of truth for the schema.
- `server/platform/app.py` runs `db.create_all()` on every boot.
- `./scripts/init-db.sh` -> `server/scripts/init_db.py` is a thin idempotent wrapper.
- `./scripts/reset-db.sh` -> `server/scripts/reset_db.py --yes` drops and recreates the schema
  (use whenever a model is reshaped in dev).

## Temporary Refactor Notes

See `TEMP_REFACTOR_NOTES.md` for the structured summary of the platform/plugin refactor,
shared flow restoration, and removal of legacy compatibility layers.

## Documentation

Full documentation lives under `docs/`:

- [`docs/tech-docs.md`](docs/tech-docs.md) - canonical technical reference (architecture, plugin contract, database, audit pipeline, analytics, runtime, deployment, testing).
- [`docs/design-decisions.md`](docs/design-decisions.md) - the *why* document: binding architectural decisions with rationale.
- [`docs/formative-examples.md`](docs/formative-examples.md) - worked recipes with code snippets (add a plugin, modality, dataset, audit table, reranking strategy).
- [`docs/equations.md`](docs/equations.md) - math reference for scoring functions.
- [`docs/admin-manual.md`](docs/admin-manual.md) - researcher manual.
- [`docs/user-manual.md`](docs/user-manual.md) - participant manual.

## Plugin notes

- New study work should target `server/plugins/steering/`.
- `server/plugins/fastcompare/` and `server/plugins/empty_template/` are
  EasyStudy-native plugin-first skeletons preserved verbatim for upstream parity.
- Shared participant-flow pages/assets live in `server/platform/participant_flow/`.
- Platform-owned admin/auth templates live in `server/platform/web/templates/`.
- EasyStudy cross-plugin primitives stay in `server/plugins/utils/`. Avoid adding
  *new* shared logic there; reach for `server/platform/shared/` for non-EasyStudy
  helpers.
