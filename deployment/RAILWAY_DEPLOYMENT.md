# Railway Deployment

Railway hosts this application as two services: one main web service that runs
the Flask + gunicorn process, and one cron service that snapshots the database
once per day. Both services share a persistent volume.

This document describes the deployment, but the Railway dashboard is the source
of truth — there is no `railway.json`, `railway-cron.json`, or `nixpacks.toml`
checked into the repository. All Railway-specific configuration lives in the
project UI.

## Main web service

- **Builder:** Dockerfile.
- **Dockerfile path:** `Dockerfile` (repository root).
- **Healthcheck:** `GET /healthz` (matches the route defined in
  [server/platform/app.py](../server/platform/app.py)).
- **Restart policy:** `ON_FAILURE` with at most 10 retries.
- **Start command:** none — the image's entrypoint
  ([server/docker-entrypoint.sh](../server/docker-entrypoint.sh)) takes over and
  starts gunicorn after validating assets and initialising the database.

Environment variables expected by the runtime are listed in
[app.env.example](app.env.example). At minimum, set `DATABASE_URL`,
`APP_SECRET_KEY`, and the GitHub Releases configuration if you want the
container to download its SAE assets on first boot
(`SAE_BOOTSTRAP_MODEL=1`, `SAE_MODEL_GITHUB_REPO`, `SAE_MODEL_RELEASE_TAG`,
optionally `GITHUB_TOKEN`).

## Cron service: daily database backup

A separate Railway service in the same project runs the backup script on a
schedule. Configure it as:

- **Builder:** Dockerfile (same `Dockerfile`).
- **Schedule (cron):** `0 3 * * *`.
- **Start command:** `python /app/server/scripts/backup_db.py`.
- **Volume:** mount the same persistent volume as the main service (see below).

The backup script writes `*.dump` files under `/app/backups/`.

## Persistent state

Five application directories need to survive restarts:

| Application path                          | Purpose                                      |
| ----------------------------------------- | -------------------------------------------- |
| `/app/server/instance/`                   | SQLite database (when `DATABASE_URL` is SQLite) |
| `/app/server/cache/`                      | Plugin cache / TensorFlow recommender caches |
| `/app/server/plugins/steering/data/`      | SAE runtime activations, label cache, semantic index |
| `/app/server/plugins/steering/models/`    | SAE checkpoint                               |
| `/app/backups/`                           | Daily DB dumps produced by the cron service  |

A previous version of this repository included a `server/railway-entrypoint.sh`
that symlinked all five paths into a single Railway volume mounted at `/data`.
That script has been removed; the equivalent pattern is now an operational
choice rather than something the image enforces. Two practical options:

1. Bake the static SAE assets into the image (or rely on the
   `SAE_BOOTSTRAP_MODEL=1` flow) and only persist `server/instance/` and
   `/app/backups/` on the Railway volume.
2. Persist all five directories on the volume by adding a small startup hook
   that symlinks them under the Railway mount point before
   `server/docker-entrypoint.sh` runs.

Whichever option you choose, the cron service must mount the same Railway
volume so the backup script can write into `/app/backups/`.

## SAE asset bootstrap

If the steering plugin assets are not baked into the image or pre-seeded on the
volume, set the following variables on the main service to download them from a
GitHub Release on first boot:

- `SAE_BOOTSTRAP_MODEL=1`
- `SAE_MODEL_GITHUB_REPO=<owner>/<repo>`
- `SAE_MODEL_RELEASE_TAG=<tag>` (or `latest`)
- `GITHUB_TOKEN=<token>` (only for private releases)

Optional overrides — `SAE_MODEL_ASSET_NAME`, `SAE_RUNTIME_ASSET_NAME`,
`SAE_LABEL_ASSET_NAME` — let you pin specific asset filenames if release tag
auto-detection is not desired. See
[server/plugins/steering/bootstrap_model.py](../server/plugins/steering/bootstrap_model.py)
for the full set of supported options.
