# Railway Deployment

Single service, one persistent volume. No Nginx, no separate database service —
SQLite lives on the volume and survives redeploys.

## 1. Create the Railway project

In Railway UI: **New Project → Deploy from GitHub repo** → select this repo.

## 2. Add a Volume

In the service settings: **Volumes → Add Volume**.

| Setting    | Value |
|------------|-------|
| Mount path | `/data` |

Everything that must survive a redeploy (SQLite DB, SAE model, dataset, cache)
is symlinked under `/data` by the entrypoint on startup.

## 3. Set environment variables

Copy from [app.env.example](app.env.example) and fill in the blanks.
Minimum required set for a first deploy:

| Variable | Value |
|----------|-------|
| `APP_SECRET_KEY` | any random string (e.g. `openssl rand -hex 32`) |
| `DATABASE_URL` | `sqlite:////data/instance/db.sqlite` |
| `DATA_ROOT` | `/data` |
| `DATASET_BOOTSTRAP` | `1` |
| `DATASET_GITHUB_REPO` | `vaclavstibor/SAE4EasyStudy` |
| `DATASET_RELEASE_TAG` | `v2.0` |
| `ML_LATEST_DATASET_ASSET` | `ml-32m-filtered.zip` |
| `SAE_BOOTSTRAP_MODEL` | `1` |
| `SAE_MODEL_GITHUB_REPO` | `vaclavstibor/SAE4EasyStudy` |
| `SAE_MODEL_RELEASE_TAG` | `v2.0` |
| `STUDY_AUTHOR_NAME` | your name |
| `STUDY_AUTHOR_CONTACT` | your e-mail |

> For private releases add `GITHUB_TOKEN=<PAT>`.

After the **first** successful deploy the assets are on the volume.
Subsequent deploys skip the download (entrypoint detects existing files)
and start in under a minute.

## 4. Builder settings

| Setting | Value |
|---------|-------|
| Builder | Dockerfile |
| Dockerfile path | `Dockerfile` (repository root) |
| Start command | *(leave empty — entrypoint handles it)* |

## 5. First-boot sequence (what the entrypoint does)

1. Symlinks `/data/{instance,cache,plugins/steering/models,plugins/steering/data,datasets}`
   into the expected `/app/server/…` paths.
2. Downloads `ml-32m-filtered.zip` from the GitHub Release and extracts it to
   `/data/datasets/` (`DATASET_BOOTSTRAP=1`).
3. Downloads SAE model checkpoint + runtime features to `/data/plugins/steering/models/`
   and `/data/plugins/steering/data/` (`SAE_BOOTSTRAP_MODEL=1`).
4. Validates all required files are present (exits with a clear error if not).
5. Runs `init_db.py` to create/update the SQLite schema.
6. Starts gunicorn.

## 6. Database backups

Backups are produced on demand by `/administration/db-backup` (admin
only). Each click runs `server/scripts/backup_db.py::create_backup_now()`
in-process, writes `db_<UTC>.{sql,sqlite}.gz`, and streams the new file
back to the browser.

Files land at `/data/backups/` on the persistent volume — the entrypoint
symlinks `/app/backups` → `${DATA_ROOT}/backups`, so the script's
default (`<repo_root>/backups` → `/app/backups`) resolves onto the
volume. Set `BACKUP_DIR=/some/other/path` to override; set `KEEP_LAST`
(default `14`) to change the rolling retention count.

If you also want unattended snapshots, schedule the same CLI externally:

```bash
python /app/server/scripts/backup_db.py
```
