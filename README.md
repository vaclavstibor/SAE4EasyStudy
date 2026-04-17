# Steering Neural Recommenders with Sparse Autoencoders

> Fork of [EasyStudy](https://github.com/pdokoupil/EasyStudy) for interactive SAE-based recommendation steering.

Collaborative filtering models learn powerful latent representations, but these embeddings suffer from **representation entanglement** - individual neurons are polysemantic, encoding multiple unrelated concepts simultaneously. This makes the models opaque and difficult to control.

**Sparse Autoencoders (SAE)** offer a solution. By projecting dense embeddings into a high-dimensional sparse space, SAEs learn disentangled features that often align with human-understandable concepts. Users can then directly manipulate these features to **steer** recommendations - boosting or suppressing specific aspects in real-time.

This repository demonstrates the steering capability through an interactive web application, part of the [Sparse4RESS](_) tutorial on sparse representations for recommendation explanations, steering, and segmentation.

## Key Features

![c4-level-2](images/c4-level-2.png)
![dual-model-data-flow](images/dual-model-data-flow.png)
![ui-mockup-sample](images/ui-mockup-sample.png)

The [sae_steering](./server/plugins/sae_steering/) plugin provides:

- **Slider steering** - continuous adjustment of individual SAE neurons
- **Text steering** - natural language queries converted to neuron activations via Sentence-BERT
- **A/B comparison** - side-by-side evaluation of different model configurations
- **Full interaction logging** - all steering actions captured for research analysis

## Quick Start

Requires Python 3.9 and Redis. The SAE bootstrap downloads both the `WWW TopKSAE-8192` checkpoint into `server/plugins/sae_steering/models/` and the precomputed runtime item features into `server/plugins/sae_steering/data/`.

```bash
# Setup
cd server
python3.9 -m venv .venv39 && source .venv39/bin/activate
pip install -r pip_requirements.txt

# Download the required SAE assets
python plugins/sae_steering/bootstrap_model.py

# Start Redis (in separate terminal)
brew install redis && brew services start redis

# Run
export FLASK_APP=app.py
flask --debug run
```

Open `http://localhost:5000`, create an SAE Steering study, and explore.

If the GitHub release asset uses a different filename, select it explicitly:

```bash
cd server
python plugins/sae_steering/bootstrap_model.py --asset-name model.pkl
```

## Docker Compose

For a CPU-only deployment, the repository now includes a `docker-compose.yml` that starts both the EasyStudy server and Redis. The container bootstraps the `WWW TopKSAE-8192` checkpoint, the precomputed SAE runtime features, and the required `ml-latest` dataset asset from GitHub Releases on first start.

```bash
# If clone complains about git-lfs, either install git-lfs
# or clone once with:
# GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/vaclavstibor/SAE4EasyStudy.git

git clone https://github.com/vaclavstibor/SAE4EasyStudy.git
cd SAE4EasyStudy
docker compose up --build
```

By default, the container looks for these release assets:

- `www_TopKSAE_8192.ckpt`
- `item_sae_features_www_TopKSAE_8192.pt`
- `ml-latest.zip`

If the runtime features file is too large to upload directly, you can upload a compressed asset such as `item_sae_features_www_TopKSAE_8192.pt.xz` instead. The bootstrap script now detects `.xz`, `.gz`, and single-file `.zip` runtime assets and extracts them automatically into `server/plugins/sae_steering/data/`.

Important:

- `git clone` downloads the source repository only
- `git lfs pull` downloads Git LFS objects tracked in git, but it does **not** download GitHub Release assets
- `www_TopKSAE_8192.ckpt`, `item_sae_features_www_TopKSAE_8192.pt`, and `ml-latest.zip` are fetched from the GitHub release at container startup by the bootstrap scripts

If you want to test the release downloads without Docker, run:

```bash
cd server
python bootstrap_datasets.py --tag v1.0
python plugins/sae_steering/bootstrap_model.py --tag v1.0
```

Useful environment overrides:

```bash
SAE_MODEL_RELEASE_TAG=v1.0 docker compose up --build
SAE_MODEL_GITHUB_REPO=vaclavstibor/SAE4EasyStudy docker compose up --build
DATASET_RELEASE_TAG=v1.0 docker compose up --build
```

## Railway deployment (Docker + persistent Postgres)

Despite the repository's history, the current Railway deployment is **Docker-based**:
[railway.json](railway.json) explicitly sets `"builder": "DOCKERFILE"` and points
at the project's [Dockerfile](Dockerfile). The optional [nixpacks.toml](nixpacks.toml)
is kept around for legacy nixpacks-based redeploys but is no longer used by the
production service.

On each push/deploy, the Docker image:

1. installs Python deps from `server/pip_requirements_railway.txt`
   (lean runtime set + CPU-only PyTorch + `psycopg2-binary` for Postgres)
2. installs `postgresql-client` (so the daily backup cron has `pg_dump`)
3. runs `server/bootstrap_datasets.py` if dataset files are missing
4. runs `server/plugins/sae_steering/bootstrap_model.py` if runtime features are missing
5. runs `flask db upgrade` to apply pending Alembic migrations on the live DB
6. starts Gunicorn with `app:create_app()`

### Required Railway setup

1. Create a Railway project from this GitHub repo.
2. **Add a Postgres add-on in the same project** (right pane → New → Database → Postgres).
   Railway auto-injects `DATABASE_URL` into the web service.
3. Add a Redis service in the same project (auto-injects `REDIS_URL`).
4. Set these variables on the web service:
   - `APP_SECRET_KEY` (strong random string)
   - `SESSION_COOKIE_NAME` (any non-empty value)
5. **Mount a single Railway volume at `/data`** (Railway allows exactly one
   volume mount per service). Recommended size: **5 GB** — enough for the
   SAE model, dataset, runtime features, and ~14 rolling DB dumps. The
   entrypoint script creates these subdirectories and symlinks them into
   the canonical application paths:
   - `/data/instance`   → `/app/server/instance` (fallback SQLite + Flask sessions)
   - `/data/cache`      → `/app/server/cache` (per-study cache)
   - `/data/sae_data`   → `/app/server/plugins/sae_steering/data` (runtime SAE features)
   - `/data/sae_models` → `/app/server/plugins/sae_steering/models` (SAE checkpoints)
   - `/data/backups`    → `/app/backups` (gzipped DB dumps)
   > The symlink bootstrap lives in [`server/railway-entrypoint.sh`](server/railway-entrypoint.sh)
   > and is idempotent. Override the mount root with `PERSIST_ROOT=/some/other`
   > if you need a different path.
6. Optional overrides:
   - `PERSIST_ROOT` (default: `/data`) — where the entrypoint expects the single volume mount
   - `SAE_MODEL_RELEASE_TAG` (default: `latest`)
   - `DATASET_RELEASE_TAG` (default: `latest`)
   - `BACKUP_DIR` (default: `/app/backups`), `KEEP_LAST` (default: `14`)
   - `SKIP_DB_UPGRADE=1` to skip migrations on a specific deploy
   - `GUNICORN_WORKERS`, `GUNICORN_TIMEOUT`, `GUNICORN_LOG_LEVEL`

> SQLAlchemy 1.4+ requires `postgresql://`, but Railway hands out the legacy
> `postgres://` URL prefix. `server/app.py` and `server/scripts/backup_db.py`
> both rewrite the prefix automatically — no manual change needed.

### Database backups

Railway Postgres already has its own persistent volume, so data survives
redeploys and restarts by itself. Dumps are there to protect against the
things the Postgres volume can't: accidental deletes, schema migrations
gone wrong, the whole Railway project being archived, or you simply wanting
an offline copy for analysis.

The repo ships [server/scripts/backup_db.py](server/scripts/backup_db.py),
which runs `pg_dump | gzip` (or copies the SQLite file when there's no
Postgres) into `BACKUP_DIR`, keeping the last 14 dumps.

**On-demand (recommended):**

- Admins click **Administration → Download DB backup**
  (`GET /administration/db-backup`). The route streams the latest dump
  from `/app/backups` and triggers a fresh `pg_dump` on the fly if no
  recent dump is present.
- From your laptop: `railway run python server/scripts/backup_db.py`
  to force a dump inside the running web service, then download it.

**Automated daily cron (optional, later):**
Each Railway service has its own volume, so a separate cron service would
not share `/app/backups` with the web service. If you want automation,
the cleanest options are either (a) adding an in-process scheduler
(`APScheduler`) inside the Flask app with `GUNICORN_WORKERS=1`, or
(b) running a cron service that uploads dumps to an S3-compatible bucket.
Neither is wired up yet — the on-demand path above is enough for a
short-running Prolific study. The stub [railway-cron.json](railway-cron.json)
is left as a reference.

### Pulling production data into local for analysis

```bash
# Force a fresh dump inside the running web service and stream it to your laptop:
railway run --service SAE4EasyStudy -- \
  python /app/server/scripts/backup_db.py && \
  railway run --service SAE4EasyStudy -- \
  sh -c 'cat /app/backups/$(ls -1t /app/backups | head -1)' > local_dump.sql.gz

# Then restore locally against a throwaway Postgres:
gunzip -c local_dump.sql.gz | psql postgresql://postgres:pw@localhost:5432/easystudy
```

### Notes

- First deploy can take longer because it downloads and extracts
  dataset/model assets.
- Railway runtime requirements intentionally skip TensorFlow-heavy
  dependencies to avoid build timeouts. If you need TF/TFRS-based
  fastcompare/vae features on Railway, install those packages in a
  separate deployment profile.

In this repository, all persistent paths live under the single `/data`
volume mount on Railway, symlinked by
[`server/railway-entrypoint.sh`](server/railway-entrypoint.sh):

- `/data/instance`   → fallback SQLite + Flask session files
- `/data/cache`      → per-study runtime cache
- `/data/sae_models` → downloaded SAE checkpoints
- `/data/sae_data`   → runtime SAE features / LLM labels
- `/data/backups`    → gzipped DB dumps

The `ml-latest.zip` asset should already contain:

- the MovieLens CSV files
- `genome-tags.csv` and `genome-scores.csv`
- `descriptions.json`
- `tmdb_data.json`
- the `img/` directory with poster JPGs

It is extracted into `/app/server/static/datasets/ml-latest`, so the resulting structure contains the dataset, metadata, and poster images expected by the app.

## Prolific integration

This section walks through the **end-to-end flow** of collecting paid
participants on [Prolific](https://app.prolific.com/) and paying them out.
Everything below assumes the app is already running (locally via Docker or
deployed on Railway).

```mermaid
sequenceDiagram
  autonumber
  participant Admin
  participant Prolific
  participant P as Participant
  participant App as EasyStudy
  participant DB as Postgres

  Admin->>Prolific: Create study draft → get Completion Code
  Admin->>App: Create SAE study, paste Completion Code
  App-->>Admin: Join URL /sae_steering/join/<guid>
  Admin->>Prolific: Submit study with the Join URL + {{%PROLIFIC_PID%}} tokens
  Prolific->>P: Participant opens study link
  P->>App: GET /sae_steering/join/<guid>?PROLIFIC_PID=..&STUDY_ID=..&SESSION_ID=..
  App->>DB: Persist PID / STUDY_ID / SESSION_ID in Participation.extra_data
  P->>App: Completes study
  App->>P: /utils/finish page (15 s auto-redirect + manual button)
  P->>Prolific: https://app.prolific.com/submissions/complete?cc=<code>
  Prolific-->>Admin: Submission marked complete → payment approval
```

### 1. Create the study on Prolific (in *Draft* state)

1. [Prolific dashboard](https://app.prolific.com/) → **New Study**.
2. Fill in title, description, eligibility filters, reward, and number of places.
3. In **Study URL** leave the default for now — we'll fill it in step 3.
4. Under **How will you record Prolific IDs?** choose
   **"I'll use URL parameters"** and make sure the three default tokens are
   present in the URL placeholder:
   - `{{%PROLIFIC_PID%}}`
   - `{{%STUDY_ID%}}`
   - `{{%SESSION_ID%}}`
5. Under **Completion codes** → **Add** → generate a code (e.g. `C1A2B3C4`)
   and pick **"Completed study"** as the action. This is the string
   participants will send back to Prolific when they finish. Save the code —
   you will paste it into EasyStudy in step 2.

> Leave the study in *Draft* until EasyStudy is fully configured.
> You cannot change the completion code once the study is published.

### 2. Create the matching study in EasyStudy

1. Open `https://<your-app>/administration` and click **Create new study**.
2. Pick **SAE Steering** as the plugin.
3. Fill in the usual study metadata (language, models, item counts,
   questionnaire variants, etc.).
4. **Paste the Prolific completion code** from step 1 into the
   *Prolific Completion Code* field (added in
   [`sae_steering_create.html`](server/plugins/sae_steering/templates/sae_steering_create.html)).
5. Submit. After initialization finishes, the admin page shows the study
   **Join URL** (`/sae_steering/join/<guid>`). Copy it.

### 3. Point Prolific at the EasyStudy join URL

Back in Prolific:

```
https://<your-app>/sae_steering/join/<guid>?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```

- Replace `<your-app>` and `<guid>` with the real values.
- Prolific expands the `{{%…%}}` tokens per participant at click time.
- Paste this URL into Prolific's **Study URL** field.
- **Publish** the study.

### 4. What participants see

1. Click the Prolific link → `GET /sae_steering/join/<guid>?PROLIFIC_PID=…`.
   The three IDs are placed into the Flask session and mirrored into
   `Participation.extra_data` on the first click, so they survive even if
   the participant abandons the study mid-way.
2. Participant goes through the study (consent, demographics,
   elicitation, arm A, questionnaire, arm B, questionnaire, final
   questionnaire, attention checks).
3. On the last page, `utils/finish` renders the completion screen:
   - **Both** a manual **"Finish"** button that redirects immediately, and
   - a **15-second auto-redirect** to
     `https://app.prolific.com/submissions/complete?cc=<your_code>`
     (see [`finish.html`](server/plugins/utils/templates/finish.html) lines
     215–222).
4. Prolific receives the completion ping and the submission moves to
   *Awaiting review* in your dashboard.

### 5. Approve submissions and pay

1. Prolific dashboard → **Submissions** tab → review each one.
2. Cross-reference against **EasyStudy Admin → Results → Participants tab**
   (the new dashboard introduced in this fork). For every row you can read:
   - `PROLIFIC_PID`, `STUDY_ID`, `SESSION_ID`
   - Join / finish timestamps, duration
   - Attention-check pass counts (phase + final + overall)
   - Status: `completed`, `in-progress`, `abandoned`
   - A one-click-copy completion URL
3. Approve / reject in Prolific. Prolific charges your balance on approval.

### 6. Where the Prolific data lives

| Artifact | Location | Button |
|----------|----------|--------|
| Raw per-interaction log incl. `prolific` block | `GET /sae_steering/export-raw/<guid>` | Admin list → **Download raw events (JSON)** |
| Raw + noisy mouse/viewport events | `GET /sae_steering/export-raw/<guid>?include_noise=1` | Results → **Download raw events + noise (JSON)** |
| Aggregated analytics + `participants_table` | `GET /sae_steering/fetch-results/<guid>` | Results → **Download analytics (JSON)** |
| Per-participant reconstructed journey | `GET /sae_steering/journey/<participation_id>` | Results → **Journey tab** picker |
| Latest DB dump | `GET /administration/db-backup` | Administration → **Download DB backup** |

All five sources include the Prolific identity block so payments can be
reconciled regardless of which export you grab.

### 7. Troubleshooting

- **"Please contact the study admin" screen at finish** — you forgot to
  paste the completion code in step 2. The participant *did* complete
  the study (data is saved), but cannot auto-return. Email them the
  code or approve manually in Prolific using their PID.
- **No `prolific` block in exports** — the participant opened the study
  URL **without** the Prolific query params (e.g. followed a raw
  administrator link). Data is still recorded, but identity linking is
  lost. Always distribute the URL *through* Prolific.
- **Attention-check failure handling** — attention results are captured
  per-phase and show up in the Participants tab.
  EasyStudy does not auto-reject; use your judgement in Prolific.
- **Participant abandoned mid-study** — `Participation.time_finished`
  stays `NULL`. The Participants tab shows them as *in-progress* /
  *abandoned*, with whatever interactions they left behind.

### 8. Testing the flow locally

```bash
# from the EasyStudy root:
open "http://localhost:5000/sae_steering/join/<your-guid>?PROLIFIC_PID=pid_test_abc&STUDY_ID=study_test&SESSION_ID=session_test"
```

Walk through the study once. Confirm `Download raw events (JSON)` contains:

```json
"prolific": {
  "pid": "pid_test_abc",
  "study_id": "study_test",
  "session_id": "session_test",
  "completion_code": "C1A2B3C4",
  "completion_url": "https://app.prolific.com/submissions/complete?cc=C1A2B3C4"
}
```

If yes, wire the real study URL into Prolific and publish.

## Release Checklist

Upload these GitHub Release assets for a fresh machine to behave the same way as the development machine:

- `www_TopKSAE_8192.ckpt`
- `item_sae_features_www_TopKSAE_8192.pt`
- `ml-latest.zip`

The `item_sae_features_www_TopKSAE_8192.pt` file lives in `server/plugins/sae_steering/data/`. If you need to compress it for release upload, run:

```bash
cd server/plugins/sae_steering/data
xz -T0 -9 -k item_sae_features_www_TopKSAE_8192.pt
```

## EasyStudy Framework

Built on [EasyStudy](https://github.com/pdokoupil/EasyStudy) by [Patrik Dokoupil](mailto:patrik.dokoupil@matfyz.cuni.cz) and [Ladislav Peska](mailto:ladislav.peska@matfyz.cuni.cz). For deployment details, dataset setup, and Docker configuration, refer to the original [documentation](https://github.com/pdokoupil/EasyStudy#readme).

## Hard TODOs


---

- Vyhledávání zaznamenat do výstupu (jak?), jestli hledal a vyhledal, hlavní rozlišit, co patří k jakému appraoch a z base, abychom potom tuto flow byly schopni bez problemu vytahnout z databaze.

---
* https://dl.acm.org/doi/epdf/10.1145/3604915.3608848
* "Link ..." 
---

### Backbone recommender issue

Potom recommender.
Porad mi prijde ze funguje trochu divne - ted zrovna jsem zkusil v preference elicitation vybirat relativne nove blockbuster fantasy a scifi (Hobbit, Chappie, District 9,...) - vysledky jsou v podstate random, poslal bych ti screenshot, ale zrovna vse bez posteru a rozhodne nic co by se dalo nazvat blockbuster a skoro nic co by sedelo na scifi nebo fantasy.
- Jak moc (ne)realne je natrenovat vlastni base recommender + SAE nad tim?
- S Patrikem jsme pro posledni studie volili spis trochu mirnejsi parametry redukce, viz. https://dl.acm.org/doi/epdf/10.1145/3699682.3728335 We used MovieLens TagGenome 2021 [10], which provides a fairly up-to-date selection of movies (release date up till 2021) ... To keepthe computational requirements feasible, the dataset was reducedas follows. First, we filtered out all movies without a cover image or plot description. Then, we removed old users (who did not interact with movies released after 2015), old movies (released before 1985),and movies without recent interactions (with < 10 remaining ratings). As a result, the dataset size reduced to |U| = 18𝐾, |I| = 16𝐾,|R|>0 = 5.1𝑀, where U, I, and R ∈ R| U | × | I | correspond to set of all users, all items, and their ratings, while |R|>0 denotes numberof non-zero entries of R. The R denotes the set of all real numbers.

---
Dobře. Teď zkontroluj, jestli se to správně zachytává do results, chceme toho co nejvíce zachytávat. 

Udělej test na resultsdisplay, jestli ti tam něco chybí, tak to klidně doplň. Důležité je, abychom zachytávali rozumně uživatelské akce, výstupy atd. Hlavně i odpovědi na dotazníky.

---

Check the recommended movies and select those you would currently consider to watch by simply clicking on them. If you click on one accidentally, another click de-selects the movie. Kdykoliv v prubehu iterace, můžete své rozhodnutí změnit. Kliknutím na tlačítko především potvrzujete, že jste na tento krok nezapomněli.

---

Nenapadlo vas nějakou init otazku, abych dokazali změřit škálu, že třeba někdo dává hodně přehnany feedback, někdo ne? Jak moc je tohle červená atd