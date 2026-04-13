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

## Railway (No Docker)

The repository now includes [railway.json](railway.json) for a no-Docker Railway deployment.

The repository also includes [nixpacks.toml](nixpacks.toml), which explicitly installs Python 3.9 for Railway Nixpacks builds.

On each push/deploy, Railway will:

1. install Python dependencies from `server/pip_requirements_railway.txt` (lean runtime set for faster Railway builds) and install CPU-only PyTorch wheel
2. run `server/bootstrap_datasets.py` only if dataset files are missing
3. run `server/plugins/sae_steering/bootstrap_model.py` only if runtime features are missing
4. remove `ml-latest.zip` after extraction to avoid wasting persistent disk space
5. start Gunicorn with `app:create_app()`

Default runtime SAE asset in Railway config is:

- `item_sae_features_www_TopKSAE_8192.pt.xz`

Required Railway setup:

1. Create a Railway project from this GitHub repo.
2. Add a Redis service in the same project.
3. Set these variables on the web service:
   - `APP_SECRET_KEY` (strong random string)
   - `SESSION_COOKIE_NAME` (any non-empty value)
   - `DATABASE_URL` (Railway Postgres URL or fallback SQLite path)
   - `REDIS_URL` (from Railway Redis service)
4. Optional overrides:
   - `SAE_MODEL_RELEASE_TAG` (default: `latest`)
   - `DATASET_RELEASE_TAG` (default: `latest`)
   - `SAE_RUNTIME_ASSET_NAME` (default: `item_sae_features_www_TopKSAE_8192.pt.xz`)
   - `SAE_RUNTIME_OUTPUT_PATH` (default: `plugins/sae_steering/data/item_sae_features_www_TopKSAE_8192.pt`)
   - `ML_DATASET_READY_FILE` (default: `static/datasets/ml-latest/ratings.csv`)
   - `GUNICORN_WORKERS`, `GUNICORN_TIMEOUT`, `GUNICORN_LOG_LEVEL`

Notes:

- If your release contains uncompressed runtime features, set `SAE_RUNTIME_ASSET_NAME=item_sae_features_www_TopKSAE_8192.pt`.
- First deploy can take longer because it downloads and extracts dataset/model assets.
- Railway runtime requirements intentionally skip TensorFlow-heavy dependencies to avoid build timeouts. If you need TF/TFRS-based fastcompare/vae features on Railway, install those packages in a separate deployment profile.
- For faster redeploys, add a Railway volume mounted to `/app/server/plugins/sae_steering/data` so runtime features persist across deployments.
- If you also want dataset persistence, mount a second Railway volume to `/app/server/static/datasets/ml-latest`.

In this repository, persistent paths are:

- SQLite data in `/app/server/instance`
- runtime cache in `/app/server/cache`
- downloaded SAE models in `/app/server/plugins/sae_steering/models`
- runtime features in `/app/server/plugins/sae_steering/data`

The `ml-latest.zip` asset should already contain:

- the MovieLens CSV files
- `genome-tags.csv` and `genome-scores.csv`
- `descriptions.json`
- `tmdb_data.json`
- the `img/` directory with poster JPGs

It is extracted into `/app/server/static/datasets/ml-latest`, so the resulting structure contains the dataset, metadata, and poster images expected by the app.

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

- [ ] Relabel sliders with human-understandable concepts
  - [ ] Several repetitive label names "Quirky", "Goofy", ...
  - [ ] Several with same Label name but slightly different descriptions >> probably merge
- [ ] "Na te uvodni strance potom co vysvetlime o cem je studie musi byt i ten informed consent (pripravim presny text) a to tlacitko by melo rict neco jako "I give my consent, lets continue""
