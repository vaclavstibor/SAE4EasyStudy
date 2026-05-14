# SAE-Based Interpretable Neural Steering for Recommendation Systems

**Author**: Bc. Václav Stibor  
**Supervisor**: Mgr. Ladislav Peška, Ph.D.  
**Institution**: Department of Software Engineering, Faculty of Mathematics and Physics, Charles University  
**Course**: Research Project (NPRG070)

## About

Neural recommender systems achieve strong predictive accuracy, but their internal reasoning stays opaque and users have little control over *how* recommendations are produced. This project introduces a **steerable recommender** that uses **Sparse Autoencoder (SAE)** features as user-facing controls, letting participants directly adjust interpretable concepts instead of treating the model as a black box.

The repository delivers a study-ready implementation of this idea: an EasyStudy-based platform, the SAE Steering plugin with multiple steering modalities (sliders, toggles, text, examples), a typed audit pipeline, a researcher dashboard, and CSV export. It supports controlled user studies on a MovieLens-derived catalog comparing baseline recommendations against steered variants.

## Study Links


| Link                                 | Description                                            |
| ------------------------------------ | ------------------------------------------------------ |
| [EasyStudy Admin Panel](https://...) | Administrator panel for creating and managing studies. |
| [Concrete Study](https://...)        | Study join link as a participant.                      |


## Repositories


| Repo                                                                      | Description                                                                                           |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **[SAE4EasyStudy](https://github.com/vaclavstibor/SAE4EasyStudy)** (this) | Research framework: platform, SAE Steering plugin, runtime, and docs.                                 |
| **[EasyStudy](https://github.com/pdokoupil/EasyStudy)**                   | Upstream framework for recommender-system user studies that this project extends.                     |
| **[OfflineEasyStudy](https://...)**                                       | Offline data preprocessing, train, neuron labeling, studies results analysis, reproducibility details |


## Documentation

- [Technical Documentation](docs/tech-docs.md) — architecture, plugin contract, database, audit pipeline, analytics, runtime, deployment, testing, and other technical details.
- [Design Decisions](docs/design-decisions.md) — rationale behind the main architectural and implementation decisions.
- [Formative Examples](docs/formative-examples.md) — worked recipes for extending the framework with new plugins, modalities, datasets, and analytics.
- [Equations](docs/equations.md) — math behind steering and reranking.
- [Admin Manual](docs/admin-manual.md) — researcher workflow and operations guide.
- [User Manual](docs/user-manual.md) — participant-facing walkthrough.

## Project Structure

```text
Dockerfile              # container build
docker-compose.yml      # local orchestration
pyproject.toml          # Python tooling
justfile                # task runner
docs/                   # project documentation
docker/                 # Docker config and examples
deployment/             # deployment notes and env examples
scripts/                # dev, test, lint, and DB helpers
tests/                  # canonical test suite
server/
  platform/             # framework kernel (Flask, persistence, participant flow)
  plugins/
    steering/           # SAE Steering plugin (thesis contribution)
    fastcompare/        # upstream EasyStudy plugin (kept verbatim)
    empty_template/     # minimal plugin skeleton
```

## Development

Run with Docker:

```bash
docker compose up --build
```

Or locally (Python 3.9 baseline):

```bash
python3.9 -m venv server/.venv39
./server/.venv39/bin/python -m pip install -r server/pip_requirements.txt pytest ruff
./scripts/init-db.sh
./scripts/run-dev.sh
```

Then open `http://localhost:5000`.

Tests and lint:

```bash
./scripts/test.sh       # or: just test
./scripts/lint.sh       # or: just lint
```

## Runtime Assets

The application expects two groups of assets to exist before the steering
blueprint can serve recommendations:


| Location                                  | Files                                                                                                                              |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `server/static/datasets/ml-32m-filtered/` | `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`, `plots.csv`; optional `img/*.jpg`                                            |
| `server/plugins/steering/models/`         | `TopKSAE-1024.ckpt` (or `.pt`)                                                                                                     |
| `server/plugins/steering/data/`           | `item_embeddings.pt`, `item_sae_features_TopKSAE-1024.pt`, `llm_labels_TopKSAE-1024_llm.json`, `semantic_merged_TopKSAE-1024.json` |


The dataset directory must always be supplied manually. For the SAE plugin
assets (checkpoint + the four data files) there are two supported flows:

- **GitHub Releases bootstrap.** Set `SAE_BOOTSTRAP_MODEL=1` and
`SAE_MODEL_GITHUB_REPO=<owner>/<repo>`, plus `SAE_MODEL_RELEASE_TAG` (defaults
to `latest`) and `GITHUB_TOKEN` for private releases. The container entrypoint
invokes `[server/plugins/steering/bootstrap_model.py](server/plugins/steering/bootstrap_model.py)`
on startup and downloads every asset into the correct location. Optional
overrides — `SAE_MODEL_ASSET_NAME`, `SAE_RUNTIME_ASSET_NAME`,
`SAE_LABEL_ASSET_NAME` — let you pin specific asset filenames.
- **Manual placement.** Place the files yourself under the paths in the table
above. The entrypoint validates their presence on startup and refuses to
launch if any are missing.

See [server/plugins/steering/data/README.md](server/plugins/steering/data/README.md)
for the per-file inventory.

## References

- Peška, L. et al. Research on explainable and controllable recommender systems, GAČR 25-16785S (Charles University).
- Dokoupil, P. et al. [EasyStudy](https://github.com/pdokoupil/EasyStudy) — upstream user-study framework for recommender systems.
- [Project Proposal](proposal.tex) — original specification and motivation.

