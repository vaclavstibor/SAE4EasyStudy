# SAE-Based Interpretable Neural Steering for Recommendation Systems

**Author**: Bc. Václav Stibor  
**Supervisor**: Mgr. Ladislav Peška, Ph.D.  
**Institution**: Department of Software Engineering, Faculty of Mathematics and Physics, Charles University  
**Course**: Research Project (NPRG070)

## About

Neural recommender systems achieve strong predictive accuracy, but their internal reasoning stays opaque and users have little control over *how* recommendations are produced. This project introduces a **steerable recommender** that uses **Sparse Autoencoder (SAE)** features as user-facing controls, letting participants directly adjust interpretable concepts instead of treating the model as a black box.

The repository delivers a study-ready implementation of this idea: an EasyStudy-based platform, the SAE Steering plugin with multiple steering modalities (sliders, toggles, text, examples), a typed audit pipeline, a researcher dashboard, and CSV export. It supports controlled user studies on a MovieLens-derived catalog comparing baseline recommendations against steered variants.

## Demo


| Link                                                                  | Role          | Description                                            |
| --------------------------------------------------------------------- | ------------- | ------------------------------------------------------ |
| [Administration](https://sae4easystudy.up.railway.app/) | Administrator | Administrator panel for creating and managing studies *(use login: `admin@admin.cz`, password: `Admin1!`)*. |
| [Study](https://...)                                    | Participant   | Concrete study join link as a participant.                      |


## Repositories

| Repo | Role | Description |
| --- | --- | --- |
| **[SAE4EasyStudy](https://github.com/vaclavstibor/SAE4EasyStudy)** *(this)* | Online study runtime | EasyStudy-based Flask platform, the SAE Steering plugin (sliders, toggles, text, examples, reranking), researcher dashboard, CSV export, and deployment scripts. |
| **[EasyStudy](https://github.com/pdokoupil/EasyStudy)** | Upstream framework | Original user-study framework for recommender systems that this project extends through the plugin contract. |
| **[OfflineEasyStudy](TODO)** *(private)* | Offline pipeline | Dataset preprocessing, ELSA + Top‑K SAE training, LLM-based neuron labeling, post-hoc study-results analysis, and reproducibility notes. Kept private because it contains raw participant data. |


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
docker/                 # Docker config and env examples
deployment/             # Railway deployment notes and env examples
scripts/                # dev, test, lint, and DB helpers
tests/                  # canonical test suite (platform + plugins)
server/
  platform/             # framework kernel (Flask app, persistence, participant flow, runtime, admin)
  plugins/
    steering/           # SAE Steering plugin — thesis contribution (modalities, audit, dashboard, export)
    fastcompare/        # upstream EasyStudy plugin — kept as a runnable plugin-contract reference
    layoutshuffling/    # upstream EasyStudy plugin — kept with a minimal demo flow
    empty_template/     # minimal plugin skeleton (developer scaffold, hidden from admin)
    vae/                # VAE algorithm wrappers consumed by fastcompare (hidden from admin)
    utils/              # shared recommender utilities (data loading, mandate allocation, normalization)
```

## Development

For local setup, Docker, tests, runtime assets, environment variables and
the production checklist, see
[`docs/tech-docs.md` §9 — Runtime and Deployment](docs/tech-docs.md#9-runtime-and-deployment).

## References

- Peška, L. et al. Research on explainable and controllable recommender systems, GAČR 25-16785S (Charles University).
- Dokoupil, P. et al. [EasyStudy](https://github.com/pdokoupil/EasyStudy) — upstream user-study framework for recommender systems.
- [Project Proposal](proposal.tex) — original specification and motivation.

