# SAE-Based Interpretable Neural Steering for Recommendation Systems

**Author**: Bc. Václav Stibor  
**Supervisor**: Mgr. Ladislav Peška, Ph.D.  
**Institution**: Department of Software Engineering, Faculty of Mathematics and Physics, Charles University  
**Course**: Research Project (NPRG070)

## About

Modern neural recommender systems achieve strong predictive accuracy, but the concepts they rely on internally are hidden from the user. Ratings, likes and skips influence the model only indirectly — there is no direct way for a participant to inspect those concepts, let alone adjust them.

This project makes those concepts explicit. It pairs an ELSA recommender with a **Sparse Autoencoder (SAE)** that decomposes the model's internal representations into sparse, human-interpretable concepts — for a movie domain, features such as *"1980s sci-fi"*, *"strong female leads"*, or *"slow-paced cinematography"*. Those features are surfaced in the participant UI as first-class controls (sliders, toggles, free-text prompts, and example-based steering), so users can directly nudge the model's reasoning instead of treating it as a black box. The result is a transparent **Steering Loop**: see recommendations → adjust interpretable concepts → see the effect → iterate.

The repository delivers a study-ready realization of the idea on top of [EasyStudy](https://github.com/pdokoupil/EasyStudy):

- the **SAE Steering plugin** with sliders, toggles, text, and example-based modalities;
- a **researcher dashboard** with per-approach analytics, per-participant journeys, and attention-check tracking, exports, and additional metrics;
- end-to-end **deployment recipes** (local, Docker, Railway) with first-boot asset bootstrap from GitHub Releases.

Controlled user studies are currently running on a MovieLens-derived catalog, comparing baseline recommendations against steered variants under several configurations. 

A short specification rationale and full requirements list (FR / NFR ids referenced throughout the docs) live in [`proposal.tex`](proposal.tex).

## Demo


| Link                                                                  | Role          | Description                                            |
| --------------------------------------------------------------------- | ------------- | ------------------------------------------------------ |
| [Administration](https://sae4easystudy.up.railway.app/) | Administrator | Administrator panel for creating and managing studies *(use login: `admin@admin.cz`, password: `Admin1!`)*. |
| [Study](https://...)                                    | Participant   | Concrete study join link as a participant.                      |


## Repositories

| Repo | Role | Description |
| --- | --- | --- |
| **[SAE4EasyStudy](https://github.com/vaclavstibor/SAE4EasyStudy)** *(this)* | Online study runtime | EasyStudy-based Flask platform, the SAE Steering plugin, researcher dashboard, exports, and deployment scripts. |
| **[OfflineEasyStudy](TODO)** *(private)* | Offline pipeline | Dataset preprocessing, ELSA + Top‑K SAE training, LLM-based neuron labeling, post-hoc study-results analysis, and reproducibility notes. Kept private because it contains raw participant data. |
| **[EasyStudy](https://github.com/pdokoupil/EasyStudy)** | Upstream framework | Original user-study framework for recommender systems that this project extends through the plugin contract. |


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

## Documentation

- [Technical Documentation](docs/tech-docs.md) — architecture, plugin contract, database, audit pipeline, analytics, runtime, deployment, testing, and other technical details.
- [Design Decisions](docs/design-decisions.md) — rationale behind the main architectural and implementation decisions.
- [Formative Examples](docs/formative-examples.md) — worked recipes for extending the framework with new plugins, modalities, datasets, and analytics.
- [Equations](docs/equations.md) — mathematical descriptions of the steering and additional modalities.
- [Admin Manual](docs/admin-manual.md) — researcher workflow and operations guide.
- [User Manual](docs/user-manual.md) — participant-facing walkthrough.


## Development

For local setup, Docker, tests, runtime assets, environment variables and
the production checklist, see
[`docs/tech-docs.md` Section 9 — Runtime and Deployment](docs/tech-docs.md#9-runtime-and-deployment).

## References

- Peška, L. et al. Research on explainable and controllable recommender systems, GAČR 25-16785S (Charles University).
- Dokoupil, P. et al. [EasyStudy](https://github.com/pdokoupil/EasyStudy) — upstream user-study framework for recommender systems.
- [Project Proposal](proposal.tex) — original specification and motivation.

