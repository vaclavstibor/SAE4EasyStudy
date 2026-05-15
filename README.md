# SAE-Based Interpretable Neural Steering for Recommendation Systems

**Author**: Bc. Václav Stibor  
**Supervisor**: Mgr. Ladislav Peška, Ph.D.  
**Institution**: Department of Software Engineering, Faculty of Mathematics and Physics, Charles University  
**Course**: Research Project (NPRG070)

## About

Neural recommender systems achieve strong predictive accuracy, yet the concepts they rely on internally are hidden from the user. Standard interaction signals (ratings, likes, preferences) only influence the model indirectly, so there is no direct way for a participant to inspect those concepts, let alone adjust them.

This project makes those concepts explicit. It pairs an ELSA recommender with a **Sparse Autoencoder (SAE)** that decomposes the model's internal representations into sparse, human-interpretable concepts; for a movie domain, examples include *"1980s sci-fi"*, *"strong female leads"*, or *"slow-paced cinematography"*. Those features are surfaced in the participant UI as first-class controls (sliders, toggles, free-text prompts, and example-based steering), so users can directly nudge the model's reasoning instead of treating it as a black box.

The result is a study-ready **Steering Loop** on top of [EasyStudy](https://github.com/pdokoupil/EasyStudy). The repository delivers:

- the **SAE Steering plugin** with multiple steering modalities and multi-approach comparisons;
- a **researcher dashboard** with per-approach analytics, per-participant journeys, and attention-check tracking, exports, and additional metrics;
- end-to-end **deployment recipes** (local, Docker, Railway) with first-boot asset bootstrap from GitHub Releases.

A full specification rationale and requirements list (FR / NFR ids referenced throughout the docs) lives in [`specification.pdf`](proposal/specification.pdf).


## Demo


| Link                                                                  | Role          | Description                                            |
| --------------------------------------------------------------------- | ------------- | ------------------------------------------------------ |
| [Administration](https://sae4easystudy.up.railway.app/) | Administrator | Administrator panel for creating and managing studies (sign in with the credentials configured on that deployment). |
| [Study](http://sae4easystudy.up.railway.app/sae_steering/join?guid=KEQ-usXvd56_YvuAG8HbRR03Q8xApwbS)                                    | Participant   | Slider and non steering study join link. |


## Repositories

| Repo | Description |
| --- | --- |
| **[SAE4EasyStudy](https://github.com/vaclavstibor/SAE4EasyStudy)** *(this)* | EasyStudy-based Flask platform, the SAE Steering plugin, researcher dashboard, exports, and deployment scripts. |
| **[OfflineEasyStudy](https://github.com/vaclavstibor/OfflineEasyStudy)** *(private)* | Dataset preprocessing, ELSA + Top‑K SAE training, LLM-based neuron labeling, post-hoc study-results analysis, and reproducibility notes. Kept private because it contains raw participant data. (*Do not hesitate to ask for access if you are interested in the details*) |
| **[EasyStudy](https://github.com/pdokoupil/EasyStudy)** | Original user-study framework for recommender systems that this project extends through the plugin contract. |


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
    steering/           # SAE Steering plugin — research contribution (modalities, audit, dashboard, export)
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
- [Contribution Summary](docs/contribution-summary.md) — how the project's research contribution aligns with the proposal's requirements, evaluation, and risks.


## Evaluation

The repository's evaluation evidence is described in [`evaluation/README.md`](evaluation/README.md). In short:

- A **200-participant Prolific study** (no-steering vs slider-steering) was conducted for a paper currently in review. The raw data, analysis, and manuscript live in the private [OfflineEasyStudy](https://github.com/vaclavstibor/OfflineEasyStudy) repository; access can be requested.
- A **supplementary in-house evaluation** with few anonymized participants across the three steering modalities (toggle, slider, text) on default settings is committed under [`evaluation/data/`](evaluation/data/) as a filled-in Likert questionnaire. 

## Development

For local setup, Docker, tests, runtime assets, environment variables and
the production checklist, see
[`docs/tech-docs.md` Section 9 — Runtime and Deployment](docs/tech-docs.md#9-runtime-and-deployment).

## References

- Dokoupil, P. et al. [EasyStudy](https://github.com/pdokoupil/EasyStudy) — upstream user-study framework for recommender systems.
- Czech Science Foundation (GAČR). Project No. 25‑16785S: “Empowering Multi-Objective Recommender Systems with Large Language Models” (PI: Mgr. Ladislav Peška, Ph.D., Charles University)

