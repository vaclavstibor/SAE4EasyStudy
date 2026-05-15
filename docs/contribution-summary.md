# Contribution Summary

This repository is the deliverable described in `proposal.tex`: an EasyStudy-based study framework extended with an SAE Steering plugin, the surrounding admin/participant UI, and a reproducible audit/export pipeline suitable for research studies.

## Contents

1. [What this repository adds to upstream EasyStudy](#what-this-repository-adds-to-upstream-easystudy)  
   Core deliverables and major additions.
2. [Evaluation alignment with `proposal.tex`](#evaluation-alignment-with-proposaltetex-section-evaluation)  
   How the contribution aligns with the proposal's evaluation section.
3. [Risks from `proposal.tex`](#risks-from-proposaltetex-section-risks)  
   How the contribution addresses the proposal's risks.

## What this repository adds to upstream EasyStudy

- **A new study plugin**: `server/plugins/steering/` implements the SAE Steering study flow as a first-class EasyStudy plugin (create → initialize → join → dispose → results).
- **A typed audit pipeline**: steering studies write domain-specific typed tables (`Sae*`) plus a thin `SaeSteeringEvent` envelope for timeline ordering. This enables stable analytics and exports without JSON parsing at read time.
- **Multiple steering modalities (FR-06)**: sliders, toggles, example-based steering, text steering (with explicit composition modes), and a dedicated reset endpoint.
- **Iteration updates are end-to-end (FR-10)**: steering actions propagate through the iteration loop and recompute recommendations. Three reranking strategies (`feature-conditioned`, `latent-perturbation`, `constrained-subset`) are implemented, selectable in the create UI, snapshotted per run, and covered by regression tests.
- **Researcher dashboard + exports (FR-16/FR-17)**: a per-approach results dashboard plus a ZIP CSV export (one file per typed table) designed for downstream analysis.
- **Operational features beyond the proposal**: per-participant journey view, attention-check evaluation declared in questionnaire HTML and persisted at submit time, and robust deployment bootstrapping of runtime assets from GitHub Releases.
- **Upstream parity preserved**: upstream EasyStudy plugins (`fastcompare`, `layoutshuffling`, `vae`, `empty_template`) remain loadable via the canonical contract so researchers see the same plugin matrix as upstream plus the steering plugin.

## Evaluation alignment with `proposal.tex` (Section “Evaluation”)

The proposal emphasises case-based validation and limited usability validation (5–10 users) rather than a full comparative study. This repository supports that approach:

- **Case-based validation**: the steering loop, reranking strategies and audit/export contracts are covered by regression tests in `tests/`.
- **Limited usability validation**: a short one-choice evaluation questionnaire is provided as `docs/evaluation_questionnaire.csv` so the team can collect quick feedback from a small group of non-expert users. In addition, we have already run a larger study for a submitted paper (≈200 participants; no-steering vs slider-steering). Details and raw processing live in the private offline repository referenced in [`tech-docs.md` Section 3.4](tech-docs.md#34-fr-03-dataset-selection-and-offline-pipeline-note).
- **Questionnaire infrastructure**: the system supports per-approach and final questionnaires as drop-in HTML files under `server/static/questionnairs/`, with attention-check specs declared inline and evaluated at submit time.
- **Documentation for reproducibility**: the audit schema, algorithms and deployment steps are documented in `docs/` (tech-docs, equations, formative recipes) so future researchers can reproduce runs and extend the framework.

## Risks from `proposal.tex` (Section “Risks”) 

- **SAE interpretability**: the submitted paper’s results indicate the interpretability problem is manageable in practice for the slider-steering setup (participants reported mostly positive experience). The system also supports fallback interaction patterns (example-based steering, feature search).
- **Natural-language mapping**: treated as an active research question; we actively investigate the right semantics for text steering and study composition for another paper submitting.
- **Steering coherence**: conflicts were not observed as a practical issue in the submitted paper; the UI and audit pipeline make it possible to detect and analyse incoherent patterns post-hoc if they occur.
- **Performance**: the system is operated in controlled participant batches (≈5–50) to manage budget and data quality; this load does not require Redis-backed sessions. The architecture remains swappable if future deployments need it.
- **Integration complexity**: keeping upstream EasyStudy parity (canonical plugin registry + smoke tests for upstream plugins) reduces integration risk; merging with original EasyStudy is being discussed with the author.
