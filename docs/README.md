# Documentation

Top-level documentation for the SAE Steering study framework, an EasyStudy derivative.

## What to read

| File | When to read it |
| --- | --- |
| [`tech-docs.md`](tech-docs.md) | The canonical technical reference. Read this first. Covers architecture, plugin contract, database schema, audit pipeline, analytics, runtime, deployment, and testing strategy. |
| [`design-decisions.md`](design-decisions.md) | The *why* document. Records the binding architectural decisions (plugin-first extension of EasyStudy, typed audit tables, no migration framework, single-writer audit service, dedicated reset endpoint, NFR-12 graceful degradation, etc.). |
| [`formative-examples.md`](formative-examples.md) | Worked recipes with code snippets — how to add a new plugin, a new steering modality, a new dataset, a new typed audit table, a new reranking strategy, a new dashboard metric, a new participant endpoint, a new CSV file. |
| [`equations.md`](equations.md) | Math reference: text-steering scoring (FR-09), SAE feature shift, ELSA seed update, reranking strategies (FR-10), example-based steering (FR-08). |
| [`admin-manual.md`](admin-manual.md) | Researcher manual: how to create a study, run it, monitor it, inspect the dashboard, export CSVs. |
| [`user-manual.md`](user-manual.md) | Participant walkthrough from join to finish. |

## Conventions

- Code paths are relative to repository root.
- The proposal (`../proposal.tex`) is the source of truth for FR / NFR ids referenced in the docs. Where the implementation deviates from the proposal (e.g. dashboard sub-items not surfaced as standalone cards, no "last 10" cap on iteration history), the deviation is called out explicitly in [`tech-docs.md` Section 11.1](tech-docs.md#111-limitations).
- Three reranking strategies (`feature-conditioned`, `latent-perturbation`, `constrained-subset`) are implemented and selectable in the create UI. The remaining research-track items — sentence-transformer text steering, Redis-backed sessions for >100 concurrent users, multi-dataset support, demographics dashboard cards — are **documented** but **not enabled** in this build. See [`tech-docs.md` Section 11](tech-docs.md#11-limitations-and-future-work) for the full limitations / future-work table.
