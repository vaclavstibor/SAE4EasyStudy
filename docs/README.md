# Documentation

Top-level documentation for the SAE Steering study framework, an EasyStudy derivative.

## Contents

- [What to read](#what-to-read) — quick “which doc for whom” index.

## What to read

| File | Audience | When to read it |
| --- | --- | --- |
| [`tech-docs.md`](tech-docs.md) | reviewers, maintainers | Canonical technical reference. Read this first. Architecture, plugin contract, DB schema, audit pipeline, analytics, runtime/deployment, testing. |
| [`design-decisions.md`](design-decisions.md) | reviewers, future maintainers | *Why* the architecture looks like this. Binding design choices and rationale. |
| [`equations.md`](equations.md) | reviewers, downstream researchers | Math behind scoring and reranking (text steering, SAE shifts, ELSA seed, reranking strategies). |
| [`admin-manual.md`](admin-manual.md) | researchers running studies | How to create/run studies, monitor results, and export CSVs. |
| [`user-manual.md`](user-manual.md) | participants | Participant walkthrough from join to finish. |
| [`formative-examples.md`](formative-examples.md) | future contributors | Worked recipes: add a plugin, modality, dataset, typed audit table, reranking strategy, CSV file. |
| [`contribution-summary.md`](contribution-summary.md) | reviewers | Short overview of what this repository adds to upstream EasyStudy. |

