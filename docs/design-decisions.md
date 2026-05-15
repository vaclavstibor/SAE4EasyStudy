# Design Decisions

This page records the binding architectural decisions and the reasoning behind them. It is a companion to [`tech-docs.md`](tech-docs.md): the tech doc tells you *what* the system looks like, this one tells you *why*. Note that this files is also for our decision making process archive between studies we have already done and will be modified (making smaller) or removed in the future.

Each section follows the same shape: **Decision, Context, Alternatives considered, Consequences**.

## Contents

### Foundations

1. [Plugin-first extension of EasyStudy](#1-plugin-first-extension-of-easystudy-rather-than-a-fork)  
   Keep upstream parity and stay mergeable.
2. [Typed audit tables + thin envelope](#2-typed-audit-tables--thin-envelope-no-json-event-soup)  
   Typed facts + stable analytics, no JSON parsing on reads.
3. [No migration framework](#3-models-are-the-single-source-of-truth--no-migration-framework)  
   Models are the schema; `create_all()` on boot.
4. [Single-writer audit service](#4-single-writer-audit-service-routes-own-the-session)  
   One writer, explicit session ownership.

### Participant loop decisions

5. [Reset endpoint](#5-reset-is-a-dedicated-endpoint-not-a-flag-on-adjust-features)  
   Reset is its own action + audit fact.
6. [Text composition modes](#6-text-steering-composition-is-a-configurable-mode-fr-09)  
   `replace` / `add` / `intersect` are explicit study choices.
7. [NFR-12 no-match](#7-nfr-12-text-steering-ambiguity-degrades-gracefully)  
   Graceful degradation and analyzable failure mode.

### Researcher surfaces and comparative studies

8. [Dashboard & questionnaire invariants](#13-the-results-dashboard-is-generic--no-hard-coded-questionnaire-ids)  
   Generic dashboard, no hardcoded questionnaire ids.
9. [Attention checks](#18-attention-checks-are-declared-in-the-questionnaire-html-and-evaluated-at-submit-time)  
   Declared in HTML, evaluated at submit time.
10. [Side-by-side comparison invariants](#22-side-by-side-comparison-invariants-audit-fan-out-list_id-routing)  
   list_id routing and audit fan-out.
11. [Reranking strategies rationale](#23-reranking-strategies-reranking_strategy-config-key)  
   Why the set of strategies exists.


## 1. Plugin-first extension of EasyStudy (rather than a fork)

**Decision.** The application is implemented as a new plugin (`server/plugins/steering/`) under upstream [pdokoupil/EasyStudy](https://github.com/pdokoupil/EasyStudy/tree/main/server). The platform half (`server/platform/`) is a thin reshuffle that preserves every upstream Flask blueprint, ORM model, and plugin contract.

**Context.** The proposal mandates an EasyStudy-based study framework, and consultants will keep evolving EasyStudy upstream. The thesis must be re-mergeable.

**Alternatives considered.**

- **Hard fork.** Rejected — it would lock out future upstream improvements (Prolific integration, questionnaire features, new auth flows) and require a manual rebase every time.
- **Branch on `pdokoupil/EasyStudy` directly.** Considering for the future.

**Consequences.**

- `Interaction` and `Message` ORM tables stay even though the steering plugin does not use them. Upstream `fastcompare` and `utils` plugins still log to them. Removing them would break upstream parity.
- The platform must not import from `server.plugins.*` at module top-level. Architectural rule #5 in [`tech-docs.md` Section 4.4](tech-docs.md#44-architectural-rules) enforces this.
- New project code lives under `server/plugins/steering/`. Other plugins (`fastcompare`, `empty_template`, `utils`) are kept.


## 2. Typed audit tables + thin envelope (no JSON event soup)

**Decision.** Every user action writes one **typed** row to a domain-specific table (e.g. `sae_feature_adjustment`) **and** one minimal `SaeSteeringEvent` envelope row that carries only ids and timestamps. Analytics joins the typed tables. `SaeSteeringEvent.raw_payload` is provenance only and is **never** read by analytics.

**Context.** The proposal mandates per-approach metrics (FR-16) and a CSV export per fact (FR-17). Upstream EasyStudy logs participant actions through the generic `Interaction(participation_id, interaction_type, data, time)` table, where `data` is a free-form JSON column. That shape is fine for the upstream `fastcompare` plugin (uniform click logs aggregated in Python), but our analytics need to filter by `approach_id`, by modality, by cluster, and by `event_id` joins — every dashboard query would otherwise have to parse JSON at read time and would lose the column-level contract that lets researchers diff schema changes against the model file.

**Alternatives considered.**

- **Keep one event table with indexed JSON columns** (Postgres `jsonb` + GIN). Rejected — SQLite has no equivalent, and the column-shape contract was already implicit in the analytics code.
- **One typed table per action *type* but a generic `payload` blob next to typed columns.** Rejected as half-measure: every analytic query would still need to know which fields are typed and which are JSON.

**Consequences.**

- Adding a new fact requires three changes: a new model, a new `audit.record_*` function, a CSV writer in `routes/results/views.py`. Documented in [`formative-examples.md` Section 4](formative-examples.md#4-add-a-new-typed-audit-table).
- The `raw_payload` column stays as a provenance escape hatch for journey rendering and manual debugging. It is intentionally not indexed and not in any dashboard query.
- Per-table CSV export is trivial: each writer reads one typed table and emits one CSV.


## 3. Models are the single source of truth — no migration framework

**Decision.** `server/platform/persistence/base_models.py` and each plugin's `persistence/models.py` are the only source of truth for the schema. `db.create_all()` runs on every boot. There is no Alembic, no `migrations/` directory, no `flask db upgrade`.

**Context.** The thesis runs a small, fixed number of studies. Schema changes are research-side decisions ("we want one more typed column"), not continuous deployments. With `db.create_all()` driven from the model files, adding a column is a one-place edit; with Alembic, the same change is two-place (model + autogenerated migration), and the autogenerate step is unreliable across SQLite and Postgres because SQLite has no `ALTER TABLE DROP COLUMN`. The cost of carrying a migration framework therefore outweighed the rare schema-evolution event.

**Alternatives considered.**

- **Make Alembic the single source of truth.** Rejected — autogenerate cannot fully cross the SQLite/Postgres boundary, and the project does not run enough independent deployments to amortise the upkeep cost.
- **Keep Alembic but stamp it on every boot.** Rejected — embeds a fallback code path on top of `db.create_all()`; researchers would not know which mechanism actually ran.

**Consequences.**

- `./scripts/init-db.sh` is idempotent: `db.create_all()` adds missing tables and leaves existing ones alone.
- `./scripts/reset-db.sh` is destructive: `db.drop_all()` then `db.create_all()`. Use whenever a model is reshaped during development.
- Production schema changes happen out of band (manual `ALTER TABLE` against the managed Postgres) before deploying the new build. This is acceptable because the thesis runs a fixed number of studies and schema changes are rare research-side decisions, not continuous deployments.
- Reintroducing Alembic is a deliberate, opt-in decision when the application moves out of thesis scope and into a long-lived production service.


## 4. Single-writer audit service; routes own the session

**Decision.** Only `server/plugins/steering/service/audit.py` writes to typed audit tables. All `record_*` functions take `participation_id` and `approach_index` as **keyword-only** arguments. The service module never reads `flask.session`. Routes pass session values in explicitly.

**Context.** The audit pipeline is the single most fragile contract in the system: every typed table, every CSV export, and every dashboard chart depends on it writing the right ids in the right order. A service that also read `flask.session` would be hard to unit-test (every test would need a `flask.request_context`) and would couple session state to write semantics. Pushing session access to the route layer keeps the audit service pure-functional from the request boundary inward.

**Alternatives considered.**

- **Service reads session, routes are thin.** Rejected because it forces tests to stand up a `flask.request_context` for unit-level audit assertions and because re-entrant calls (`ensure_study_run` invoking helpers that need the same context) become hard to reason about.
- **No service layer; routes write rows directly.** Rejected because it would distribute the audit contract across ~20 route handlers and make consistent envelope writing impossible.

**Consequences.**

- The service is unit-testable without a request context.
- The recursion is gone: `ensure_study_run` is called once per request, and `audit.record_*` callers pass `approach_index` explicitly so `ensure_approach_run` is idempotent.
- Adding a new typed table is a one-place change ([`formative-examples.md` Section 4](formative-examples.md#4-add-a-new-typed-audit-table)).


## 5. Reset is a dedicated endpoint, not a flag on `/adjust-features`

**Decision.** `POST /sae_steering/reset` is its own endpoint. It writes one `SaeResetAction` row + one envelope and clears session state. `/adjust-features` does not accept a "reset" flag.

**Context.** FR-12 in the proposal is "single-click reset, recorded". Overloading `/adjust-features` with a `reset_all` payload would produce ambiguous audit rows (a reset that looks like a degenerate adjustment) and would force the analytics layer to special-case `applied_via='reset'` rows everywhere it counts adjustments. A separate endpoint keeps the typed-audit contract clean: a reset is a fact of its own type, with its own row, its own column set, and its own analytics card.

**Alternatives considered.**

- **Keep the flag, but write a distinct `SaeResetAction` row alongside the adjustment.** Rejected — the route still does two things and the dashboard query still has to coalesce.
- **`DELETE /sae_steering/adjustments`.** Rejected — `DELETE` over a collection that doesn't map to a REST resource was harder to document; the participant action is "reset", not "delete adjustments".

**Consequences.**

- The "Reset all controls" button POSTs to `/reset`. The iteration controller no longer reads `reset_all`.
- Analytics counts `sae_reset_action` rows directly. No coalescing.
- Tests cover the contract: see `test_steering_actions_and_security.py::test_reset_writes_one_audit_row_and_clears_session`.


## 6. Text steering composition is a configurable mode (FR-09)

**Decision.** Each study chooses how successive NL prompts compose: `replace` (default), `add`, or `intersect`. The composed adjustments are returned by `/parse-text-steering` and persisted as `SaeApproachRun.composition_mode` snapshot.

**Context.** Pilot feedback showed two patterns: participants who liked giving one description and being done (matches `replace`), and participants who built up intent iteratively ("I like Marvel", then "but only the comedies", matches `add` or `intersect`). The previous behaviour was implicit `replace`, which felt brittle for the iterative pattern.

**Alternatives considered.**

- **Always `add`.** Rejected — too easy to drift toward all-clusters-active.
- **Heuristically detect "but" / "however" and switch mode.** Rejected — fragile, and not in the participant's mental model.

**Consequences.**

- Three explicit modes, validated in `study_config.py::normalize_text_composition_mode`.
- `add` mode clamps each cluster sum to `[-0.95, 0.95]` so weights stay in the recommender's effective range.
- `intersect` mode keeps only clusters present in both iterations, using iteration N's weight. Useful for participants refining a stable intent.
- See [`equations.md` Section 4.5](equations.md#45-top-k-and-composition-across-iterations) for the formal definition.


## 7. NFR-12: text-steering ambiguity degrades gracefully

**Decision.** When the lexical resolver matches zero clusters, `/parse-text-steering` returns HTTP 200 with `status="no-match"` and a friendly message. A `SaeTextSteeringQuery` row is still written (with zero matches), so the failure mode is analyzable offline.

**Context.** NFR-12 (graceful degradation) is in the proposal. Off-topic or contradictory prompts ("I don't know what I want", "more action but no action") legitimately produce empty match sets in a lexical resolver. Returning an empty `features` array without a status hint would leave the UI in an ambiguous state ("did nothing change because I'm wrong, or because the system is broken?"); the explicit `no-match` status with a friendly message answers that question once, in one place.

**Consequences.**

- The UI renders an alert: "We could not match your text to any feature, try different wording".
- Analytics can count zero-match queries (`COUNT(*) FROM sae_text_steering_query WHERE NOT EXISTS (SELECT 1 FROM sae_text_steering_match WHERE query_id = q.id)`).
- The participant's previous adjustments are preserved (no destructive write on empty match).


## 8. Reranking strategy as a typed enum (FR-10) — schema and dispatch contract

**Decision.** `study_config.reranking_strategy` is a typed enum (`SUPPORTED_RERANKING_STRATEGIES` in `constants.py`) and is snapshotted onto `SaeApproachRun.reranking_strategy` so historical study runs always carry the strategy that was active when they ran. Validation happens at config-normalisation time; unknown values fall back to the default. The dispatch lives inside `recommendation/sae_recommender.py::get_recommendations` so call sites stay uniform regardless of which strategy is active. The mathematical content of each strategy is in [`equations.md` Section 10](equations.md#10-reranking-strategies-rerankingstrategy-config-key); the implementation rationale for the *set* of strategies is in Section 23 below.

**Consequences.**

- `SaeApproachRun.reranking_strategy` is a stable string column, so studies running any of the three supported strategies co-exist in the same database. No backfill is needed when a new strategy is added — only the enum and the dispatch branch change.
- Adding a fourth strategy is a one-place change: add the enum value to `SUPPORTED_RERANKING_STRATEGIES`, add a branch inside `recommendation/sae_recommender.py::get_recommendations`, and document the math in `equations.md` Section 10. The recipe lives in [`formative-examples.md` Section 5](formative-examples.md#5-add-a-new-reranking-strategy).


## 9. Researcher endpoints require login; participant endpoints do not

**Decision.** Every researcher-facing endpoint carries `@login_required`. Participant endpoints (`/join`, `/preference-elicitation`, `/parse-text-steering`, `/adjust-features`, `/reset`, ...) do **not** because participants are anonymous from a Flask-Login perspective; their identity is the session `participation_id`.

**Context.** Researcher endpoints expose the list of studies, participant tallies, and the raw CSV bundle. Without `@login_required` anyone with a URL could enumerate studies and download participant data. Participant endpoints, by contrast, must remain anonymous so the Prolific link flow works without sign-up; the participant's identity is the session `participation_id`, which is bound to the opaque `Participation.uuid`.

**Consequences.**

- A parametrized regression test (`test_steering_actions_and_security.py::test_researcher_routes_require_login`) covers every researcher endpoint.
- The participant flow remains anonymous and Prolific-friendly.
- `Participation.uuid` (URL-safe, opaque) is the participant identifier; it is sufficient because the participant only ever acts on their own session.

---

## 10. Replace `assert` with `flask.abort(400)` in request handlers

**Decision.** Every request-validation check uses `if condition: flask.abort(400, "message")` instead of `assert`. Asserts are reserved for unreachable-state invariants in service code.

**Context.** Python's `assert` is removed by `-O`. Production Gunicorn invocations could end up with no validation at all. Worse, asserts produced traceback responses that leaked code paths.

**Consequences.**

- All admin route validators (`/user-study`, `/create-user-study`) and plugin join routes (`/sae_steering/join`, `/empty_template/join`) use explicit `abort(400, ...)`.
- Asserts remain in service code only where they catch programmer errors that should crash loudly in dev.

---

## 11. Pickle paths are constrained to the project's cache root

**Decision.** Every `pickle.load(...)` call resolves its path through a real-path check that confirms the path lies under `server/cache/utils/`. Paths outside this root raise `ValueError` before the file is opened.

**Context.** `data_loading.py` accepted a `ml_variant` string from study config and built `cache/utils/<ml_variant>/data_cache.pckl`. A malicious value (`../../etc/passwd`) would let an attacker read arbitrary paths if they could submit a study config. The `pickle.load` itself is an even larger risk if the file is attacker-controlled.

**Consequences.**

- `_resolve_safe_cache_path(ml_variant)` validates `ml_variant` against `^[A-Za-z0-9._-]+$` and resolves the cache path through `Path.resolve()`. Anything escaping the cache root raises `ValueError`.
- `rlprop_wrapper.py::_load_cache` re-checks the path before `pickle.load`.

---

## 12. Equations live in their own file

**Decision.** The math (text steering scoring, SAE shift expansion, ELSA seed update, reranking, example-based steering) is in [`equations.md`](equations.md). The technical narrative in [`tech-docs.md`](tech-docs.md) cross-references it.

**Context.** [`equations.md`](equations.md) has a different reading mode (formula lookup) and a different audience (reviewers and downstream researchers verifying scoring correctness). Bundling it inline would make `tech-docs.md` long without making the math easier to find.

**Consequences.**

- New scoring functions added in the future also belong in `equations.md`.
- Each section of `equations.md` is anchored, so `tech-docs.md` can deep-link without re-stating formulas.

---

## 13. The Results dashboard is generic — no hard-coded questionnaire ids

**Decision.** The dashboard only knows about the typed audit columns (`rank`, `delta`, `cluster_label`, `query_text`, `mood`, …). It must **not** branch on question-level ids like `p_attention_check` or `f1_preference`. Per-questionnaire summaries are produced by the modular *Questionnaire Monitor*, which auto-discovers fields from `SaeQuestionnaireResponse.answers` and infers a kind per field.

**Context.** Hard-wiring Likert deltas, attention-funnel and preference distribution charts to specific question ids from the bundled `sae_*_questionnaire.html` files would break silently whenever a researcher uploaded a different questionnaire — or worse, the analytics layer would pretend a result existed by reading missing keys. Researchers must be able to ship custom questionnaires (FR-15) without editing the dashboard, so the dashboard cannot know question ids in advance.

**Consequences.**

- `analytics._questionnaire_monitor` aggregates one section per `questionnaire_file`. Within each section, every key in the answers JSON becomes a row whose summary is chosen from the inferred kind (`likert` / `numeric` / `categorical` / `text`).
- The Overview tab focuses on the *behavioural* signal of steering: selected-movie rank distribution by approach, slider movement by cluster, text-prompt-to-cluster mappings. None of these depend on a specific questionnaire.
- Adding a new questionnaire is a no-code operation (drop a template in `server/static/questionnairs/`, point an approach at it). See [`formative-examples.md` Section 9](formative-examples.md#9-add-a-new-questionnaire).
- Researchers who want bespoke statistics (attention-funnel breakdowns, composite Likert deltas, …) compute them in R or pandas from the CSV bundle. The dashboard intentionally stays domain-agnostic.

---

## 14. Every typed audit row carries an `event_id` FK

**Decision.** Every typed table that records a participant action has an `event_id` FK to `sae_steering_event`. The audit service always writes the envelope **first** and then attaches the typed row using `event.id`.

**Context.** The envelope row is the timeline anchor: it carries the timestamp, the `event_type` enum and the participation / approach / iteration ids that every downstream view uses. If the typed row were written first and the envelope inherited *from it*, then any future code path that needs to find "the action that produced this typed row" would have to back-reference through column-specific joins. Pinning the direction (envelope first, typed row second, `event_id` FK on the typed row) keeps the read pattern uniform: timeline rendering, raw export and CSV export all walk `SaeSteeringEvent → typed table` in one direction.

**Consequences.**

- `SaeMovieFeedback.event_id` is `NOT NULL` and CASCADEs from the envelope.
- `audit.record_movie_feedback` now writes the envelope first and the feedback row inherits `event.study_run_id` / `approach_run_id` / `event_id` for full provenance.
- The journey view, raw export, and CSV export all surface `event_id` for movie feedback.
- Future typed tables MUST follow this order. The audit service docstring is the contract.

---

## 15. Author/contact lives in platform config, not in `PluginMetadata`

**Decision.** `PluginMetadata` carries only what the platform actually needs to identify a plugin (`name`, `version`, `description`). The study-author display name and contact email are platform-level configuration — `STUDY_AUTHOR_NAME` and `STUDY_AUTHOR_CONTACT` environment variables, surfaced through Flask config and a single Jinja `context_processor` in `server/platform/app.py`.

**Context.** Authorship information belongs to a *deployment*, not to a *plugin*. A single study deployment has one researcher contact, regardless of which plugins are loaded. Surfacing the contact through `PluginMetadata.author` would either duplicate the same string across every plugin (drift waiting to happen) or omit it (forcing per-template `mailto:` strings). The participant-facing surfaces that need an authorship signal — the study intro meta-strip, the finish screen contact chip, the participant footer, and the admin hero subcopy — should all read from one source.

**Alternatives considered.**

- **Keep `author` and `author_contact` on `PluginMetadata` and just rename the admin column.** Rejected — the data lives at the wrong granularity. A study deployment has one author, not one per plugin. Pushing the same string through every plugin contract is duplication waiting to drift.
- **Hard-code the contact in every template.** Rejected — the embedit / student / future-employer email rotation problem. One change should be one edit.
- **Read it from `pyproject.toml` at boot.** Rejected — `pyproject.toml` is build-time metadata; the contact may need to change per deployment (a Prolific run vs. a local demo vs. a published study) without rebuilding.

**Consequences.**

- `PluginMetadata` carries only three fields (`name`, `version`, `description`).
- `server/platform/app.py` registers a single `inject_study_author_info` context processor, so every Jinja template (platform or plugin) sees `study_author_contact` and `study_author_name` without per-route plumbing.
- Defaults are baked in (`Václav Stibor` / `vaclav.stibor@student.cuni.cz`) and can be overridden via env: `STUDY_AUTHOR_NAME`, `STUDY_AUTHOR_CONTACT`. Both `deployment/app.env.example` and `docker/compose.env.example` advertise the new variables.
- Four participant- and researcher-facing surfaces read from the same source: `study_intro.html` (meta-pill), `participant_flow/finish.html` (`.contact-chip`), `participant_flow/footer.html` and `layoutshuffling/footer.html` (mailto chip), and `administration.html` (hero subcopy). The platform footer is canonical for everything routed through the `utils` (participant-flow) blueprint, while `layoutshuffling` keeps its own footer because its routes resolve templates against its own blueprint folder first.
- `formative-examples.md` does not show `PLUGIN_AUTHOR*` in the copy-paste plugin skeleton — that channel is platform-level, not plugin-level.

---

## 16. Upstream `layoutshuffling` and `vae` plugins are wired through the canonical contract

**Decision.** The two upstream EasyStudy plugins (`server/plugins/layoutshuffling` and `server/plugins/vae`) are registered in `CANONICAL_PLUGIN_MODULES` so the kernel loads all five plugins (`sae_steering`, `fastcompare`, `emptytemplate`, `layoutshuffling`, `vae`). Which of these the admin "Available templates" picker shows is a separate decision — see Section 17. The migration is minimum-viable: routes stay in `__init__.py`, but each plugin now exposes a `PLUGIN: StudyPluginContract` and a `get_plugin()` callable so the kernel can register the blueprint exactly like a thesis-owned plugin.

**Context.** Upstream [`pdokoupil/EasyStudy`](https://github.com/pdokoupil/EasyStudy) ships five plugins (`fastcompare`, `utils`, `layoutshuffling`, `vae`, `empty_template`). The thesis adds a sixth (`steering`) and preserves the other five so a researcher who clones this repo gets the same plugin matrix as upstream plus the new steering work. `CANONICAL_PLUGIN_MODULES` therefore lists all canonical plugins (`steering`, `fastcompare`, `empty_template`, `layoutshuffling`, `vae`); `utils` is consumed as a library, not registered as a blueprint.

**Alternatives considered.**

- **Delete `layoutshuffling/` and `vae/` outright.** Rejected — they are upstream references that demonstrate two patterns the steering plugin does not (multi-algorithm shuffling layouts and a long-running async initialiser).
- **Rewrite both plugins to the new `plugin.py` + `routes/` layout used by `fastcompare`.** Rejected as out of scope: would require touching ~700 lines of upstream code with no functional gain. The contract surface is what the kernel needs; the internal layout is owned by each plugin.

**Consequences.**

- `Blueprint(__plugin_name__, __plugin_name__, …)` is replaced with `Blueprint(PLUGIN_NAME, __name__, …)` in both plugins so Flask resolves the template/static roots relative to the package rather than relative to the literal plugin name string.
- Upstream relative imports (`from plugins.utils.*`) are normalised to absolute imports (`from server.plugins.utils.*`) so the modules load under the `server.*` package layout.
- A regression test (`test_all_canonical_plugins_load_and_register` in `tests/plugins/steering/test_initialization.py`) asserts that every entry in `CANONICAL_PLUGIN_MODULES` loads cleanly **and** registers at least one route, so a future plugin-renaming refactor cannot silently drop an upstream package.
- Offline tooling around the typed-audit JSON shape stays consistent: `scripts/reconstruct_journey.py` imports `build_journey` from `server.plugins.steering.results.journey_builder`, which adapts the v1 export schema produced by `/export-raw/<guid>` into the same `{timeline, summary}` envelope the on-screen admin journey view uses.

---

## 17. Admin "Available templates" is filtered by `PluginMetadata.hidden_from_admin`

**Decision.** A plugin shows up in `/loaded-plugins` (the JSON feed behind the admin "Available templates" table) only when it (a) registered a `/<name>/create` endpoint and (b) did not opt out via `PluginMetadata.hidden_from_admin`. The flag defaults to `False`, so existing plugins keep showing. Two canonical plugins set it to `True`:

| Plugin | Reason to hide |
| --- | --- |
| `emptytemplate` (`server/plugins/empty_template`) | Developer scaffold. It is the copy-paste starter for new plugins (see [`formative-examples.md` Section 1](formative-examples.md#1-add-a-new-plugin)); listing it as a study type would invite researchers to "create studies" out of a placeholder. |
| `vae` (`server/plugins/vae`) | Algorithm wrapper. Provides VAE algorithm hooks consumed by `fastcompare`; there is no `/vae/create` because there is no VAE-only study type. |

**Context.** With all canonical plugins registered (Section 16), the admin picker would otherwise surface `emptytemplate` and `vae` next to `sae_steering` and `fastcompare` — neither is a study type a researcher should pick (one is a scaffold, the other is an algorithm wrapper). A hard-coded skip list in `get_loaded_plugins()` would hide that intent inside the platform; instead each plugin declares its admin-visibility through its own metadata, so the filter rule is uniform and traceable from the plugin file.

**Alternatives considered.**

- **Hide plugins by removing their `/create` route.** Rejected — `empty_template` documents the `/create` flow precisely so a new plugin author can copy the surface. Removing the route would defeat its purpose as a scaffold.
- **Hard-code a skip list in the admin route.** Rejected — couples the platform to specific plugin names. Adding a new internal plugin (e.g. a future debug or benchmark helper) would require an unrelated edit in the admin module.
- **Introduce `PluginMetadata.kind = "study" | "scaffolding" | "algorithm"`.** Rejected as premature — the only distinction the kernel actually makes today is "show in admin or not". A richer taxonomy can be added later if the platform grows more plugin classes; it would still subsume `hidden_from_admin` rather than replace it.

**Consequences.**

- `PluginMetadata` gains one optional field (`hidden_from_admin: bool = False`). `get_loaded_plugins()` carries a docstring describing the filter contract.
- New plugins default to **visible**. Internal plugins must explicitly set `hidden_from_admin=True`; future contributors learn this from the `emptytemplate` and `vae` contracts plus this decision.
- A regression test (`test_admin_available_templates_excludes_hidden_plugins` in `tests/plugins/steering/test_initialization.py`) asserts that `emptytemplate` and `vae` are absent from `get_loaded_plugins()` while the three researcher-facing plugins are present.
- The admin manual ([`admin-manual.md` Section 2](admin-manual.md#2-create-a-study)) and the new-plugin walk-through ([`formative-examples.md` Section 1](formative-examples.md#1-add-a-new-plugin)) point at this decision so authors of future plugins know how to control admin visibility.

---

## 18. Attention checks are declared in the questionnaire HTML and evaluated at submit time

**Decision.** Each questionnaire HTML file ships its attention-check answer key as an inline JSON block. The audit pipeline evaluates the submission against that key once, at submit time, and stores the verdict on `SaeQuestionnaireResponse.attention_check_passed` (`True` / `False` / `NULL` when no spec is declared). The Results dashboard reads only the stored verdict and lets the admin set a per-study pass threshold in the participants table.

**Context.** A `QUESTIONNAIRES = <count>` column in the participants table would merely echo the configured approach count plus the final questionnaire — uninformative for the researcher who wants to know whether the participant actually read the questions. That signal already exists inside the answers JSON (`p_attention_check`, `f_attention_check`, …). Recomputing it in the dashboard every page load would couple analytics to a moving spec (editing the questionnaire HTML afterwards would silently change historical verdicts), so the verdict is computed once on write and persisted.

The bundled questionnaires now ship the following specs (`server/static/questionnairs/sae_*_questionnaire.html`):

| File | Field | Rule |
| --- | --- | --- |
| `sae_explicit_feedback_approach_questionnaire.html` | `p_attention_check` | `expected_one_of: ["1", "2", "3"]` |
| `sae_implicit_feedback_approach_questionnaire.html` | `p_attention_check` | `expected: "7"` |
| `sae_final_questionnaire.html` | `f_attention_check` | `expected: "same"` |

**Spec format.** `server/plugins/steering/results/attention_checks.py` parses `<script type="application/json" data-attention-checks>{ ... }</script>` and supports three condition keys per field: `expected` (exact string equality), `expected_one_of` (list membership), and `expected_range` (inclusive numeric range). A submission passes iff **every** declared field passes; missing fields fail. Custom researcher-uploaded questionnaires can declare specs the same way; questionnaires that ship no spec record `NULL` and are excluded from the pass/total ratio.

**Storage.** `SaeQuestionnaireResponse.attention_check_passed: Boolean | NULL` — one verdict per submission. Per-questionnaire details (which field was wrong) are NOT stored; the journey view recomputes them on demand from the spec and the answers, so editing the spec retroactively never desyncs the table.

**Threshold semantics.** A participant has a `passed / total` tally aggregated server-side in `analytics.build_results_payload` (only submissions with a verdict contribute to `total`). The participants table renders a green `PASS p/t` or red `FAIL p/t` badge based on a per-study threshold the admin types into the filter bar. The threshold persists in `localStorage` under `attn_threshold:<guid>` and defaults to the highest observed `total` (i.e. "every declared check must pass"). This is intentionally *not* stored in the database — the threshold is a reading lens over the same data, not a property of the submissions.

**Alternatives considered.**

- **Compute pass/fail on every page load.** Rejected — couples the dashboard to the live HTML; editing a question wording without touching the spec would silently change historical verdicts.
- **Add a runtime migration framework to backfill the column for old DBs.** Rejected (see Section 3). The column is additive and nullable; researchers who want the new column on an existing DB run `./scripts/reset-db.sh` and create a fresh study, which is the project's standing migration story.
- **Bake the spec into Python config next to the questionnaire path.** Rejected — questionnaires are self-contained drop-in HTML files (see [`formative-examples.md` Section 9](formative-examples.md#9-add-a-new-questionnaire)). Forcing researchers to edit a Python config to add a check would split a single artefact across two files.

**Consequences.**

- New column `SaeQuestionnaireResponse.attention_check_passed` (nullable Boolean). Picked up by `db.create_all()` on a fresh DB — no migration; reset the DB to use it on an existing deployment.
- `audit.record_questionnaire_response` calls `attention_checks.evaluate_for_file(file, answers)` once per submission and writes the verdict.
- `analytics.build_results_payload` adds `participants_table[].attention_checks = { passed, total }`; the JSON results export carries this verbatim.
- `journey.py` adds `attention_check_passed` plus `attention_check_details` to each `questionnaire_responses` row so the journey view can show a PASS/FAIL badge with a hover-tooltip detail (`field: expected X, got Y`).
- The participants table swaps the `QUESTIONNAIRES` column for `ATTENTION CHECKS` and adds the threshold input; the `Questionnaires` count is still surfaced in the journey summary card, where it remains useful.
- Tests in `tests/plugins/steering/test_attention_checks.py` cover the evaluator semantics, malformed JSON resilience, and the spec/answer contract of every bundled questionnaire (so editing one of those HTML files without re-running tests fails loudly). `tests/plugins/steering/test_sae_audit.py::test_record_questionnaire_response_stores_attention_check_verdict` covers the cross-cut between the evaluator and the audit pipeline.

---

## 19. Cross-participant analytics group by `approach_id`, never by `approach_index`

**Decision.** Every aggregation in `analytics.py` that crosses participants groups by `SaeApproachRun.approach_id` (a stable identifier from `conf['models'][i]['id']`) and never by `approach_index`. `approach_index` records the per-participant phase position (1st, 2nd, … approach the participant saw) and is meaningless across participants when `randomize_approach_order` is enabled. The dashboard returns each per-approach dict pre-seeded in config-models order so the rendering order is deterministic without sorting on the client.

**Context.** With two participants and `randomize_approach_order=True`, participant P1 might see `approach_a` at phase 0 and `approach_b` at phase 1, while P2 sees the opposite. The audit rows correctly store both `approach_index` (the phase position the participant saw) and `approach_id` (the stable identity from `conf`). Grouping cross-participant aggregates by `approach_index` would pick a bucket label from whichever `approach_name` SQL returned first for that index — every cross-participant view (Selection Dynamics, Slider Movement, Selected movie rank distribution, Approach Overview) would then show two rows both labelled "Approach A" with the data being a 50/50 mix of two approaches under one label, silently destroying the comparison.

**Approach identity (`approach_id`)** and **phase position (`approach_index`)** are different concepts, and the only correct cross-participant key is identity. Frontend code relies on insertion order, so the analytics layer is responsible for emitting buckets in canonical (config-models) order.

**Alternatives considered.**

- **Group by `approach_name` instead of `approach_id`.** Rejected — `approach_name` is researcher-facing free text that may be edited mid-study. `approach_id` is the immutable identifier (`approach_1`, `approach_2`, …) committed at study-create time.
- **Disable randomization to avoid the bug.** Rejected — randomization is a study-design feature (counterbalancing) the framework MUST support; the bug was in the analytics layer, not in the randomizer.
- **Compute the canonical mapping in the dashboard JS.** Rejected — couples three views (table, two chart cards) to the same client-side reconciliation step. Cleaner to fix once at the SQL layer.

**Consequences.**

- `_selected_rank_distribution`, `_slider_movement_by_position`, `_selection_dynamics`, `_approach_overview` all take `config_models: list[dict]` and group by `SaeApproachRun.approach_id`. The first three join `SaeApproachRun` from the fact table (`SaeMovieFeedback`, `SaeFeatureAdjustment`) when the fact table itself does not carry `approach_id`. The returned dict is keyed by `approach_id` and seeded in config order so the dashboard renders A / B / … left-to-right.
- `_slider_movement_by_position` additionally filters out placeholder `cluster_label` values (`feature_*`, `cluster_*`, `Feature cluster_*`) at the SQL layer. The chart is advertised as a *named-cluster* movement chart; unnamed-cluster ids belong to the raw export and would otherwise dominate the y-axis.
- Frontend stops sorting `Object.entries(dist)` (the sort was numeric and silently broke once keys became strings like `approach_1`). Insertion order from the server is the contract.
- A new typed table column (`SaeApproachRun.approach_id`, already present) becomes load-bearing. Any future fact table whose analytics cross participants MUST either store `approach_id` directly or join `SaeApproachRun` to recover it.
- `tests/plugins/steering/test_approach_order_and_results.py::test_analytics_groups_by_approach_id_not_phase_index` is the regression guard. It seeds two participants with opposite randomized orders and asserts every cross-participant aggregate has the right label per bucket and the right number of buckets (one per approach).

---

## 20. The Modalities dashboard is driven by `enabled_modalities`, not by audit-table contents

**Decision.** The Modalities tab and the Overview "Modality usage by approach" card grid render exactly the modalities each approach declared in `conf['models'][i]['enabled_modalities']`. The audit-table contents inform the *counts* inside each modality card, but they never decide which cards are shown. An approach with `enabled_modalities=["text"]` never gets a slider card — even if the typed adjustment table contains rows for that approach (which the text pipeline does write, with placeholder cluster labels).

**Context.** Hardcoding two chart cards labelled "Approach A" and "Approach B" that both render *Slider Movement by Cluster* would mislabel modalities for any study where one approach uses sliders and another uses text. The text pipeline writes feature-adjustment rows too (composition produces cluster weights), but their `cluster_label` is a placeholder (`feature_<n>` / `cluster_<n>`); after Section 19 filters placeholders out of the slider chart, a card for a text-only approach would stay visually present but empty — the dashboard would imply "this approach has slider data" when it does not. A single global "Modality usage" card would hide the fact that one approach used only text and the other only sliders.

The dashboard must therefore read the *study contract* (`enabled_modalities`) as the source of truth for what cards to render. The audit table answers a different question (what happened) and is consulted only after the contract decides what to show.

**Backend design.** `analytics._approach_modality_breakdown(user_study_id, config_models)` returns one entry per approach keyed by `approach_id`, in config order:

```json
{
  "approach_a": {
    "label": "Approach A",
    "steering_mode": "sliders",
    "modalities_enabled": ["sliders", "reset"],
    "participations": 2,
    "modalities": {
      "sliders": { "label": "Slider steering", "metrics": [{"key": "adjustments", "label": "Adjustments", "value": 9, "fmt": "int"}, ...] },
      "reset":   { "label": "Reset events",    "metrics": [{"key": "reset_count",  "label": "Reset events", "value": 0, "fmt": "int"}] }
    }
  },
  "approach_b": {
    "label": "Approach B",
    "steering_mode": "text",
    "modalities_enabled": ["text"],
    "participations": 2,
    "modalities": {
      "text": { "label": "Text steering", "metrics": [{"key": "queries", ...}, {"key": "distinct_prompts", ...}, {"key": "cluster_mappings", ...}] }
    }
  }
}
```

Five canonical modalities are wired today: `sliders`, `toggles`, `text`, `examples`, `reset`. Each has a tiny metric function in `_approach_modality_breakdown` that reads exactly one typed audit table (e.g. `_slider_metrics` reads `SaeFeatureAdjustment` filtered by the approach's run ids). Adding a new modality means (a) adding the canonical name to `_MODALITY_LABELS`, (b) writing a `_<name>_metrics(run_ids)` closure that returns a list of `{key, label, value, fmt}` rows, and (c) wiring it in the `metric_fns` dict — the frontend renders the result generically without any modality-specific code.

**Frontend design.** Two render entry points consume the breakdown.

1. *Overview tab.* `renderModalityBreakdownCards(data)` produces one `.card` per approach inside `.card-grid`. The card header carries the approach label and a `modality-mode-pill` showing `steering_mode`; the body lists one `.modality-block` per modality with a 2-column key/value grid for that modality's metrics. Zero-valued metrics are dimmed.
2. *Modalities tab.* `renderModalitiesTab(data)` produces one `<section>` per approach. Each section's chart grid contains one card per modality that benefits from a chart/table:
   - `sliders` / `toggles`: Chart.js horizontal bar of mean $|\Delta|$ on the top `15` named clusters (read from `approaches.slider_movement[approach_id]`).
   - `text`: a per-approach prompt-to-cluster table (filtered from the global `modalities.text_prompt_mappings` array by `approach_id`).
   - Approaches with no chart-worthy modality (e.g. only `reset` enabled) get a single empty-state card explaining which modalities they declared. The whole tab gets a study-wide empty state if no approach declared any steerable modality.

Both paths iterate over `Object.entries(modality_breakdown)`; they never branch on a hardcoded modality name. The Modalities tab no longer hardcodes `sliderApproachAChart` / `sliderApproachBChart` ids — canvases are minted per `approach_id` (`modSliderChart_<approach_id>`), so an N-approach study renders N chart cards automatically.

**Alternatives considered.**

- **Show every modality card with "no data" placeholder when the approach didn't declare it.** Rejected — visually noisy and gives the false impression that the researcher could enable a different modality post-hoc by editing the chart.
- **Derive `enabled_modalities` from the audit table instead of the config.** Rejected — the typed tables contain side effects of pipelines (e.g. the text composer writes feature adjustments), so reverse-engineering intent from data would conflate intent with side effects.
- **Hardcode a per-modality SQL query in the frontend.** Rejected — pushes domain logic to the wrong layer and couples the dashboard to plugin internals.

**Consequences.**

- New analytics function `_approach_modality_breakdown` + helper `_approach_run_ids_by_approach`. Payload gains `approaches.modality_breakdown`. The old payload key `modalities.modality_usage` (the global Counter) is still emitted for raw-export consumers but no longer rendered.
- `_text_prompt_cluster_mappings` now joins `SaeApproachRun` and emits `approach_id` + `approach_label` on every row so the frontend can shard the table by approach.
- The old `Structured Steering Events` Overview section, both static slider-chart canvases (`sliderApproachAChart` / `sliderApproachBChart`), and the standalone `text-mappings-table` are removed from the template. Render functions `renderStructuredEvents`, `renderSliderMovement`, `renderTextMappings` are deleted; `renderModalityBreakdownCards` + `renderModalitiesTab` supersede them.
- Adding a new approach to the study (or removing one) requires no template change — the dashboard scales with `conf['models']`.
- `tests/plugins/steering/test_approach_order_and_results.py::test_modality_breakdown_is_driven_by_enabled_modalities` is the regression guard. It seeds one slider-only and one text-only approach, plus a text-driven feature adjustment with a placeholder cluster label, and asserts: (a) the slider approach has `sliders` + `reset` cards with correct counts; (b) the text approach has only the `text` card despite the feature-adjustment row; (c) modalities the approach did not declare are absent.

---

## 21. Text-steering state is namespaced by `(study_guid, phase_index)` and the UI surface is reset per iteration

**Decision.** Two complementary rules govern text-steering state across iterations / phases / studies:

1. **Backend scope guard.** `session["last_text_steering"]` carries a `scope` field equal to `"<study_guid>:<phase_index>"`. Both `parse_text_steering` (which writes it) and the `previous_text_query` lookup in `session_controller` (which reads it for the "You said before" banner) ignore the payload when the stored scope does not match the current scope. Without this, a Flask session cookie that survives a study transition would re-surface the prior study's last prompt in the new study's UI, and a phase-2 participant would inherit phase-1's prompt history.
2. **Per-iteration UI clear.** When `fetchAndRender` (Get Recommendations) succeeds, the steering interface explicitly clears the prompt input, the "detected cluster chips" container, the "You said before" banner, and the "no match" hint. The current iteration's prompt is already encoded in the adjustments that just produced the new recommendations; leaving the prompt text + chips visible implies "your next recommendations will again be because you said X", which is misleading. Server-side `last_text_steering` is **untouched** by the UI clear — composition modes that need the prior dict (`add`, `intersect`) continue to work for the next parse within the same `(guid, phase)` scope.

**Context.** Without a scope tag, `last_text_steering` would be a flat dict (`{"query": ..., "adjustments": ...}`) in the Flask session with no isolation. A participant who finished study A and then joined study B in the same browser would see A's last prompt in the "You said before" banner; the same leak would occur when advancing between approaches (phase-1 participants seeing phase-0's prompt). Symmetrically, the iteration boundary on Get Recommendations would leave the prompt input and chip tags from the previous iteration on screen, even though those tags describe an interaction that has already been consumed and persisted into adjustments — a misleading affordance for the participant.

**Implementation.**

- `routes/steering/actions.py` exposes two helpers: `_text_steering_scope()` derives `"<guid>:<phase>"` from the session, and `_read_scoped_text_steering()` returns the stored payload only when its `scope` matches. `parse_text_steering` calls both — reads the previous dict via `_read_scoped_text_steering()` for composition, then writes `{"scope": _text_steering_scope(), "query": ..., "adjustments": ...}`.
- `service/session_controller.py` performs the same scope check inline when populating `previous_text_query` for the template.
- `routes/study.py::_do_advance_phase` and `show_features` `pop("last_text_steering")` as belt-and-suspenders cleanup so the session payload doesn't carry dead bytes between phases / studies (the scope check would already filter them out, this just keeps the cookie small).
- `templates/steering_interface.html::fetchAndRender` resets `#text-input`, `#detected-tags-container`, `#previous-text-query`, and `#text-steering-no-match` on success.

**Alternatives considered.**

- *Store one entry per scope in a nested dict (`session["last_text_steering"][scope] = ...`).* Rejected — unbounded growth across $N$ phases and $M$ studies, and we never read past entries anyway. A single-slot store with a scope tag is strictly simpler.
- *Clear `last_text_steering` on every iteration server-side.* Rejected — would break `add` / `intersect` composition modes, which by design accumulate cluster weights across the iterations of a single phase (FR-09).
- *Clear the UI surface on iteration only when composition mode is `replace`.* Rejected — composition mode is a backend concern; the participant's mental model is that "Get Recommendations" closes one iteration and opens the next, regardless of the math under the hood.
- *Move `last_text_steering` from the Flask session into the database.* Rejected — adds persistence overhead for a transient UI affordance that is already an opt-in convenience, not a contract.

**Consequences.**

- Two new helpers in `actions.py` (`_text_steering_scope`, `_read_scoped_text_steering`). `parse_text_steering` writes a 3-key payload (`scope`, `query`, `adjustments`) instead of 2.
- The existing reset test (`test_reset_clears_in_session_steering_state_and_writes_one_audit_row`) continues to assert that `session["last_text_steering"] == {}` after a global reset.
- `tests/plugins/steering/test_steering_actions_and_security.py` adds four regression tests:
  - `test_parse_text_steering_stamps_scope_on_session_payload`
  - `test_parse_text_steering_ignores_previous_payload_from_other_study`
  - `test_parse_text_steering_ignores_previous_payload_from_other_phase`
  - `test_parse_text_steering_composition_uses_previous_when_scope_matches`
- The UI clear is a pure CSS/JS change in `steering_interface.html`; no backend contract was broken.

## 22. Side-by-side comparison: invariants, audit fan-out, list_id routing

**Context.** A side-by-side study renders **two recommendation columns** for the same participant at the same time. The UI has one shared slider grid, one shared text input, and one set of buttons. Two `SaeApproachRun` rows exist for the participant — one per column. The participant likes movies in either column, but does not advance phases (there is only one phase in side-by-side).

This design has three subtle invariants that are easy to break:

**Invariant 1 — Exactly two approaches.**

```232:244:server/plugins/steering/study_config.py
    comparison_mode = (conf.get("comparison_mode") or "").strip().lower()
    if len(models) <= 1:
        comparison_mode = "none"
    elif len(models) > 2:
        # Side-by-side is a strict two-approach contract — there are
        # exactly two recommendation columns, one shared slider grid,
        # one shared text input. With three or more approaches the
        # layout has nowhere to put approach 3+, so the study auto-
        # falls back to sequential and the participant walks approaches
        # one at a time. See design-decisions.md Section 22.
        comparison_mode = "sequential"
```

A study with 3+ approaches *cannot* be side-by-side; the UI has no third column. Config normalisation enforces this silently by downgrading the mode to `sequential`. With 2 approaches, side-by-side is the right answer; with 1 model, comparison is disabled.

**Invariant 2 — Movie feedback is routed by `list_id`, not by `current_phase`.**

In sequential mode the participant always sees exactly one recommendation list with `list_id="recs-single"`, and `current_phase` advances from `0` to `1` to `2`, and so on. Audit lookup keys on `(approach_index, list_id)` and the two consistently agree.

In side-by-side mode the participant sees two lists with `list_id="recs-model-a"` and `list_id="recs-model-b"` simultaneously, and `current_phase` stays at 0 throughout. A naïve `approach_index = current_phase` (0 for both columns) breaks the audit:

- Recommendation set A is written with `approach_index=0, list_id="recs-model-a"` ✓
- Recommendation set B is written with `approach_index=1, list_id="recs-model-b"` ✓
- A like on column A looks up `(approach_index=0, list_id="recs-model-a")` ✓
- A like on column B looks up `(approach_index=0, list_id="recs-model-b")` — **does not exist**, so the audit raises `AuditContractError` and the like is lost.

The fix is in `record_movie_feedback`: a small canonical mapping resolves the column to the right approach.

```413:430:server/plugins/steering/service/audit.py
_COMPARISON_LIST_TO_APPROACH = {
    "recs-model-a": 0,
    "recs-model-b": 1,
}


def record_movie_feedback(
    data: dict,
    *,
    participation_id: int,
    approach_index: int,
    iteration: int,
) -> SaeMovieFeedback:
    ...
    if list_id in _COMPARISON_LIST_TO_APPROACH:
        approach_index = _COMPARISON_LIST_TO_APPROACH[list_id]
```

Regression test: `test_record_movie_feedback_routes_column_b_to_approach_1`. Without the remap, every like on column B would be discarded — side-by-side studies would then appear as if every participant unanimously preferred Approach A (because Approach B would carry zero recorded likes), silently corrupting the comparison the study was designed to make.

**Invariant 3 — Shared steering events fan out to both approaches.**

A side-by-side study has *one* slider grid driving *two* recommendation lists. When the participant moves a slider, that motion is semantically a steering action against **both** approaches at once — the same adjustment is fed into both recommenders. For the per-approach analytics (Modalities dashboard, slider-movement charts) to show non-empty data for the second approach, the audit row must be written for **each** approach run.

This is handled by a small helper that the iteration controller and the steering routes consult:

```292:320:server/plugins/steering/study_config.py
def get_audit_approach_indices(conf, current_phase: int) -> list:
    """Return the list of approach indices a single steering action affects."""
    ...
    if (
        conf.get("comparison_mode") == "side_by_side"
        and conf.get("enable_comparison")
        and len(models) >= 2
    ):
        return [0, 1]
    return [int(current_phase or 0)]
```

Sites that fan out:

| Audit call | File | What gets duplicated |
|---|---|---|
| `record_feature_adjustment` | `iteration_controller.py` | The slider/toggle adjustment map, once per approach. |
| `record_text_steering` | `actions.py::parse_text_steering` | The text prompt + matched clusters, once per approach. |
| `record_example_steering` | `actions.py::apply_example_steering` | The example-movie ids, once per approach. |
| `record_global_reset` | `actions.py::reset_steering` | The reset row, once per approach. |
| `record_movie_feedback` | `actions.py::log_movie_feedback` (via the `list_id` remap in `audit.py`) | **NOT fanned out** — a like is intrinsically per-column. |

Consequence: in side-by-side mode the steering-event count in the audit log is approximately $2\times$ the count of user interactions. This is honest and expected: the participant's single slider-move did happen against both approaches. The Modalities dashboard then displays symmetric, non-empty data for each approach.

**About the "select-on-both-columns" UX hack.**

The frontend renders one shared `likedMovies` map; `syncCardSelectionUI(movieId)` toggles every `.rec-card[data-movie-id="${movieId}"]` regardless of which column it lives in. So when a movie appears in both column A and column B (frequent — both lists are CF-ranked on the same seed and similar adjustments) and the participant clicks it on one side, both UIs light up.

This is safe **because each click still fires exactly one `log_movie_feedback` request**, carrying the `list_id` of the actual DOM card that was clicked. The mirror update is purely a UI affordance. Data-wise:

- Click on column-A card: one feedback row, `list_id="recs-model-a"`, `approach_index=0`.
- Click on column-B card: one feedback row, `list_id="recs-model-b"`, `approach_index=1` (after the remap).
- Toggling off (un-liking) on either column also fires one event with the correct `list_id`.

The mirror is therefore a visual sync, not an audit duplicate.

**Alternatives considered.**

- *Force every multi-approach study into sequential.* Rejected — side-by-side is a legitimate comparison design (same seed, two systems) that the proposal explicitly calls out. Dropping it would weaken the research package.
- *Add a third "shared" `approach_index = -1` and have analytics expand to all approaches on read.* Rejected — every read path in `analytics.py` would need a special case; the typed-audit schema would no longer be self-describing. The fan-out at write time keeps the read path uniform.
- *Schema migration to add a `shared_across_approaches` boolean on `SaeSteeringEvent`.* Rejected for two reasons: (1) the project has no migration framework — schema changes require a DB reset, which is too disruptive for a structural invariant; (2) the fan-out solution is already correct and adds no schema surface.

**Consequences.**

- The audit log size for side-by-side studies is roughly $2\times$ compared to a same-iteration sequential study with one approach. Researchers should size storage accordingly. Side-by-side studies are also capped at two approaches, so $2\times$ is the worst case.
- Phase-questionnaire wiring uses `current_phase=0`, so only approach A's phase questionnaire would be shown if both approaches were configured with `phase_questionnaire_file`. Side-by-side studies should therefore put their comparison questionnaire on the study-level `questionnaire_file` (final questionnaire). Per-approach phase questionnaires for side-by-side are not supported.
- Regression tests:
  - `test_get_audit_approach_indices_fans_out_side_by_side` — helper contract.
  - `test_record_movie_feedback_routes_column_b_to_approach_1` — Bug B1 regression.

## 23. Reranking strategies (`reranking_strategy` config key)

**Context.** Three strategies are available in this build: `feature-conditioned` (the production default), `latent-perturbation`, and `constrained-subset`. The math for each is documented in `equations.md` Section 10; this section captures the design choices behind the *set* of strategies and how they coexist.

**Decision.** Implement all three behind a single config key (`reranking_strategy`), with strategy-specific parameters (`latent_perturbation_alpha`, `constrained_subset_tau`) that fall back to constants when not supplied. The strategy enum is snapshotted on every `SaeApproachRun` so retrospective analyses can correctly attribute outcomes to the strategy that was active.

**Rationale.**

- The default (`feature-conditioned`) is the only strategy that has been piloted and that the dashboard analytics have been validated against. It must remain the default to keep production behaviour stable.
- The proposal commits to evaluating "latent-perturbation" and "constrained-optimisation" alternatives. Without an implementation those are vapourware. The two simple formulations in Section 10.2 and Section 10.3 are research-grade — they are defensible, mathematically clean, and use only objects that already exist in the runtime (`sae_model.decoder_w`, `item_features`).
- Routing the strategy through a single parameter on `get_recommendations` keeps the recommender's call sites uniform: every blend math change is internal to `sae_recommender.py`.

**Trade-offs.**

- `latent-perturbation` decodes through `W_{\text{dec}}^{\top}`, which assumes the SAE has been trained on the same ELSA embedding space the recommender uses. This invariant is guaranteed by the model loader (`recommender.sae_model` and `recommender.item_embeddings` come from the same paired snapshot in `model_store.py`), but it is worth flagging because swapping SAE checkpoints mid-study would silently break the assumption.
- `constrained-subset` can return fewer items than `n_items` when the $\tau$ filter is strict; the fallback ("drop the mask if no item survives") preserves availability but quietly downgrades the guarantee. The debug payload (`debug.constrained_subset_survivors`) records the survivor count so analysts can detect when the fallback fired.
- All three strategies share the same `cf_score`, `genre_score`, `steering_score` columns in the debug payload, which makes a head-to-head comparison straightforward in the dashboard.

**Alternatives considered.**

- *Pin to `feature-conditioned` and ship only the default.* Rejected — the proposal lists the alternatives as deliverables, and shipping vapourware enums (the previous state) is worse than implementing them.
- *Make the strategies pluggable via a strategy-pattern registry.* Rejected as overkill for three strategies that share most of their setup. A flat `if / elif / elif` inside `get_recommendations` is easier to read and to test in a paper context.
- *Expose a per-iteration strategy switch.* Rejected — strategies are study-design choices, not per-iteration tuning knobs. Changing strategy mid-study would corrupt cross-iteration comparisons.

**Consequences.**

- `SaeApproachRun.reranking_strategy` is populated from the iteration controller for every approach run and is queryable in retrospective analyses. Existing studies that pre-date this section still snapshot the value (they used `feature-conditioned`); no data migration is required.
- The admin UI dropdown (`sae_steering_create.html`) lists all three strategies with one-sentence descriptions and a deep link to `equations.md Section 10`.
- Regression tests:
  - `test_feature_conditioned_strategy_is_default`
  - `test_latent_perturbation_strategy_rotates_seed_and_drops_additive_term`
  - `test_constrained_subset_strategy_filters_out_non_conformant_items`
  - `test_constrained_subset_strategy_falls_back_when_no_survivors`
