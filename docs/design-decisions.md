# Design Decisions

This page records the binding architectural decisions and the reasoning behind them. It is a companion to [`tech-docs.md`](tech-docs.md): the tech doc tells you *what* the system looks like, this one tells you *why*.

Each section follows the same shape: **Decision → Context → Alternatives considered → Consequences**.

---

## 1. Plugin-first extension of EasyStudy (rather than a fork)

**Decision.** The application is implemented as a new plugin (`server/plugins/steering/`) under upstream [pdokoupil/EasyStudy](https://github.com/pdokoupil/EasyStudy/tree/main/server). The platform half (`server/platform/`) is a thin reshuffle that preserves every upstream Flask blueprint, ORM model, and plugin contract.

**Context.** The proposal mandates an EasyStudy-based study framework, and consultants will keep evolving EasyStudy upstream. The thesis must be re-mergeable.

**Alternatives considered.**

- **Hard fork.** Rejected — it would lock out future upstream improvements (Prolific integration, questionnaire features, new auth flows) and require a manual rebase every time.
- **Branch on `pdokoupil/EasyStudy` directly.** Rejected — upstream code is owned by a different research group; opening a long-lived branch is fragile across versions and licensing.

**Consequences.**

- `Interaction` and `Message` ORM tables stay even though the steering plugin does not use them. Upstream `fastcompare` and `utils` plugins still log to them. Removing them would break upstream parity.
- The platform must not import from `server.plugins.*` at module top-level. Architectural rule #5 in [`tech-docs.md`](tech-docs.md) §4.4 enforces this.
- New thesis code lives under `server/plugins/steering/`. Other plugins (`fastcompare`, `empty_template`, `utils`) are kept verbatim.

---

## 2. Typed audit tables + thin envelope (no JSON event soup)

**Decision.** Every user action writes one **typed** row to a domain-specific table (e.g. `sae_feature_adjustment`) **and** one minimal `SaeSteeringEvent` envelope row that carries only ids and timestamps. Analytics joins the typed tables. `SaeSteeringEvent.raw_payload` is provenance only and is **never** read by analytics.

**Context.** The proposal mandates per-approach metrics (FR-16) and a CSV export per fact (FR-17). The initial prototype stored everything in a single `Interaction` table with a JSON `data` column. Each dashboard query had to parse JSON in Python; queries took seconds; column meanings drifted across iterations.

**Alternatives considered.**

- **Keep one event table with indexed JSON columns** (Postgres `jsonb` + GIN). Rejected — SQLite has no equivalent, and the column-shape contract was already implicit in the analytics code.
- **One typed table per action *type* but a generic `payload` blob next to typed columns.** Rejected as half-measure: every analytic query would still need to know which fields are typed and which are JSON.

**Consequences.**

- Adding a new fact requires three changes: a new model, a new `audit.record_*` function, a CSV writer in `routes/results/views.py`. Documented in [`formative-examples.md`](formative-examples.md) §4.
- The `raw_payload` column stays as a provenance escape hatch for journey rendering and manual debugging. It is intentionally not indexed and not in any dashboard query.
- Per-table CSV export is trivial: each writer reads one typed table and emits one CSV.

---

## 3. Models are the single source of truth — no migration framework

**Decision.** `server/platform/persistence/base_models.py` and each plugin's `persistence/models.py` are the only source of truth for the schema. `db.create_all()` runs on every boot. There is no Alembic, no `migrations/` directory, no `flask db upgrade`.

**Context.** Earlier iterations used Flask-Migrate / Alembic. After a Phase 2 refactor reshaped half of the SAE tables, the Alembic flow broke in two ways:

1. `db.create_all()` ran first inside `create_app()`, creating the new tables. Then `flask db upgrade` tried to `CREATE TABLE` for the same tables and failed with `OperationalError: table sae_feature_adjustment already exists`.
2. Because the migration aborted mid-way, the `ALTER TABLE sae_steering_event ADD COLUMN raw_payload` step never ran, and every subsequent request crashed with `OperationalError: ... has no column named raw_payload`.

We also discovered two `migrations/` directories existed (`migrations/` and a stale leftover `server/migrations/`).

**Alternatives considered.**

- **Make Alembic the single source of truth.** Rejected — autogenerate is unreliable across SQLite and Postgres (SQLite has no `ALTER TABLE DROP COLUMN`), and the migration history would have to be rebuilt from scratch.
- **Keep Alembic but stamp it on every boot.** Rejected — that is exactly the half-working state we just fixed; it embeds a "fallback" code path the user explicitly forbade.

**Consequences.**

- `./scripts/init-db.sh` is idempotent: `db.create_all()` adds missing tables and leaves existing ones alone.
- `./scripts/reset-db.sh` is destructive: `db.drop_all()` then `db.create_all()`. Use whenever a model is reshaped during development.
- Production schema changes happen out of band (manual `ALTER TABLE` against the managed Postgres) before deploying the new build. This is acceptable because the thesis runs a fixed number of studies and schema changes are rare research-side decisions, not continuous deployments.
- Reintroducing Alembic is a deliberate, opt-in decision when the application moves out of thesis scope and into a long-lived production service.

---

## 4. Single-writer audit service; routes own the session

**Decision.** Only `server/plugins/steering/service/audit.py` writes to typed audit tables. All `record_*` functions take `participation_id` and `approach_index` as **keyword-only** arguments. The service module never reads `flask.session`. Routes pass session values in explicitly.

**Context.** Earlier code had `audit.record_event` reading from `flask.session` directly, plus a self-rebuilding `ensure_study_run` that read `session["approach_order"]` and called helpers that *also* called `ensure_study_run`, producing a `RecursionError` and intermittent `UNIQUE constraint failed: sae_study_run.participation_id` errors when the order had not been seeded yet.

**Alternatives considered.**

- **Service reads session, routes are thin.** Rejected because of the recursion above and because it forces tests to stand up a `flask.request_context` for unit-level audit assertions.
- **No service layer; routes write rows directly.** Rejected because it would distribute the audit contract across ~20 route handlers and make consistent envelope writing impossible.

**Consequences.**

- The service is unit-testable without a request context.
- The recursion is gone: `ensure_study_run` is called once per request, and `audit.record_*` callers pass `approach_index` explicitly so `ensure_approach_run` is idempotent.
- Adding a new typed table is a one-place change ([`formative-examples.md`](formative-examples.md) §4).

---

## 5. Reset is a dedicated endpoint, not a flag on `/adjust-features`

**Decision.** `POST /sae_steering/reset` is its own endpoint. It writes one `SaeResetAction` row + one envelope and clears session state. The legacy `reset_all: true` flag inside `/adjust-features` payloads is gone.

**Context.** FR-12 in the proposal is "single-click reset, recorded". The original implementation smuggled `reset_all=true` and `reset_reason="..."` into the `/adjust-features` JSON body and let the iteration controller branch internally. This produced confusing audit rows where a reset looked like a degenerate adjustment, and the analytics had to special-case `applied_via='reset'` rows.

**Alternatives considered.**

- **Keep the flag, but write a distinct `SaeResetAction` row alongside the adjustment.** Rejected — the route still does two things and the dashboard query still has to coalesce.
- **`DELETE /sae_steering/adjustments`.** Rejected — `DELETE` over a collection that doesn't map to a REST resource was harder to document; the participant action is "reset", not "delete adjustments".

**Consequences.**

- The "Reset all controls" button POSTs to `/reset`. The iteration controller no longer reads `reset_all`.
- Analytics counts `sae_reset_action` rows directly. No coalescing.
- Tests cover the contract: see `test_steering_actions_and_security.py::test_reset_writes_one_audit_row_and_clears_session`.

---

## 6. Text steering composition is a configurable mode (FR-09)

**Decision.** Each study chooses how successive NL prompts compose: `replace` (default), `add`, or `intersect`. The composed adjustments are returned by `/parse-text-steering` and persisted as `SaeApproachRun.composition_mode` snapshot.

**Context.** Pilot feedback showed two patterns: participants who liked giving one description and being done (matches `replace`), and participants who built up intent iteratively ("I like Marvel" → "but only the comedies", matches `add` or `intersect`). The previous behaviour was implicit `replace`, which felt brittle for the iterative pattern.

**Alternatives considered.**

- **Always `add`.** Rejected — too easy to drift toward all-clusters-active.
- **Heuristically detect "but" / "however" and switch mode.** Rejected — fragile, and not in the participant's mental model.

**Consequences.**

- Three explicit modes, validated in `study_config.py::normalize_text_composition_mode`.
- `add` mode clamps each cluster sum to `[-0.95, 0.95]` so weights stay in the recommender's effective range.
- `intersect` mode keeps only clusters present in both iterations, using iteration N's weight. Useful for participants refining a stable intent.
- See [`equations.md`](equations.md) §1.5 for the formal definition.

---

## 7. NFR-12: text-steering ambiguity degrades gracefully

**Decision.** When the lexical resolver matches zero clusters, `/parse-text-steering` returns HTTP 200 with `status="no-match"` and a friendly message. A `SaeTextSteeringQuery` row is still written (with zero matches), so the failure mode is analyzable offline.

**Context.** NFR-12 (graceful degradation) is in the proposal. The pilot exposed cases where participants wrote off-topic prompts ("I don't know what I want") that produced empty match sets. The earlier implementation returned an empty `features` array silently; participants kept clicking and got confused.

**Consequences.**

- The UI renders an alert: "We could not match your text to any feature, try different wording".
- Analytics can count zero-match queries (`COUNT(*) FROM sae_text_steering_query WHERE NOT EXISTS (SELECT 1 FROM sae_text_steering_match WHERE query_id = q.id)`).
- The participant's previous adjustments are preserved (no destructive write on empty match).

---

## 8. Reranking strategy as a forward-compatible enum (FR-10)

**Decision.** `study_config.reranking_strategy` is an enum with three valid values: `feature-conditioned` (implemented), `latent-perturbation` (reserved), `constrained-opt` (reserved). The iteration controller validates the enum and only branches into `feature-conditioned`.

**Context.** The proposal lists three reranking strategies. Implementing all three is out of thesis scope, but binding the schema to only `feature-conditioned` would force a migration when the others land. We instead validate the enum at config-normalization time and snapshot it onto `SaeApproachRun.reranking_strategy`, so historical study runs carry their strategy correctly.

**Consequences.**

- `SaeApproachRun.reranking_strategy` is a string column, snapshotting the choice per approach per participant.
- The two reserved values pass `study_config` validation but raise `NotImplementedError` if a route would dispatch into them. Production studies must select `feature-conditioned`.
- Adding the third value later requires zero schema work — only a new branch in `iteration_controller.py`.

---

## 9. Researcher endpoints require login; participant endpoints do not

**Decision.** Every researcher-facing endpoint carries `@login_required`. Participant endpoints (`/join`, `/preference-elicitation`, `/parse-text-steering`, `/adjust-features`, `/reset`, ...) do **not** because participants are anonymous from a Flask-Login perspective; their identity is the session `participation_id`.

**Context.** The earlier code had `/loaded-plugins`, `/existing-user-studies`, `/user-study`, `/user-study-participants`, and `/results/<plugin>/<guid>` open. Anyone with the URL could enumerate studies. The CSV export was also open.

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

**Decision.** The dashboard only knows about the typed audit columns (rank, delta, cluster_label, query_text, mood, …). It must **not** branch on question-level ids like `p_attention_check` or `f1_preference`. Per-questionnaire summaries are produced by the modular *Questionnaire Monitor*, which auto-discovers fields from `SaeQuestionnaireResponse.answers` and infers a kind per field.

**Context.** The first iteration of the dashboard wired the Likert deltas, attention-funnel, and preference distribution charts to specific question ids from the bundled `sae_*_questionnaire.html` files. Whenever a researcher uploaded a different questionnaire the charts broke silently — or worse, the analytics pretended a result existed by reading missing keys.

**Consequences.**

- `analytics._questionnaire_monitor` aggregates one section per `questionnaire_file`. Within each section, every key in the answers JSON becomes a row whose summary is chosen from the inferred kind (`likert` / `numeric` / `categorical` / `text`).
- The Overview tab focuses on the *behavioural* signal of steering: selected-movie rank distribution by approach, slider movement by cluster, text-prompt → cluster mappings. None of these depend on a specific questionnaire.
- Adding a new questionnaire is a no-code operation (drop a template in `server/static/questionnairs/`, point an approach at it). See [`formative-examples.md`](formative-examples.md) §9.
- The legacy "attention-check funnel" / "Steered − Baseline" / "composite delta" charts are deleted. Researchers who want those specific statistics can compute them in R or pandas from the CSV bundle.

---

## 14. Every typed audit row carries an `event_id` FK

**Decision.** Every typed table that records a participant action has an `event_id` FK to `sae_steering_event`. The audit service always writes the envelope **first** and then attaches the typed row using `event.id`.

**Context.** All typed tables (`SaeFeatureAdjustment`, `SaeTextSteeringQuery`, `SaeFeatureSearch`, `SaeResetAction`, `SaeExampleSteering`) followed this pattern from day one — except `SaeMovieFeedback`, which was the first table written and accidentally did the reverse (feedback row → envelope) with the feedback id stored in `event.raw_payload`. The journey view tried to use `feedback.event_id`, which didn't exist, and the page 500'd.

**Consequences.**

- `SaeMovieFeedback.event_id` is `NOT NULL` and CASCADEs from the envelope.
- `audit.record_movie_feedback` now writes the envelope first and the feedback row inherits `event.study_run_id` / `approach_run_id` / `event_id` for full provenance.
- The journey view, raw export, and CSV export all surface `event_id` for movie feedback.
- Future typed tables MUST follow this order. The audit service docstring is the contract.
