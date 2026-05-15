# Researcher (Admin) Manual

This is the researcher-facing manual. For the participant perspective see [`user-manual.md`](user-manual.md); for design rationale see [`design-decisions.md`](design-decisions.md); for the full technical reference see [`tech-docs.md`](tech-docs.md).

## 1. Sign in

1. Visit the deployment root.
2. The landing page redirects to `/login`.
3. Log in with a researcher account. An admin-flagged account sees every study; a non-admin account sees only studies they created.

## 2. Create a study

The **Available templates** table lists the plugins you can pick as the parent for a new study. It is populated from `/loaded-plugins`, which filters the loaded plugin contracts down to those that (a) registered a `/<name>/create` endpoint and (b) did not opt out via `PluginMetadata.hidden_from_admin`. Two plugins are intentionally hidden: `empty_template` (developer scaffold — copy this when building a new plugin, see [`formative-examples.md`](formative-examples.md) Section 1) and `vae` (algorithm wrapper consumed by `fastcompare`, not a stand-alone study type). For design rationale see [`design-decisions.md`](design-decisions.md) Section 17.

1. From `/administration`, click **Create new study** and pick **SAE Steering** as the parent plugin.
2. You land on `/sae_steering/create`. Configure:

   **Dataset**
   - `MovieLens 32M Filtered` (8328 movies) is the only built-in option. Adding more is documented in [`formative-examples.md`](formative-examples.md) Section 3.

   **Approaches**
   One or more approach blocks. Each approach picks:
   - Base model (default `elsa`).
   - SAE model (default `topk_sae`).
   - **Feature controls** — `Sliders` (continuous), `Toggles` (boost / suppress), or `None`.
   - **Selection strength** — how strongly liked movies bias the recommender. See [`equations.md`](equations.md) Section 3.
   - **Feature-selection algorithm** — personalized grouped top-K or global label-diverse top-K.
   - Toggle weight (only when `Toggles`).
   - **Enable text prompts** (yes/no, FR-09). Adds NL steering on top of the other controls. Shown only when enabled: **Text top-K** and **Text steering composition** (`replace` / `add` / `intersect`, controls how iteration N+1 combines with iteration N — default `replace`). Each approach can pick its own composition rule, so the two arms of a study can be configured to compare stacking strategies. See [`design-decisions.md`](design-decisions.md) Section 6 and [`equations.md`](equations.md) Section 1.5.
   - **Show global reset control** (FR-12). Adds the dedicated reset button in the participant UI.
   - **Use selected movies as example-based steering** (FR-08). Drives the recommender from movies the participant already liked.

   **Reranking strategy** (FR-10)
   - `feature-conditioned` (default) — additive blend of CF + genre + adaptive γ · SAE score, with per-iteration clamping. This is the strategy every existing pilot used.
   - `latent-perturbation` — decode the SAE adjustment vector back to ELSA embedding space and rotate the user seed by `α · direction`; rank with pure CF on the rotated seed (no additive SAE term). Defensible "steering = user-profile shift" alternative; default `α = 0.30`.
   - `constrained-subset` — keep only candidates whose SAE score is at least `τ × max-positive-SAE`, then rank survivors by base CF + genre; falls back to base ranking if no item satisfies the constraint. Defensible "guaranteed on-target" alternative; default `τ = 0.25`.
   - The choice is captured on every `SaeApproachRun` row (`reranking_strategy` column) so retrospective analyses can correctly attribute outcomes to the active strategy. See [`equations.md`](equations.md) Section 10 for the math and [`design-decisions.md`](design-decisions.md) Section 23 for the rationale.

   **Interaction history mode**
   - `cumulative` (default) — within one approach, both the slider/toggle/text adjustment map and the like-weighted ELSA seed are carried across iterations. Touched sliders are filtered out of the next pool, the cumulative offsets stack, and likes from earlier iterations keep nudging the base recommender's seed.
   - `reset` — every iteration inside an approach is independent. At the start of iteration N+1 the iteration controller wipes `cumulative_adjustments`, `user_touched_features`, `boosted_liked_ids`, `last_text_steering`, `last_example_steering`, and `excluded_movies_from_text`, and refreshes the ELSA seed from the original preference-elicitation movies only (no like reweighting). Recommendations therefore recompute from the same baseline every round, while the typed-audit tables still record each iteration's adjustments, likes, recommendation sets, and resets.
   - Both modes still wipe the per-approach state when moving on to the next approach, so cross-approach pollution never happens (`_do_advance_phase` in `routes/study.py`).

   **Recommendations per approach** and **max iterations**
   - Typical values: 10 recommendations × 3 iterations.

   **Comparison mode**
   - `sequential` — each participant sees approaches one at a time, in their assigned (possibly randomised) order. There is exactly one recommendation column on screen; the `list_id` is always `recs-single`. Use this for any study with 1, 3, or more approaches.
   - `side_by_side` — two recommendation columns rendered next to each other for the same participant in the same iteration; one shared slider grid, one shared text input, one set of buttons. **Use only with exactly two approaches.** Studies with 3+ approaches that select `side_by_side` are silently downgraded to `sequential` (the UI has no third column).
   - Side-by-side records likes per column (`list_id="recs-model-a"` → approach 0; `list_id="recs-model-b"` → approach 1) and fans every shared steering action (slider move, text prompt, example pick, reset) out to both approach runs so the per-approach Modalities dashboard charts are symmetric. Phase-questionnaire wiring uses approach 0's `phase_questionnaire_file` only; for a between-approach comparison questionnaire, use the study-level `questionnaire_file` (final questionnaire). See [`design-decisions.md`](design-decisions.md) Section 22 for the full invariants and audit semantics.

   **Final questionnaire**
   - Upload a JSON questionnaire or use the bundled default.

   **Prolific completion code** (optional)

3. Click **Create study**. The server runs the long initialization (loads dataset caches, prepares SAE clusters).
4. Once `initialized = true` in `/user-studies`, set the study to **active** to start accepting participants.

## 3. Invite participants

- The join URL is shown in `/administration` for each study.
- Forward it directly or paste it into your Prolific completion-code chain.
- Participants land on `/sae_steering/join?guid=<study-guid>`.

## 4. Monitor a running study

- `/administration` lists all studies you can see with participant counts and an **active** toggle.
- `/existing-user-studies` returns the same data as JSON (login required).
- `/participations` returns the live participation rows (login required).

## 5. Inspect results

The researcher dashboard is at `/sae_steering/results?guid=<study-guid>` (login required). It has five tabs (FR-16):

### Overview

| Section | What it shows |
| --- | --- |
| Approach Overview table | Participants, mean iterations, mean abs adjustment, mean adjustment events, mean slider changes — one row per approach. |
| Selected Movie Ranks by Approach | Histogram of *like* events bucketed by the rank in the recommendation list, one series per approach. Tighter-to-the-top distributions mean steering pulled preferred movies higher. |
| Selection Dynamics | Per-approach likes / removals and the mean number of likes per participant. |
| Modality Usage by Approach | One card per approach showing only the modalities that approach declared in `enabled_modalities`. Each card lists per-modality counts (sliders → adjustments / distinct named clusters / mean \|Δ\|, text → prompts / distinct prompts / prompt→cluster matches, etc.) plus a `steering_mode` pill. See [design-decisions.md Section 20](design-decisions.md#20-the-modalities-dashboard-is-driven-by-enabled_modalities-not-by-audit-table-contents) for the contract. |

### Modalities

One section per approach. Each section renders only the cards that approach has enabled in `enabled_modalities`:

- **Feature movement by cluster** — top 15 named clusters by mean absolute Δ (horizontal bar). Rendered for approaches with `sliders` or `toggles` enabled.
- **Text prompt → cluster mappings** — top *(prompt, cluster)* pairs with mean signed weight and number of observations. Rendered for approaches with `text` enabled, scoped to that approach's submissions.

Approaches that don't enable any chartable modality (e.g. an approach using only `reset` as a utility action) show a short note explaining which modalities they did declare instead of an empty chart. If the whole study has no steerable modality (e.g. a purely observational study), the tab shows a single empty state.

### Questionnaires

One section per uploaded questionnaire file (per-approach files and the final file). Every key in the answers JSON is auto-detected as *likert* (1..7), *numeric*, *categorical*, or *text*, and aggregated accordingly. **Adding a new questionnaire is a no-code operation**: drop an HTML file in `server/static/questionnairs/`, point an approach at it from the create UI, and the monitor will pick it up after the next submission. `server/static/questionnairs/sae_sample_questionnaire.html` is the copy-paste template.

### Participants

One row per participation: Prolific PID + study/session ids, status, joined timestamp, duration, approach order, attention-check verdict, completion URL, journey link. When a participant didn't come through Prolific the study/session cell shows an em-dash instead of empty placeholder text.

**Attention-check verdict.** Each row aggregates attention-check pass/fail across every questionnaire the participant submitted. The verdict is computed once at submit time from the JSON spec inside the questionnaire HTML (see [design-decisions.md Section 18](design-decisions.md#18-attention-checks-are-declared-in-the-questionnaire-html-and-evaluated-at-submit-time)) and persisted as `SaeQuestionnaireResponse.attention_check_passed`. Use the **Attention-check threshold** input in the filter bar to set the minimum number of correct checks for a row to count as PASS; the threshold is per-study and remembered in your browser's `localStorage`. Submissions whose questionnaire file ships no spec show **no checks** and do not contribute to the ratio.

### Journey

Click **View →** in the participants table to open the journey for one participation. The timeline is rendered directly from the typed tables (no JSON parsing). Each questionnaire submission is listed with its file name, answer count and a per-submission PASS/FAIL badge (hover the FAIL badge to see which field was wrong); the full answers JSON is included in the journey response for inline inspection.

## 6. Export

| Endpoint | Use |
| --- | --- |
| `/sae_steering/export-raw/<guid>` | Per-participant JSON event log, mouse / viewport noise filtered. Use for payment reconciliation and journey reconstruction. |
| `/sae_steering/export-csv/<guid>` | **FR-17**: ZIP of one CSV per typed audit table. Use this for stats tools (R / pandas / Stata). Column headers match [`tech-docs.md`](tech-docs.md) Section 5.2 exactly. |

Both endpoints require login. The CSV ZIP contains 16 files; a recommended downstream pipeline is described in [`tech-docs.md`](tech-docs.md) Section 8.4.

## 7. Tear down a study

- `DELETE /user-study/<id>` (admin UI button) removes the study row, cascades to `Participation`, and lets each plugin tear down its files via the plugin's `/dispose` route.
- The SAE Steering `/dispose` endpoint clears cached recommender artifacts for that study.

## 8. Operational checks

- The healthz endpoint is `/healthz`.
- The DB backup endpoint is `/administration/db-backup` (admin only). It returns the most recent `db_*.gz` snapshot from `BACKUP_DIR` (default `/app/backups`).
- The session backend is SQLAlchemy-backed Flask-Session. To swap to Redis for >100 concurrent users (NFR-02), edit `server/platform/app.py::create_app`, set `app.config["SESSION_TYPE"] = "redis"`, and provide `SESSION_REDIS`. The architecture keeps the backend swappable.
- To reshape the schema (after editing a model), run `./scripts/reset-db.sh`. To bootstrap a fresh deployment, run `./scripts/init-db.sh`. There is no migration framework; see [`design-decisions.md`](design-decisions.md) Section 3.
