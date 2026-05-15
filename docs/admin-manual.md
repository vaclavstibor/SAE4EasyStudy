# Admin (Researcher) Manual

This is the researcher-facing manual. For the participant perspective see [`user-manual.md`](user-manual.md); for design rationale see [`design-decisions.md`](design-decisions.md); for the full technical reference see [`tech-docs.md`](tech-docs.md).

## Contents

1. [Sign in](#1-sign-in)  
   Access the admin UI.
2. [Create a study](#2-create-a-study)  
   Configure approaches, modalities, reranking, and questionnaires.
3. [Invite participants](#3-invite-participants)  
   Distribute the join link (Prolific or direct).
4. [Monitor a running study](#4-monitor-a-running-study)  
   Live participation table and status.
5. [Inspect results](#5-inspect-results)  
   Dashboard tabs, attention checks, journey view.
6. [Export](#6-export)  
   Raw JSON and typed CSV ZIP exports.
7. [Tear down a study](#7-tear-down-a-study)  
   Dispose and cascade behaviour.
8. [Operational checks](#8-operational-checks)  
   Healthz, backups, and session notes.

## 1. Sign in

1. Visit the deployment root.
2. The landing page redirects to `/login`.
3. Log in with an admin account if you have one, otherwise sign up with a new account. Each account has its own set of studies and can only see and manage its own studies.

## 2. Create a study

The **Available templates** table lists the plugins you can pick as the parent for a new study. It is populated from `/loaded-plugins`, which filters the loaded plugin contracts down to those that (a) registered a `/<name>/create` endpoint and (b) did not opt out via `PluginMetadata.hidden_from_admin`. Two plugins are intentionally hidden: `emptytemplate` (folder `server/plugins/empty_template`, developer scaffold — copy this when building a new plugin, see [`formative-examples.md` Section 1](formative-examples.md#1-add-a-new-plugin)) and `vae` (algorithm wrapper consumed by `fastcompare`, not a stand-alone study type). For design rationale see [`design-decisions.md` Section 17](design-decisions.md#17-admin-available-templates-is-filtered-by-pluginmetadatahiddenfromadmin).

1. From `/administration`, click **Create new study** and pick **SAE Steering** as the parent plugin.
2. You land on `/sae_steering/create`. Configure the study in the groups below.

### Dataset

`MovieLens 32M Filtered` (8328 movies) is the only built-in option. Adding more is documented in [`formative-examples.md` Section 3](formative-examples.md#3-add-a-new-dataset).

### Approach configuration

A study can define one or more approach blocks. Each block describes one recommender variant shown to a participant.

| Field | Meaning |
| --- | --- |
| Base model | Default `elsa`. |
| SAE model | Default checkpoint id `TopKSAE-1024` (UI label may show the human-readable name). |
| Feature controls | `Sliders` (continuous), `Toggles` (boost / suppress), or `None`. |
| Selection strength | How strongly liked movies bias the recommender. See [`equations.md` Section 7](equations.md#7-elsa-seed-re-weighting-from-likes). |
| Feature-selection algorithm | Personalized grouped `top-K` or global label-diverse `top-K`. |
| Toggle weight | Used only when `Toggles` are enabled. |
| Text prompts (FR-09) | Optional NL steering on top of the other controls. When enabled, configure **Text `top-K`** and **Text steering composition** (`replace` / `add` / `intersect`). The composition controls how iteration $N+1$ combines with iteration $N$; default is `replace`. See [`design-decisions.md` Section 6](design-decisions.md#6-text-steering-composition-is-a-configurable-mode-fr-09) and [`equations.md` Section 4.5](equations.md#45-top-k-and-composition-across-iterations). |
| Global reset (FR-12) | Shows the dedicated reset button in the participant UI. |
| Example-based steering (FR-08) | Uses movies the participant already liked as an additional steering signal. |

### Reranking strategy (FR-10)

| Strategy | Behaviour |
| --- | --- |
| `feature-conditioned` (default) | Additive blend of $CF + genre + \gamma \cdot SAE\ score$, with per-iteration clamping. This is the strategy every existing pilot used. |
| `latent-perturbation` | Decodes the SAE adjustment vector back to ELSA embedding space and rotates the user seed by $\alpha \cdot direction$; ranks with pure CF on the rotated seed (no additive SAE term); Default $\alpha = 0.30$. |
| `constrained-subset` | Keeps only candidates whose SAE score is at least $\tau \times max\text{-}positive\text{-}SAE$, then ranks survivors by base CF + genre; falls back to base ranking if no item satisfies the constraint; default $\tau = 0.25$. |

The choice is captured on every `SaeApproachRun` row (`reranking_strategy` column) so retrospective analyses can correctly attribute outcomes to the active strategy. See [`equations.md` Section 10](equations.md#10-reranking-strategies-rerankingstrategy-config-key) for the math and [`design-decisions.md` Section 23](design-decisions.md#23-reranking-strategies-rerankingstrategy-config-key) for the rationale.

### Interaction history mode

| Mode | Behaviour |
| --- | --- |
| `cumulative` (default) | Within one approach, both the slider/toggle/text adjustment map and the like-weighted ELSA seed are carried across iterations. Touched sliders are filtered out of the next pool, the cumulative offsets stack, and likes from earlier iterations keep nudging the base recommender's seed. |
| `reset` | Every iteration inside an approach is independent. At the start of iteration $N+1$ the iteration controller wipes `cumulative_adjustments`, `user_touched_features`, `boosted_liked_ids`, `last_text_steering`, `last_example_steering`, and `excluded_movies_from_text`, and refreshes the ELSA seed from the original preference-elicitation movies only (no like reweighting). Recommendations therefore recompute from the same baseline every round, while the typed-audit tables still record each iteration's adjustments, likes, recommendation sets, and resets. |

Both modes still wipe the per-approach state when moving on to the next approach, so cross-approach pollution never happens (`_do_advance_phase` in `routes/study.py`).

### Iterations and comparison mode

Typical values are `10` recommendations across `3` iterations.

| Mode | When to use it |
| --- | --- |
| `sequential` | Each participant sees approaches one at a time, in their assigned (possibly randomised) order. There is exactly one recommendation column on screen; the `list_id` is always `recs-single`. Use this for any study with 1, 3, or more approaches. |
| `side_by_side` | Two recommendation columns rendered next to each other for the same participant in the same iteration; one shared slider grid, one shared text input, one set of buttons. **Use only with exactly two approaches.** Studies with 3+ approaches that select `side_by_side` are silently downgraded to `sequential` (the UI has no third column). |

Side-by-side records likes per column (`list_id="recs-model-a"` maps to approach `0`; `list_id="recs-model-b"` maps to approach `1`) and fans every shared steering action (slider move, text prompt, example pick, reset) out to both approach runs so the per-approach Modalities dashboard charts are symmetric. **Sequential** studies: after each approach, the participant gets that approach model’s `phase_questionnaire_file` (or the study-level `phase_questionnaire_file` fallback when the model omits it). **Side-by-side** studies: there is no per-approach phase questionnaire between the two columns mid-task — put between-system comparison items on the study-level **Final questionnaire** (`questionnaire_file`). See [`design-decisions.md` Section 22](design-decisions.md#22-side-by-side-comparison-invariants-audit-fan-out-listid-routing).

### Questionnaires and completion

For each approach, the **Approach Questionnaires** stack lets you pick the bundled **implicit** or **explicit** feedback HTML template, or upload a custom `.html` file. That choice is stored in config as `phase_questionnaire_file` and is **not** overwritten when you later change feature controls or steering mode in the form — switch the radio explicitly if you want a different template.

Use the **Final questionnaire** field to upload an end-of-study file or keep the bundled default. Use the optional **Prolific completion code** field if you recruit participants through Prolific.

After configuring the form, click **Create study**. The server runs initialization (loads dataset caches, prepares SAE clusters). Once `initialized = true` in `/user-studies`, set the study to **active** to start accepting participants.

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


| Section                          | What it shows                                                                                                                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Approach Overview table          | Participants, mean iterations, mean abs adjustment, mean adjustment events, mean slider changes — one row per approach.                                                                        |
| Selected Movie Ranks by Approach | Histogram of *like* events bucketed by the rank in the recommendation list, one series per approach. Tighter-to-the-top distributions mean steering pulled preferred movies higher.            |
| Selection Dynamics               | Per-approach likes / removals and the mean number of likes per participant.                                                                                                                    |
| Modality Usage by Approach       | One card per approach showing only the modalities declared in `enabled_modalities`, with per-modality counts such as slider adjustments, distinct named clusters, mean $|\Delta|$, prompts and prompt-to-cluster matches. |


### Modalities

One section per approach. Each section renders only the cards that approach has enabled in `enabled_modalities`:

- **Feature movement by cluster** — top `15` named clusters by mean absolute $|\Delta|$ (horizontal bar). Rendered for approaches with `sliders` or `toggles` enabled.
- **Text prompt to cluster mappings** — top *(prompt, cluster)* pairs with mean signed weight and number of observations. Rendered for approaches with `text` enabled, scoped to that approach's submissions.

Approaches that don't enable any chartable modality (e.g. an approach using only `reset` as a utility action) show a short note explaining which modalities they did declare instead of an empty chart. If the whole study has no steerable modality (e.g. a purely observational study), the tab shows a single empty state.

### Questionnaires

One section per uploaded questionnaire file (per-approach files and the final file). Every key in the answers JSON is auto-detected as *likert* (1..7), *numeric*, *categorical*, or *text*, and aggregated accordingly. **Adding a new questionnaire is a no-code operation**: drop an HTML file in `server/static/questionnairs/`, point an approach at it from the create UI, and the monitor will pick it up after the next submission. `server/static/questionnairs/sae_sample_questionnaire.html` is the copy-paste template.

### Participants

One row per participation: Prolific PID + study/session ids, status, joined timestamp, duration, approach order, attention-check verdict, completion URL, journey link. When a participant didn't come through Prolific the study/session cell shows an em-dash instead of empty placeholder text.

**Attention-check verdict.** Each row aggregates attention-check pass/fail across every questionnaire the participant submitted. The verdict is computed once at submit time from the JSON spec inside the questionnaire HTML (see [design-decisions.md Section 18](design-decisions.md#18-attention-checks-are-declared-in-the-questionnaire-html-and-evaluated-at-submit-time)) and persisted as `SaeQuestionnaireResponse.attention_check_passed`. Use the **Attention-check threshold** input in the filter bar to set the minimum number of correct checks for a row to count as PASS; the threshold is per-study and remembered in your browser's `localStorage`. Submissions whose questionnaire file ships no spec show **no checks** and do not contribute to the ratio.

### Journey

Click **View** in the participants table to open the journey for one participation. The timeline is rendered directly from the typed tables. Each questionnaire submission is listed with its file name, answer count and a per-submission PASS/FAIL badge (hover the FAIL badge to see which field was wrong); the full answers JSON is included in the journey response for inline inspection.

## 6. Export


| Endpoint                          | Use                                                                                                                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `/sae_steering/export-raw/<guid>` | Per-participant JSON event log. Use for additional analysis and journey reconstruction.                                                                                  |
| `/sae_steering/export-csv/<guid>` | **FR-17**: ZIP of one CSV per typed audit table. Use this for stats tools (R / pandas / Stata). The CSVs are emitted from the typed ORM models; Section 5.2 in [`tech-docs.md`](tech-docs.md#52-sae-steering-tables-plugin-owned) is the canonical schema reference. |


Both endpoints require login. The CSV ZIP contains 16 files; a recommended downstream pipeline is described in [`tech-docs.md` Section 8.4](tech-docs.md#84-fr-17-csv-export).

## 7. Tear down a study

- `DELETE /user-study/<id>` (admin UI button) removes the study row, cascades to `Participation`, and lets each plugin tear down its files via the plugin's `/dispose` route.
- The SAE Steering `/dispose` endpoint clears cached recommender artifacts for that study.

## 8. Operational checks

- The healthz endpoint is `/healthz`.
- The DB backup endpoint is `/administration/db-backup` (admin only). Each click creates a fresh snapshot on the server and streams it back as `db_<UTC>.{sql,sqlite}.gz`. Files are kept under the directory resolved by `server.platform.shared.common.resolve_backup_dir()`: `BACKUP_DIR` if set, otherwise `<repo_root>/backups` (which is `/app/backups` inside the Docker image; on Railway the entrypoint symlinks that to `${DATA_ROOT}/backups` so backups survive redeploys). Rolling retention keeps the most recent `KEEP_LAST` (default `14`) archives. The same `server/scripts/backup_db.py` can be run as a CLI for unattended snapshots.
- The session backend is SQLAlchemy-backed Flask-Session. To swap to Redis for >100 concurrent users (NFR-02), edit `server/platform/app.py::create_app`, set `app.config["SESSION_TYPE"] = "redis"`, and provide `SESSION_REDIS`. The architecture keeps the backend swappable. Deployments in this project have run with tens of concurrent participants on SQLAlchemy sessions without issues; Redis is documented as the scale-out path if you need it.
- To reshape the schema (after editing a model), run `./scripts/reset-db.sh`. To bootstrap a fresh deployment, run `./scripts/init-db.sh`. There is no migration framework; see [`design-decisions.md` Section 3](design-decisions.md#3-models-are-the-single-source-of-truth-no-migration-framework).

