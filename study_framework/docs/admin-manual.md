# Researcher (Admin) Manual

This is the researcher-facing manual. For the participant perspective see [`user-manual.md`](user-manual.md); for design rationale see [`design-decisions.md`](design-decisions.md); for the full technical reference see [`tech-docs.md`](tech-docs.md).

## 1. Sign in

1. Visit the deployment root.
2. The landing page redirects to `/login`.
3. Log in with a researcher account. An admin-flagged account sees every study; a non-admin account sees only studies they created.

## 2. Create a study

1. From `/administration`, click **Create new study** and pick **SAE Steering** as the parent plugin.
2. You land on `/sae_steering/create`. Configure:

   **Dataset**
   - `MovieLens 32M Filtered` (8328 movies) is the only built-in option. Adding more is documented in [`formative-examples.md`](formative-examples.md) §3.

   **Approaches**
   One or more approach blocks. Each approach picks:
   - Base model (default `elsa`).
   - SAE model (default `topk_sae`).
   - **Feature controls** — `Sliders` (continuous), `Toggles` (boost / suppress), or `None`.
   - **Selection strength** — how strongly liked movies bias the recommender. See [`equations.md`](equations.md) §3.
   - **Feature-selection algorithm** — personalized grouped top-K or global label-diverse top-K.
   - Toggle weight (only when `Toggles`).
   - **Enable text prompts** (yes/no, FR-09). Adds NL steering on top of the other controls. Shown only when enabled: **Text top-K** and **Text steering composition** (`replace` / `add` / `intersect`, controls how iteration N+1 combines with iteration N — default `replace`). Each approach can pick its own composition rule, so the two arms of a study can be configured to compare stacking strategies. See [`design-decisions.md`](design-decisions.md) §6 and [`equations.md`](equations.md) §1.5.
   - **Show global reset control** (FR-12). Adds the dedicated reset button in the participant UI.
   - **Use selected movies as example-based steering** (FR-08). Drives the recommender from movies the participant already liked.

   **Reranking strategy** (FR-10)
   - `feature-conditioned` is implemented. `latent-perturbation` and `constrained-opt` are reserved enum values and not enabled in this build. See [`design-decisions.md`](design-decisions.md) §8.

   **Interaction history mode**
   - `cumulative` (default) — adjustments persist across iterations.
   - `reset-each-iteration` — adjustments start from zero each round.

   **Recommendations per approach** and **max iterations**
   - Typical values: 10 recommendations × 3 iterations.

   **Comparison mode**
   - `sequential` — each participant sees approaches in their assigned order.
   - `side_by_side` — only used with exactly two approaches.

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
| Structured Steering Events | Modality usage counts and integrity check (resets, text queries, example events, recommendation impressions, search-then-adjust). |

### Modalities

- **Slider movement by cluster** — top 15 most-moved clusters per approach (horizontal bar). Useful for "which controls do participants reach for?".
- **Text prompt → cluster mappings** — top 50 *(prompt, cluster)* pairs with mean signed weight and number of observations. Useful for "did the NL parser route this prompt sensibly?".

### Questionnaires

One section per uploaded questionnaire file (per-approach files and the final file). Every key in the answers JSON is auto-detected as *likert* (1..7), *numeric*, *categorical*, or *text*, and aggregated accordingly. **Adding a new questionnaire is a no-code operation**: drop an HTML file in `server/static/questionnairs/`, point an approach at it from the create UI, and the monitor will pick it up after the next submission. `server/static/questionnairs/sae_sample_questionnaire.html` is the copy-paste template.

### Participants

One row per participation: Prolific PID + study/session ids, status, joined timestamp, duration, approach order, number of questionnaire submissions, completion URL, journey link. When a participant didn't come through Prolific the study/session cell shows an em-dash instead of empty placeholder text.

### Journey

Click **View →** in the participants table to open the journey for one participation. The timeline is rendered directly from the typed tables (no JSON parsing). Each questionnaire submission is also listed with its file name and answer count; the full answers JSON is included in the journey response for inline inspection.

## 6. Export

| Endpoint | Use |
| --- | --- |
| `/sae_steering/export-raw/<guid>` | Per-participant JSON event log, mouse / viewport noise filtered. Use for payment reconciliation and journey reconstruction. |
| `/sae_steering/export-csv/<guid>` | **FR-17**: ZIP of one CSV per typed audit table. Use this for stats tools (R / pandas / Stata). Column headers match [`tech-docs.md`](tech-docs.md) §5.2 exactly. |

Both endpoints require login. The CSV ZIP contains 16 files; a recommended downstream pipeline is described in [`tech-docs.md`](tech-docs.md) §8.4.

## 7. Tear down a study

- `DELETE /user-study/<id>` (admin UI button) removes the study row, cascades to `Participation`, and lets each plugin tear down its files via the plugin's `/dispose` route.
- The SAE Steering `/dispose` endpoint clears cached recommender artifacts for that study.

## 8. Operational checks

- The healthz endpoint is `/healthz`.
- The DB backup endpoint is `/administration/db-backup` (admin only). It returns the most recent `db_*.gz` snapshot from `BACKUP_DIR` (default `/app/backups`).
- The session backend is SQLAlchemy-backed Flask-Session. To swap to Redis for >100 concurrent users (NFR-02), edit `server/platform/app.py::create_app`, set `app.config["SESSION_TYPE"] = "redis"`, and provide `SESSION_REDIS`. The architecture keeps the backend swappable.
- To reshape the schema (after editing a model), run `./scripts/reset-db.sh`. To bootstrap a fresh deployment, run `./scripts/init-db.sh`. There is no migration framework; see [`design-decisions.md`](design-decisions.md) §3.
