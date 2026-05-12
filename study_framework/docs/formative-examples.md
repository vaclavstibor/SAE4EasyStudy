# Formative Examples — Extending the Framework

This page is the *how-to* companion to [`tech-docs.md`](tech-docs.md) and [`design-decisions.md`](design-decisions.md). It collects worked recipes for the things future contributors will most often want to do.

Every recipe is grounded in the real code. File paths are relative to `study_framework/`. Where a recipe references a binding decision, it links back to [`design-decisions.md`](design-decisions.md).

| Recipe | Section |
| --- | --- |
| Add a new plugin (full skeleton) | §1 |
| Add a new steering modality | §2 |
| Add a new dataset | §3 |
| Add a new typed audit table | §4 |
| Add a new reranking strategy | §5 |
| Add a new researcher dashboard metric | §6 |
| Add a new participant-facing endpoint | §7 |
| Add a new CSV file to the export | §8 |
| Write tests for the above | §9 |

---

## 1. Add a new plugin

The framework's plugin contract is documented in [`tech-docs.md`](tech-docs.md) §4.2. A plugin is any Python package under `server/plugins/` that exposes `get_plugin() -> StudyPluginContract` from its `__init__.py`.

The simplest possible plugin is `empty_template` (kept verbatim from upstream EasyStudy). The simplest *new* SAE-derivative plugin is the one we already built. Here is the skeleton you would produce for a new plugin called `mystudy`.

### 1.1 Directory layout

```
server/plugins/mystudy/
├── __init__.py             from .plugin import PLUGIN, get_plugin
├── constants.py            PLUGIN_NAME, PLUGIN_VERSION, ...
├── plugin.py               Blueprint + StudyPluginContract
├── routes/
│   ├── __init__.py
│   └── study.py            /create, /initialize, /dispose, /join, /results
├── persistence/
│   ├── __init__.py
│   └── models.py           MystudyTrial(db.Model) + any other typed tables
├── templates/
│   ├── mystudy_create.html
│   └── mystudy_results.html
└── service/
    ├── __init__.py
    └── audit.py            record_* writers for the plugin's typed tables
```

### 1.2 `constants.py`

```python
PLUGIN_NAME = "mystudy"
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "Your Name"
PLUGIN_AUTHOR_CONTACT = "you@example.org"
PLUGIN_DESCRIPTION = "One-line description that shows up on /administration."
```

### 1.3 `plugin.py`

This is the canonical contract surface, mirroring `server/plugins/steering/plugin.py`:

```python
from flask import Blueprint

from server.platform.runtime import PluginMetadata, StudyPluginContract

from .constants import (
    PLUGIN_AUTHOR,
    PLUGIN_AUTHOR_CONTACT,
    PLUGIN_DESCRIPTION,
    PLUGIN_NAME,
    PLUGIN_VERSION,
)

bp = Blueprint(
    PLUGIN_NAME,
    __name__,
    url_prefix=f"/{PLUGIN_NAME}",
    template_folder="templates",
    static_folder="static",
)

from .routes import study  # noqa: E402,F401  (registers @bp.route handlers)


PLUGIN = StudyPluginContract(
    metadata=PluginMetadata(
        name=PLUGIN_NAME,
        version=PLUGIN_VERSION,
        author=PLUGIN_AUTHOR,
        author_contact=PLUGIN_AUTHOR_CONTACT,
        description=PLUGIN_DESCRIPTION,
    ),
    blueprint=bp,
    persistence_hooks={
        "models_module": "server.plugins.mystudy.persistence.models",
    },
)


def get_plugin():
    return PLUGIN
```

### 1.4 Register the plugin

Add the module path to `server/platform/runtime/plugin_registry.py`:

```python
CANONICAL_PLUGIN_MODULES = [
    "server.plugins.steering",
    "server.plugins.fastcompare",
    "server.plugins.empty_template",
    "server.plugins.mystudy",        # <-- new
]
```

The platform will:

1. Import `server.plugins.mystudy.persistence.models` (because `persistence_hooks["models_module"]` points there). This is how SQLAlchemy discovers your tables before `db.create_all()` runs.
2. Register the blueprint at `/mystudy/*`.
3. List your plugin on `/administration` as a parent-plugin choice on the create-study page.

### 1.5 Required endpoints

The plugin must implement five endpoints (`tech-docs.md` §4.2). Minimal skeleton:

```python
# server/plugins/mystudy/routes/study.py
from flask import jsonify
from flask_login import login_required

from ..plugin import bp


@bp.route("/create")
@login_required
def create():
    return "mystudy create page"


@bp.route("/initialize")
@login_required
def initialize():
    return jsonify({"status": "ok"})


@bp.route("/dispose", methods=["DELETE"])
@login_required
def dispose():
    return "OK"


@bp.route("/join")
def join():
    return "mystudy join page"


@bp.route("/results")
@login_required
def results():
    return "mystudy results page"
```

### 1.6 Rebuild the DB

Models in a new plugin require the schema to be (re-)materialised:

```bash
./scripts/init-db.sh                # adds any missing tables
# or, if you reshaped existing tables:
./scripts/reset-db.sh
```

See [`design-decisions.md`](design-decisions.md) §3 for the no-migrations rationale.

---

## 2. Add a new steering modality

The SAE Steering plugin already has four modalities (`sliders`, `toggles`, `text`, `examples`). Adding a fifth — call it `mood` — requires four small files.

### 2.1 Implement the strategy

```python
# server/plugins/steering/modalities/mood.py
from __future__ import annotations

from typing import Any, Dict

from ..constants import Modalities
from .base import SteeringModality, SteeringResult


class MoodSteering(SteeringModality):
    modality_id = Modalities.MOOD

    def apply(self, data: Dict[str, Any], *, conf: dict, active_model: dict) -> SteeringResult:
        mood = (data.get("mood") or "").strip().lower()
        intensity = float(data.get("intensity") or 0.5)

        # Replace with real cluster scoring. This stub maps "happy" -> a single
        # named cluster with intensity-derived weight.
        if mood == "happy":
            return SteeringResult(
                features=[{"id": "cluster_42", "label": "Feel-good"}],
                adjustments={"cluster_42": min(0.95, max(-0.95, intensity))},
                metadata={"mood": mood, "intensity": intensity},
            )

        # No-match (NFR-12 pattern): empty adjustments, route layer decides messaging.
        return SteeringResult(features=[], adjustments={}, metadata={"mood": mood})
```

### 2.2 Add the modality id

```python
# server/plugins/steering/constants.py
class Modalities:
    SLIDERS = "sliders"
    TOGGLES = "toggles"
    TEXT = "text"
    EXAMPLES = "examples"
    RESET = "reset"
    MOOD = "mood"        # <-- new
```

### 2.3 Register the strategy

```python
# server/plugins/steering/modalities/registry.py
from .mood import MoodSteering

_REGISTRY: Dict[str, SteeringModality] = {
    Modalities.SLIDERS: SliderSteering(),
    Modalities.TOGGLES: ToggleSteering(),
    Modalities.TEXT: TextSteering(),
    Modalities.EXAMPLES: ExampleSteering(),
    Modalities.MOOD: MoodSteering(),       # <-- new
}
```

### 2.4 Expose it in study config and the create UI

`study_config.py::derive_enabled_modalities` must accept the new id, and the create page (`templates/sae_steering_create.html`) should add a checkbox. Both follow the existing patterns for `text` and `examples`.

### 2.5 Add a route to invoke it from the participant UI

```python
# server/plugins/steering/routes/steering/actions.py
@bp.route("/apply-mood-steering", methods=["POST"])
def apply_mood_steering():
    data = request.get_json(force=True) or {}
    conf = normalize_study_config(load_user_study_config(session.get("user_study_id")))
    active_model = get_active_model_config(conf)
    derived = get_modality_strategy("mood").apply(data, conf=conf, active_model=active_model)

    participation_id = session.get("participation_id")
    if participation_id:
        # If the modality writes facts, call a typed audit writer (see §4).
        audit.record_event(
            "mood-steering-applied",
            participation_id=participation_id,
            approach_index=int(session.get("current_phase", 0)),
            iteration=session.get("iteration", 1),
            modality="mood",
            raw_payload={"mood": data.get("mood"), "intensity": data.get("intensity")},
        )
    return jsonify({
        "status": "ok",
        "features": derived.features,
        "adjustments": derived.adjustments,
    })
```

### 2.6 Tests

Write a unit test for the strategy itself (pure function, no DB), plus an integration test that hits the new route. See `tests/plugins/steering/test_steering_actions_and_security.py` for the pattern.

---

## 3. Add a new dataset

The framework's dataset loader is `server/plugins/utils/data_loading.py`. Adding `ml-100k` (as an example) is three changes:

### 3.1 Drop the dataset on disk

```
server/static/datasets/ml-100k/
├── ratings.csv      (userId, movieId, rating, timestamp)
├── movies.csv       (movieId, title, genres)
├── tags.csv         (optional, same shape as MovieLens)
├── links.csv        (optional)
├── plots.csv        (optional)
└── img/             (optional poster art)
```

### 3.2 Whitelist the variant

```python
# server/plugins/steering/constants.py
SUPPORTED_DATASET_VARIANTS = {
    "ml-32m-filtered",
    "ml-100k",      # <-- new
}
```

The `_resolve_safe_cache_path(ml_variant)` helper (see [`design-decisions.md`](design-decisions.md) §11) validates the variant against `^[A-Za-z0-9._-]+$` and resolves the cache directory under `server/cache/utils/<variant>/`, so the new variant is automatically sandboxed.

### 3.3 Surface it in the create UI

The `sae_steering_create.html` page reads `SUPPORTED_DATASET_VARIANTS` for the dataset dropdown. No template change needed — the new option appears automatically.

---

## 4. Add a new typed audit table

The audit pipeline is documented in [`tech-docs.md`](tech-docs.md) §7. Adding a fact requires exactly three changes.

### 4.1 Declare the model

```python
# server/plugins/steering/persistence/models.py
class SaeMoodLog(db.Model):
    __tablename__ = "sae_mood_log"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    study_run_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_study_run.id", ondelete="CASCADE"),
        nullable=False,
    )
    approach_run_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_approach_run.id", ondelete="CASCADE"),
        nullable=False,
    )
    participation_id = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_steering_event.id", ondelete="CASCADE"),
        nullable=False,
    )
    iteration = db.Column(db.Integer, nullable=False)
    mood = db.Column(db.String, nullable=False)
    intensity = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index("ix_sae_mood_log_approach_iter", "approach_run_id", "iteration"),
    )
```

The four FKs (`study_run_id`, `approach_run_id`, `participation_id`, `event_id`) and `CASCADE` on `participation.id` are the binding contract — see [`tech-docs.md`](tech-docs.md) §5.2 cascades.

### 4.2 Add the single-writer function

```python
# server/plugins/steering/service/audit.py
from server.plugins.steering.persistence.models import SaeMoodLog  # add to imports


def record_mood_steering(
    *,
    participation_id: int,
    approach_index: int,
    iteration: int,
    mood: str,
    intensity: float,
    active_model: Optional[dict] = None,
) -> SaeMoodLog:
    event = record_event(
        "mood-steering-applied",
        participation_id=participation_id,
        approach_index=approach_index,
        iteration=iteration,
        modality="mood",
        steering_mode=(active_model or {}).get("steering_mode"),
        raw_payload={"mood": mood, "intensity": intensity},
    )
    row = SaeMoodLog(
        study_run_id=event.study_run_id,
        approach_run_id=event.approach_run_id,
        participation_id=event.participation_id,
        event_id=event.id,
        iteration=int(iteration),
        mood=str(mood),
        intensity=float(intensity),
        created_at=utcnow(),
    )
    db.session.add(row)
    db.session.commit()
    return row
```

Architectural rule #1 (see [`tech-docs.md`](tech-docs.md) §4.4): **only** `audit.record_*` writes to typed tables. Routes call this; no other module inserts `SaeMoodLog` rows.

### 4.3 Rebuild the DB

```bash
./scripts/reset-db.sh
```

### 4.4 Reading from the new table

Analytics joins the new table in `results/analytics.py`. The dashboard pattern is one helper per metric. To count moods per approach:

```python
# server/plugins/steering/results/analytics.py
def _mood_counts(approach_run_ids):
    if not approach_run_ids:
        return {}
    rows = (
        db.session.query(SaeMoodLog.mood, db.func.count(SaeMoodLog.id))
        .filter(SaeMoodLog.approach_run_id.in_(approach_run_ids))
        .group_by(SaeMoodLog.mood)
        .all()
    )
    return {mood: int(count) for mood, count in rows}
```

---

## 5. Add a new reranking strategy

`SaeApproachRun.reranking_strategy` is already a snapshot column with three accepted enum values: `feature-conditioned` (implemented), `latent-perturbation` (reserved), `constrained-opt` (reserved). See [`design-decisions.md`](design-decisions.md) §8.

Implementing `latent-perturbation`:

### 5.1 Branch in the iteration controller

```python
# server/plugins/steering/service/iteration_controller.py
def apply_feature_adjustment_iteration(data):
    ...
    strategy = conf.get("reranking_strategy", "feature-conditioned")

    if strategy == "feature-conditioned":
        candidates = _rerank_feature_conditioned(...)
    elif strategy == "latent-perturbation":
        candidates = _rerank_latent_perturbation(...)
    else:
        raise NotImplementedError(f"Reranking strategy {strategy!r} is not implemented")
    ...
```

### 5.2 Implement the new helper

```python
def _rerank_latent_perturbation(*, base_candidates, neuron_shifts, recommender, k):
    # Perturb each candidate's item embedding in SAE latent space, project back,
    # rescore against the user seed, keep top-k. The math goes in equations.md §4.
    ...
```

### 5.3 No schema change

Because `reranking_strategy` is already an enum column, no DB change is needed. The CSV export, the dashboard, and the per-approach snapshot work unchanged.

### 5.4 Add the math to `equations.md`

The scoring formula lives in [`equations.md`](equations.md) §4. Each new strategy should add a subsection there.

---

## 6. Add a new researcher dashboard metric

Dashboard data is built by `results/analytics.py::build_results_payload(guid)`. Pattern: one private helper per metric, called from `build_results_payload`, returned as a JSON-safe nested dict. Helpers take `user_study_id` (or similar typed ids) and run a single grouped SQL query.

```python
# server/plugins/steering/results/analytics.py
def _mood_counts(user_study_id):
    rows = (
        db.session.query(
            SaeApproachRun.approach_index,
            SaeApproachRun.approach_name,
            SaeMoodLog.mood,
            db.func.count(SaeMoodLog.id),
        )
        .join(SaeStudyRun, SaeApproachRun.study_run_id == SaeStudyRun.id)
        .join(SaeMoodLog, SaeMoodLog.approach_run_id == SaeApproachRun.id)
        .filter(SaeStudyRun.user_study_id == user_study_id)
        .group_by(SaeApproachRun.approach_index, SaeApproachRun.approach_name, SaeMoodLog.mood)
        .all()
    )
    by_approach: dict[str, dict] = {}
    for idx, name, mood, cnt in rows:
        bucket = by_approach.setdefault(str(idx), {"label": name, "counts": {}})
        bucket["counts"][mood] = int(cnt)
    return by_approach


def build_results_payload(guid):
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    ...
    return {
        ...
        "modalities": {
            ...,
            "mood_counts": _mood_counts(user_study.id),
        },
    }, 200
```

The dashboard JS in `sae_steering_results.html` then reads `payload.modalities.mood_counts` and renders a Chart.js bar/heatmap inside the **Modalities** tab. The two existing helpers (`_slider_movement_by_position`, `_text_prompt_cluster_mappings`) are the simplest models to copy.

---

## 7. Add a new participant-facing endpoint

Participant routes live in `server/plugins/steering/routes/steering/actions.py`. They:

- read `participation_id`, `user_study_id`, `current_phase`, `iteration` from `flask.session`,
- pass those ids into the audit service,
- never read the session inside the service layer.

Template:

```python
# server/plugins/steering/routes/steering/actions.py
@bp.route("/my-new-action", methods=["POST"])
def my_new_action():
    data = request.get_json(force=True) or {}
    participation_id = session.get("participation_id")
    if not participation_id:
        return jsonify({"status": "skip", "reason": "no participation"}), 200

    audit.record_event(
        "my-new-action",
        participation_id=participation_id,
        approach_index=int(session.get("current_phase", 0)),
        iteration=int(session.get("iteration", 1)),
        modality="custom",
        raw_payload=data,
    )
    return jsonify({"status": "ok"})
```

If the new endpoint creates a fact (not just an envelope), follow §4 to declare a typed table and a `record_*` function.

---

## 8. Add a new CSV file to the export

The CSV export is `routes/results/views.py::export_csv_data`. Each typed table has a `files["<table_name>.csv"] = write_rows(headers, rows)` call. To add `sae_mood_log.csv`:

```python
# server/plugins/steering/routes/results/views.py
files["sae_mood_log.csv"] = write_rows(
    [
        "id", "study_run_id", "approach_run_id", "participation_id", "event_id",
        "iteration", "mood", "intensity", "created_at",
    ],
    [
        [
            r.id, r.study_run_id, r.approach_run_id, r.participation_id, r.event_id,
            r.iteration, r.mood, r.intensity,
            r.created_at.isoformat() if r.created_at else None,
        ]
        for r in SaeMoodLog.query.filter(
            SaeMoodLog.study_run_id.in_(study_run_ids)
        ).all()
    ] if study_run_ids else [],
)
```

The export test (`test_export_csv_authed_returns_zip_with_expected_files`) will then need the new filename added to its `expected` set.

---

## 9. Add a new questionnaire

The questionnaire monitor on the Results page is fully *modular*: it auto-discovers every key inside a `SaeQuestionnaireResponse.answers` JSON and classifies it as likert / numeric / categorical / text. **Adding a new questionnaire requires no code changes — only a template file.**

### 9.1 Drop in a template

Copy `server/static/questionnairs/sae_sample_questionnaire.html`, rename it (e.g. `my_post_study_questionnaire.html`), and edit the fields. Conventions:

- `name="field_id"` on every input becomes the JSON key in the response.
- Use unique names. Prefix them (`s_age`, `s_role`, `p1a_accuracy`, …) so different questionnaires don't collide on the dashboard.
- Likert questions are integer-valued radios `1`..`7` (the monitor will detect them as `likert`).
- Categorical questions are radios / select with short string values (`never`, `daily`, …).
- Free-text questions are `<textarea>` (the monitor stores the first 10 samples).

### 9.2 Point an approach at it

In the create UI (`/sae_steering/create`), each approach has a *Phase questionnaire file* field; you can also upload a custom one. The Final questionnaire field controls the questionnaire shown after the last approach. The chosen filename ends up on `SaeQuestionnaireResponse.questionnaire_file`.

### 9.3 Verify on the dashboard

After the first submission, open `/sae_steering/results?guid=<guid>` → **Questionnaires** tab. The new file appears as a section; each field becomes a row with its inferred kind and a compact summary (mean + distribution for likert/numeric, count table for categorical, samples for text). No analytics code, no template change, no migration.

### 9.4 Export

The full answers JSON is already in `sae_questionnaire_response.csv` (column `answers_json`) and in `export-raw`. You can post-process with R or pandas; no schema work needed.

---

## 10. Testing patterns

The full test suite is documented in [`tech-docs.md`](tech-docs.md) §10. Key conventions:

- **Unit test pure functions directly.** E.g. `_compose_text_adjustments` is tested without any DB fixture in `test_steering_actions_and_security.py::TestComposeTextAdjustments`.
- **Integration tests use `app_ctx`.** The fixture in `tests/conftest.py` gives you a freshly-created app + DB. Seed the data with `_seed_participation` (see `tests/plugins/steering/test_sae_audit.py`).
- **For routes that require authentication**, set the session manually:
  ```python
  client = app.test_client()
  with client.session_transaction() as sess:
      sess["_user_id"] = admin.email
      sess["_fresh"] = True
  ```
- **For routes that require a participation**, set the participation fields:
  ```python
  with client.session_transaction() as sess:
      sess["participation_id"] = participation.id
      sess["user_study_id"] = study.id
      sess["approach_order"] = [0, 1]
      sess["current_phase"] = 0
      sess["iteration"] = 1
  ```
- **Skip the ML stack** when the route would call `load_semantic_clusters` and you don't need real models: `monkeypatch.setattr("server.plugins.steering.modalities.text.load_semantic_clusters", lambda _id: {"clusters": []})`.

Run the suite locally:

```bash
./scripts/test.sh                  # full suite (~12 s, 43 tests today)
./scripts/test.sh -k mood          # only the new tests
./scripts/test.sh -x --tb=short    # stop at first failure with concise traceback
```
