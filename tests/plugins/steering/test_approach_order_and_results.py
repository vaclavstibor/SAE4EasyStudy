"""Tests for randomized approach ordering and the results aggregation endpoint."""

import json
from datetime import datetime


def _three_approach_config(*, randomize=True):
    return {
        "enable_comparison": True,
        "comparison_mode": "sequential",
        "randomize_approach_order": randomize,
        "models": [
            {
                "id": "approach_a",
                "name": "Approach A",
                "base": "elsa",
                "sae": "TopKSAE-1024",
                "steering_mode": "both",
                "enabled_modalities": ["sliders", "text", "reset"],
            },
            {
                "id": "approach_b",
                "name": "Approach B",
                "base": "elsa",
                "sae": "TopKSAE-1024",
                "steering_mode": "toggles",
                "enabled_modalities": ["toggles", "reset"],
            },
            {
                "id": "approach_c",
                "name": "Approach C",
                "base": "elsa",
                "sae": "TopKSAE-1024",
                "steering_mode": "text",
                "enabled_modalities": ["text", "reset"],
            },
        ],
    }


def _seed_three_approach_participation(db, *, randomize=True):
    from server.platform.persistence.base_models import Participation, User, UserStudy

    user = User(email="admin@example.com", password="x", authenticated=True, admin=True)
    study = UserStudy(
        creator=user.email,
        guid="study-guid-3",
        parent_plugin="sae_steering",
        settings=json.dumps(_three_approach_config(randomize=randomize)),
        time_created=datetime.utcnow(),
        active=True,
        initialized=True,
    )
    db.session.add(user)
    db.session.add(study)
    db.session.flush()
    participation = Participation(
        participant_email="participant@example.com",
        user_study_id=study.id,
        time_joined=datetime.utcnow(),
        uuid="participant-uuid-3",
        language="en",
    )
    db.session.add(participation)
    db.session.commit()
    return study, participation


def test_get_effective_models_keeps_fixed_order_for_n_approaches(app_ctx):
    app, _ = app_ctx

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.service.participation import get_effective_models

        models = get_effective_models(_three_approach_config(randomize=False))

        assert [model["name"] for model in models] == ["Approach A", "Approach B", "Approach C"]
        assert session["approach_order"] == [0, 1, 2]


def test_get_effective_models_randomizes_n_approaches_once(app_ctx, monkeypatch):
    app, _ = app_ctx

    class FakeRandom:
        def shuffle(self, values):
            values[:] = [2, 0, 1]

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.service import participation

        monkeypatch.setattr(participation.secrets, "SystemRandom", lambda: FakeRandom())

        models = participation.get_effective_models(_three_approach_config(randomize=True))
        models_again = participation.get_effective_models(_three_approach_config(randomize=True))

        assert [model["name"] for model in models] == ["Approach C", "Approach A", "Approach B"]
        assert [model["name"] for model in models_again] == [
            "Approach C",
            "Approach A",
            "Approach B",
        ]
        assert session["approach_order"] == [2, 0, 1]


def test_audit_study_run_uses_canonical_order_mapping_for_randomized_n_approaches(app_ctx):
    app, db = app_ctx
    study, participation = _seed_three_approach_participation(db, randomize=True)

    with app.test_request_context("/"):
        from flask import session

        from server.plugins.steering.service import audit

        session["participation_id"] = participation.id
        session["user_study_id"] = study.id
        session["approach_order"] = [2, 0, 1]
        session["current_phase"] = 0

        study_run = audit.ensure_study_run(participation.id, approach_order=[2, 0, 1])

        assert study_run.approach_order == [2, 0, 1]
        assert study_run.effective_order == ["Approach C", "Approach A", "Approach B"]


def test_analytics_groups_by_approach_id_not_phase_index(app_ctx):
    """Regression for the 'two Approach A rows' bug.

    Two participants with opposite randomized orders previously collapsed
    into ``approach_index`` buckets (per-participant phase position), so
    cross-participant aggregates showed two rows whose label was decided
    by SQL row order. The fix is to group by ``approach_id`` (stable
    identity from the study config). This test asserts:

    - Each approach gets exactly one bucket regardless of randomization.
    - Bucket labels equal the configured ``model['name']`` (not whichever
      participant happened to see that approach first).
    - The result dict preserves config-models order (A, B, C).
    """
    app, db = app_ctx
    from datetime import datetime

    from server.platform.persistence.base_models import Participation, User, UserStudy
    from server.plugins.steering.persistence.models import (
        SaeApproachRun,
        SaeMovieFeedback,
        SaeRecommendationSet,
        SaeSteeringEvent,
        SaeStudyRun,
    )
    from server.plugins.steering.results.analytics import (
        _approach_overview,
        _selected_rank_distribution,
        _selection_dynamics,
    )

    config = _three_approach_config(randomize=True)
    config_models = config["models"]

    user = User(email="admin2@example.com", password="x", authenticated=True, admin=True)
    study = UserStudy(
        creator=user.email,
        guid="study-grouping",
        parent_plugin="sae_steering",
        settings=json.dumps(config),
        time_created=datetime.utcnow(),
        active=True,
        initialized=True,
    )
    db.session.add_all([user, study])
    db.session.flush()

    # Two participants with FLIPPED randomized orders. The first sees
    # approach_a at phase 0; the second sees approach_b at phase 0.
    seedings = [
        ("p1", [("approach_a", "Approach A", 0), ("approach_b", "Approach B", 1)]),
        ("p2", [("approach_b", "Approach B", 0), ("approach_a", "Approach A", 1)]),
    ]
    for participant_uuid, ordering in seedings:
        participation = Participation(
            participant_email=f"{participant_uuid}@example.com",
            user_study_id=study.id,
            time_joined=datetime.utcnow(),
            uuid=participant_uuid,
            language="en",
        )
        db.session.add(participation)
        db.session.flush()
        study_run = SaeStudyRun(
            user_study_id=study.id,
            participation_id=participation.id,
            study_guid=study.guid,
            config_snapshot=config,
            approach_order=[i for _, _, i in ordering],
            effective_order=[name for _, name, _ in ordering],
            started_at=datetime.utcnow(),
            status="active",
        )
        db.session.add(study_run)
        db.session.flush()
        for approach_id, approach_name, phase_idx in ordering:
            ar = SaeApproachRun(
                study_run_id=study_run.id,
                participation_id=participation.id,
                approach_index=phase_idx,
                approach_id=approach_id,
                approach_name=approach_name,
                steering_mode="both",
                enabled_modalities=["sliders", "text", "reset"],
                sae_model_id="TopKSAE-1024",
                base_model_id="elsa",
                started_at=datetime.utcnow(),
                status="active",
            )
            db.session.add(ar)
            db.session.flush()
            # One like at rank 1, observed under this approach for this participant.
            envelope = SaeSteeringEvent(
                study_run_id=study_run.id,
                approach_run_id=ar.id,
                participation_id=participation.id,
                approach_index=phase_idx,
                approach_name=approach_name,
                event_type="movie-feedback",
                created_at=datetime.utcnow(),
            )
            db.session.add(envelope)
            db.session.flush()
            rec_set = SaeRecommendationSet(
                study_run_id=study_run.id,
                approach_run_id=ar.id,
                participation_id=participation.id,
                approach_index=phase_idx,
                approach_name=approach_name,
                iteration=1,
                list_id=f"{participant_uuid}-{approach_id}-1",
                steering_mode="both",
                generated_at=datetime.utcnow(),
            )
            db.session.add(rec_set)
            db.session.flush()
            db.session.add(
                SaeMovieFeedback(
                    study_run_id=study_run.id,
                    approach_run_id=ar.id,
                    recommendation_set_id=rec_set.id,
                    participation_id=participation.id,
                    event_id=envelope.id,
                    approach_index=phase_idx,
                    approach_name=approach_name,
                    movie_id=123,
                    title="t",
                    genres="g",
                    action="like",
                    rank=1,
                    iteration=1,
                    list_id=rec_set.list_id,
                    created_at=datetime.utcnow(),
                )
            )
    db.session.commit()

    # Approach overview: keyed by stable approach_id, in config order.
    overview = _approach_overview(study.id, config_models)
    assert list(overview.keys()) == ["approach_a", "approach_b", "approach_c"]
    assert overview["approach_a"]["label"] == "Approach A"
    assert overview["approach_a"]["participants"] == 2
    assert overview["approach_b"]["label"] == "Approach B"
    assert overview["approach_b"]["participants"] == 2
    assert overview["approach_c"]["participants"] == 0  # nobody reached phase 3

    # Selection dynamics: one row per approach_id, in config order, labels
    # match config NAMES even though phase orders were opposite.
    dynamics = _selection_dynamics(study.id, config_models)
    assert list(dynamics.keys()) == ["approach_a", "approach_b", "approach_c"]
    assert dynamics["approach_a"]["label"] == "Approach A"
    assert dynamics["approach_a"]["total_like_events"] == 2  # one like per participant
    assert dynamics["approach_b"]["total_like_events"] == 2
    assert dynamics["approach_c"]["total_like_events"] == 0

    # Rank distribution: same — no duplicate buckets, no label collisions.
    rank_dist = _selected_rank_distribution(study.id, config_models)
    assert list(rank_dist.keys()) == ["approach_a", "approach_b", "approach_c"]
    assert rank_dist["approach_a"]["total"] == 2
    assert rank_dist["approach_a"]["rank_counts"] == {"1": 2}
    assert rank_dist["approach_b"]["total"] == 2


def test_modality_breakdown_is_driven_by_enabled_modalities(app_ctx):
    """Per-approach modality breakdown only exposes modalities the approach declared.

    Regression for the empty 'Approach B' slider chart bug: when one
    approach is slider-only and another is text-only, the dashboard MUST
    NOT render a slider card for the text approach (or vice versa). The
    audit table may still contain feature-adjustment rows for the text
    approach (text-driven feature deltas), but those should be exposed
    under the text modality's metrics, not the slider modality's.
    """
    app, db = app_ctx
    from datetime import datetime

    from server.platform.persistence.base_models import Participation, User, UserStudy
    from server.plugins.steering.persistence.models import (
        SaeApproachRun,
        SaeFeatureAdjustment,
        SaeSteeringEvent,
        SaeStudyRun,
        SaeTextSteeringMatch,
        SaeTextSteeringQuery,
    )
    from server.plugins.steering.results.analytics import (
        _approach_modality_breakdown,
    )

    config = {
        "enable_comparison": True,
        "models": [
            {
                "id": "appr_slider",
                "name": "Slider approach",
                "base": "elsa",
                "sae": "TopKSAE-1024",
                "steering_mode": "sliders",
                "enabled_modalities": ["sliders", "reset"],
            },
            {
                "id": "appr_text",
                "name": "Text approach",
                "base": "elsa",
                "sae": "TopKSAE-1024",
                "steering_mode": "text",
                "enabled_modalities": ["text"],
            },
        ],
    }
    config_models = config["models"]
    user = User(email="modbreakdown@example.com", password="x", authenticated=True, admin=True)
    study = UserStudy(
        creator=user.email,
        guid="study-modbreakdown",
        parent_plugin="sae_steering",
        settings=json.dumps(config),
        time_created=datetime.utcnow(),
        active=True,
        initialized=True,
    )
    db.session.add_all([user, study])
    db.session.flush()
    participation = Participation(
        participant_email="p@example.com",
        user_study_id=study.id,
        time_joined=datetime.utcnow(),
        uuid="p-mod",
        language="en",
    )
    db.session.add(participation)
    db.session.flush()
    study_run = SaeStudyRun(
        user_study_id=study.id,
        participation_id=participation.id,
        study_guid=study.guid,
        config_snapshot=config,
        approach_order=[0, 1],
        effective_order=["Slider approach", "Text approach"],
        started_at=datetime.utcnow(),
        status="active",
    )
    db.session.add(study_run)
    db.session.flush()

    # Slider approach: 2 slider adjustments on named clusters.
    ar_slider = SaeApproachRun(
        study_run_id=study_run.id,
        participation_id=participation.id,
        approach_index=0,
        approach_id="appr_slider",
        approach_name="Slider approach",
        steering_mode="sliders",
        enabled_modalities=["sliders", "reset"],
        sae_model_id="TopKSAE-1024",
        base_model_id="elsa",
        started_at=datetime.utcnow(),
        status="active",
    )
    db.session.add(ar_slider)
    db.session.flush()
    envelope_slider = SaeSteeringEvent(
        study_run_id=study_run.id,
        approach_run_id=ar_slider.id,
        participation_id=participation.id,
        approach_index=0,
        approach_name="Slider approach",
        event_type="feature-adjust",
        created_at=datetime.utcnow(),
    )
    db.session.add(envelope_slider)
    db.session.flush()
    for cluster_label in ("Darkly Whimsical Fantasy", "Modern Social Commentary"):
        db.session.add(
            SaeFeatureAdjustment(
                study_run_id=study_run.id,
                approach_run_id=ar_slider.id,
                participation_id=participation.id,
                event_id=envelope_slider.id,
                iteration=1,
                feature_id="1",
                cluster_label=cluster_label,
                before_value=0.0,
                after_value=0.5,
                delta=0.5,
                applied_via="displayed",
                created_at=datetime.utcnow(),
            )
        )

    # Text approach: a text query with two cluster matches, plus a
    # text-driven feature adjustment with a PLACEHOLDER cluster label
    # (this is the row that previously corrupted the slider chart for B).
    ar_text = SaeApproachRun(
        study_run_id=study_run.id,
        participation_id=participation.id,
        approach_index=1,
        approach_id="appr_text",
        approach_name="Text approach",
        steering_mode="text",
        enabled_modalities=["text"],
        sae_model_id="TopKSAE-1024",
        base_model_id="elsa",
        started_at=datetime.utcnow(),
        status="active",
    )
    db.session.add(ar_text)
    db.session.flush()
    envelope_text = SaeSteeringEvent(
        study_run_id=study_run.id,
        approach_run_id=ar_text.id,
        participation_id=participation.id,
        approach_index=1,
        approach_name="Text approach",
        event_type="text-steer",
        created_at=datetime.utcnow(),
    )
    db.session.add(envelope_text)
    db.session.flush()
    query = SaeTextSteeringQuery(
        study_run_id=study_run.id,
        approach_run_id=ar_text.id,
        participation_id=participation.id,
        event_id=envelope_text.id,
        iteration=1,
        query_text="dark fantasy films",
        length_chars=len("dark fantasy films"),
        created_at=datetime.utcnow(),
    )
    db.session.add(query)
    db.session.flush()
    for cluster_id, label, weight in (
        ("c10", "Darkly Whimsical Fantasy", 0.6),
        ("c11", "Modern Social Commentary", 0.3),
    ):
        db.session.add(
            SaeTextSteeringMatch(
                query_id=query.id,
                cluster_id=cluster_id,
                label=label,
                weight=weight,
            )
        )
    db.session.add(
        SaeFeatureAdjustment(
            study_run_id=study_run.id,
            approach_run_id=ar_text.id,
            participation_id=participation.id,
            event_id=envelope_text.id,
            iteration=1,
            feature_id="55",
            cluster_label="Feature cluster_55",
            before_value=0.0,
            after_value=0.4,
            delta=0.4,
            applied_via="displayed",
            created_at=datetime.utcnow(),
        )
    )
    db.session.commit()

    breakdown = _approach_modality_breakdown(study.id, config_models)
    assert list(breakdown.keys()) == ["appr_slider", "appr_text"]

    slider = breakdown["appr_slider"]
    assert slider["label"] == "Slider approach"
    assert slider["steering_mode"] == "sliders"
    assert set(slider["modalities"].keys()) == {"sliders", "reset"}
    slider_metrics = {m["key"]: m["value"] for m in slider["modalities"]["sliders"]["metrics"]}
    assert slider_metrics["adjustments"] == 2
    assert slider_metrics["distinct_clusters"] == 2  # both named clusters
    assert slider_metrics["mean_abs_delta"] == 0.5
    # Reset modality is enabled but had zero events.
    reset_metrics = {m["key"]: m["value"] for m in slider["modalities"]["reset"]["metrics"]}
    assert reset_metrics["reset_count"] == 0

    text = breakdown["appr_text"]
    assert text["label"] == "Text approach"
    # CRITICAL: even though SaeFeatureAdjustment has 1 row for this
    # approach, no slider card appears — text never declared sliders.
    assert "sliders" not in text["modalities"]
    assert "text" in text["modalities"]
    text_metrics = {m["key"]: m["value"] for m in text["modalities"]["text"]["metrics"]}
    assert text_metrics["queries"] == 1
    assert text_metrics["distinct_prompts"] == 1
    assert text_metrics["cluster_mappings"] == 2
