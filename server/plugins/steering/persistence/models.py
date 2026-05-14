"""SQLAlchemy models that back the typed audit pipeline for SAE steering studies.

Every user action lands in one or more strongly-typed rows here plus a minimal
``SaeSteeringEvent`` envelope for timeline ordering. See ``service/audit.py``
for the single-writer service that populates these tables.
"""

from server.platform.persistence.db import db


class SaeStudyRun(db.Model):
    __tablename__ = "sae_study_run"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    participation_id = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    user_study_id = db.Column(
        db.Integer,
        db.ForeignKey("userstudy.id", ondelete="CASCADE"),
        nullable=False,
    )
    schema_version = db.Column(db.Integer, nullable=False, default=1)
    study_guid = db.Column(db.String, nullable=False)
    config_snapshot = db.Column(db.JSON, nullable=False)
    approach_order = db.Column(db.JSON, nullable=False)
    effective_order = db.Column(db.JSON, nullable=False)
    started_at = db.Column(db.DateTime, nullable=False)
    finished_at = db.Column(db.DateTime)
    status = db.Column(db.String, nullable=False, default="active")

    __table_args__ = (
        db.Index("ix_sae_study_run_participation_id", "participation_id"),
        db.Index("ix_sae_study_run_user_study_id", "user_study_id"),
    )


class SaeApproachRun(db.Model):
    __tablename__ = "sae_approach_run"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    study_run_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_study_run.id", ondelete="CASCADE"),
        nullable=False,
    )
    participation_id = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
        nullable=False,
    )
    approach_index = db.Column(db.Integer, nullable=False)
    approach_id = db.Column(db.String, nullable=False)
    approach_name = db.Column(db.String, nullable=False)
    steering_mode = db.Column(db.String, nullable=False)
    enabled_modalities = db.Column(db.JSON, nullable=False)
    sae_model_id = db.Column(db.String, nullable=False)
    base_model_id = db.Column(db.String, nullable=False)
    started_at = db.Column(db.DateTime, nullable=False)
    completed_at = db.Column(db.DateTime)
    status = db.Column(db.String, nullable=False, default="active")
    final_liked_count = db.Column(db.Integer)
    iterations_used = db.Column(db.Integer)
    total_slider_changes = db.Column(db.Integer, nullable=False, default=0)
    composition_mode = db.Column(db.String)
    reranking_strategy = db.Column(db.String)
    summary = db.Column(db.JSON, nullable=False, default=dict)

    __table_args__ = (
        db.UniqueConstraint(
            "study_run_id", "approach_index", name="uq_sae_approach_run_study_approach"
        ),
        db.Index("ix_sae_approach_run_participation_id", "participation_id"),
        db.Index("ix_sae_approach_run_study_run_id", "study_run_id"),
    )


class SaeSteeringEvent(db.Model):
    __tablename__ = "sae_steering_event"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    study_run_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_study_run.id", ondelete="CASCADE"),
        nullable=False,
    )
    approach_run_id = db.Column(
        db.Integer, db.ForeignKey("sae_approach_run.id", ondelete="CASCADE")
    )
    participation_id = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type = db.Column(db.String, nullable=False)
    approach_index = db.Column(db.Integer)
    approach_name = db.Column(db.String)
    iteration = db.Column(db.Integer)
    modality = db.Column(db.String)
    steering_mode = db.Column(db.String)
    source = db.Column(db.String)
    search_query = db.Column(db.String)
    raw_payload = db.Column(db.JSON, nullable=False, default=dict)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index(
            "ix_sae_steering_event_participation_time",
            "participation_id",
            "created_at",
        ),
        db.Index("ix_sae_steering_event_approach_iteration", "approach_run_id", "iteration"),
        db.Index("ix_sae_steering_event_type", "event_type"),
        db.Index("ix_sae_steering_event_source", "source"),
        db.Index(
            "ix_sae_steering_event_run_index_iter",
            "study_run_id",
            "approach_index",
            "iteration",
        ),
        db.Index("ix_sae_steering_event_type_modality", "event_type", "modality"),
    )


class SaeRecommendationSet(db.Model):
    __tablename__ = "sae_recommendation_set"

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
    approach_index = db.Column(db.Integer, nullable=False)
    approach_name = db.Column(db.String, nullable=False)
    iteration = db.Column(db.Integer, nullable=False)
    list_id = db.Column(db.String, nullable=False)
    steering_mode = db.Column(db.String, nullable=False)
    generated_at = db.Column(db.DateTime, nullable=False)
    debug_payload = db.Column(db.JSON, nullable=False, default=dict)

    __table_args__ = (
        db.Index(
            "ix_sae_recommendation_set_approach_iteration",
            "approach_run_id",
            "iteration",
        ),
        db.Index("ix_sae_recommendation_set_participation_id", "participation_id"),
    )


class SaeRecommendationItem(db.Model):
    __tablename__ = "sae_recommendation_item"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    recommendation_set_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_recommendation_set.id", ondelete="CASCADE"),
        nullable=False,
    )
    movie_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String, nullable=False)
    genres = db.Column(db.String, nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    list_id = db.Column(db.String, nullable=False)
    score = db.Column(db.Float)
    cf_score = db.Column(db.Float)
    genre_score = db.Column(db.Float)
    steering_score = db.Column(db.Float)
    raw_payload = db.Column(db.JSON, nullable=False, default=dict)

    __table_args__ = (
        db.UniqueConstraint(
            "recommendation_set_id", "rank", name="uq_sae_recommendation_item_set_rank"
        ),
        db.Index("ix_sae_recommendation_item_movie_id", "movie_id"),
    )


class SaeMovieFeedback(db.Model):
    __tablename__ = "sae_movie_feedback"

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
    recommendation_set_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_recommendation_set.id", ondelete="CASCADE"),
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
    approach_index = db.Column(db.Integer, nullable=False)
    approach_name = db.Column(db.String, nullable=False)
    iteration = db.Column(db.Integer, nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String, nullable=False)
    genres = db.Column(db.String, nullable=False)
    rank = db.Column(db.Integer)
    list_id = db.Column(db.String, nullable=False)
    action = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index("ix_sae_movie_feedback_approach_iteration", "approach_run_id", "iteration"),
        db.Index("ix_sae_movie_feedback_movie_id", "movie_id"),
        db.Index("ix_sae_movie_feedback_event_id", "event_id"),
    )


class SaeQuestionnaireResponse(db.Model):
    __tablename__ = "sae_questionnaire_response"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    study_run_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_study_run.id", ondelete="CASCADE"),
        nullable=False,
    )
    approach_run_id = db.Column(
        db.Integer, db.ForeignKey("sae_approach_run.id", ondelete="CASCADE")
    )
    participation_id = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
        nullable=False,
    )
    response_type = db.Column(db.String, nullable=False)
    approach_index = db.Column(db.Integer)
    approach_name = db.Column(db.String)
    questionnaire_file = db.Column(db.String)
    answers = db.Column(db.JSON, nullable=False)
    submitted_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index("ix_sae_questionnaire_response_participation_id", "participation_id"),
        db.Index("ix_sae_questionnaire_response_approach_run_id", "approach_run_id"),
    )


class SaeFeatureAdjustment(db.Model):
    """One row per feature whose value changed in a single apply.

    Replaces parsing of
    ``event.input_payload.control_state_before_get_recommendation``.
    """

    __tablename__ = "sae_feature_adjustment"

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
    feature_id = db.Column(db.String, nullable=False)
    cluster_label = db.Column(db.String)
    before_value = db.Column(db.Float, nullable=False)
    after_value = db.Column(db.Float, nullable=False)
    delta = db.Column(db.Float, nullable=False)
    applied_via = db.Column(db.String, nullable=False)
    search_query = db.Column(db.String)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index(
            "ix_sae_feature_adjustment_approach_iteration",
            "approach_run_id",
            "iteration",
        ),
        db.Index("ix_sae_feature_adjustment_feature_id", "feature_id"),
        db.Index("ix_sae_feature_adjustment_applied_via", "applied_via"),
    )


class SaeFeatureSearch(db.Model):
    """One row per user-issued feature search."""

    __tablename__ = "sae_feature_search"

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
    query_text = db.Column("query_text", db.String, nullable=False)
    result_count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index(
            "ix_sae_feature_search_approach_iteration",
            "approach_run_id",
            "iteration",
        ),
    )


class SaeFeatureSearchHit(db.Model):
    __tablename__ = "sae_feature_search_hit"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    search_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_feature_search.id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_id = db.Column(db.String, nullable=False)
    label = db.Column(db.String)
    match_score = db.Column(db.Float)
    rank = db.Column(db.Integer, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("search_id", "rank", name="uq_sae_feature_search_hit_search_rank"),
    )


class SaeTextSteeringQuery(db.Model):
    """One row per parsed text-steering submission."""

    __tablename__ = "sae_text_steering_query"

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
    query_text = db.Column("query_text", db.String, nullable=False)
    length_chars = db.Column(db.Integer, nullable=False)
    composition_mode = db.Column(db.String, nullable=False, default="replace")
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index(
            "ix_sae_text_steering_query_approach_iteration",
            "approach_run_id",
            "iteration",
        ),
    )


class SaeTextSteeringMatch(db.Model):
    __tablename__ = "sae_text_steering_match"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    query_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_text_steering_query.id", ondelete="CASCADE"),
        nullable=False,
    )
    cluster_id = db.Column(db.String, nullable=False)
    label = db.Column(db.String)
    weight = db.Column(db.Float, nullable=False)
    match_score = db.Column(db.Float)
    direction = db.Column(db.String)


class SaeExampleSteering(db.Model):
    """One row per example-based steering application."""

    __tablename__ = "sae_example_steering"

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
    example_strength = db.Column(db.Float)
    example_top_k = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index(
            "ix_sae_example_steering_approach_iteration",
            "approach_run_id",
            "iteration",
        ),
    )


class SaeExampleSteeringMovie(db.Model):
    __tablename__ = "sae_example_steering_movie"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    example_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_example_steering.id", ondelete="CASCADE"),
        nullable=False,
    )
    movie_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String)
    rank = db.Column(db.Integer, nullable=False)

    __table_args__ = (
        db.UniqueConstraint(
            "example_id", "rank", name="uq_sae_example_steering_movie_example_rank"
        ),
    )


class SaeResetAction(db.Model):
    """One row per reset action (global or single-feature)."""

    __tablename__ = "sae_reset_action"

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
    trigger = db.Column(db.String)
    scope = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index(
            "ix_sae_reset_action_approach_iteration",
            "approach_run_id",
            "iteration",
        ),
    )


class SaeElicitationPick(db.Model):
    """Typed audit row for the SAE steering preference-elicitation phase.

    Recorded alongside the EasyStudy-native ``Interaction`` log so analytics
    can be built from typed columns without parsing JSON blobs. Inherited
    EasyStudy plugins continue to read only from ``Interaction``.
    """

    __tablename__ = "sae_elicitation_pick"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    study_run_id = db.Column(
        db.Integer,
        db.ForeignKey("sae_study_run.id", ondelete="CASCADE"),
    )
    participation_id = db.Column(
        db.Integer,
        db.ForeignKey("participation.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_study_id = db.Column(
        db.Integer,
        db.ForeignKey("userstudy.id", ondelete="CASCADE"),
        nullable=False,
    )
    movie_id = db.Column(db.Integer, nullable=False)
    action = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.Index(
            "ix_sae_elicitation_pick_participation_time",
            "participation_id",
            "created_at",
        ),
    )
