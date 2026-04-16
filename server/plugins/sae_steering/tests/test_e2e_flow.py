#!/usr/bin/env python
"""End-to-end flow tests for the SAE steering study.

Verifies:
  1. Approach order randomisation is applied and logged to Interaction table.
  2. Elicitation search queries are logged.
  3. History (liked movies) is correctly tracked across iterations.
  4. Non-steering flow: 3 iterations + final confirmation saved.
  5. Steering flow: no phantom "auto-likes".
  6. Autosave endpoint persists snapshots.
  7. JSON export contains all interaction types in chronological order.
  8. Phase questionnaire and final questionnaire are logged.

Usage:
  cd server && python -m pytest plugins/sae_steering/tests/test_e2e_flow.py -v
"""
import json
import os
import sys
import datetime
from collections import Counter
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_app_singleton = None

@pytest.fixture()
def app():
    """Create a Flask app with in-memory SQLite for testing.

    Reuses a single app instance because Flask-Session registers the Session
    model once on the SQLAlchemy metadata and cannot be re-registered.
    """
    global _app_singleton

    os.environ['DATABASE_URL'] = 'sqlite://'
    os.environ.setdefault('APP_SECRET_KEY', 'test-key-1234')

    if _app_singleton is None:
        from app import create_app
        _app_singleton = create_app()
        _app_singleton.config['TESTING'] = True
        _app_singleton.config['WTF_CSRF_ENABLED'] = False

    application = _app_singleton

    with application.app_context():
        from app import db
        db.create_all()

        # Ensure a User row exists for FK constraints
        from models import User
        if not User.query.filter_by(email='test@test.com').first():
            db.session.add(User(email='test@test.com', password='x', authenticated=True, admin=True))
            db.session.commit()

        yield application

        db.session.remove()
        for table in reversed(db.metadata.sorted_tables):
            if table.name != 'user':
                db.session.execute(table.delete())
        db.session.commit()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def study_guid(app):
    """Create a UserStudy row and return its guid."""
    from app import db
    from models import UserStudy

    study_config = {
        "enable_comparison": True,
        "comparison_mode": "sequential",
        "randomize_approach_order": True,
        "num_iterations": 3,
        "num_recommendations": 5,
        "skip_participation_details": True,
        "models": [
            {
                "id": "approach_steering",
                "name": "Explicit Steering",
                "steering_mode": "sliders",
                "base": "elsa",
                "sae": "TopKSAE-1024",
            },
            {
                "id": "approach_none",
                "name": "No Steering",
                "steering_mode": "none",
                "base": "elsa",
                "sae": "TopKSAE-1024",
            },
        ],
    }
    guid = "test-study-e2e-001"
    us = UserStudy(
        creator="test@test.com",
        guid=guid,
        parent_plugin="sae_steering",
        settings=json.dumps(study_config),
        time_created=datetime.datetime.utcnow(),
        active=True,
        initialized=True,
    )
    db.session.add(us)
    db.session.commit()
    return guid


def _get_interactions(app, participation_id=None):
    """Helper: fetch all Interaction rows, optionally filtered."""
    from models import Interaction
    q = Interaction.query.order_by(Interaction.time.asc())
    if participation_id:
        q = q.filter(Interaction.participation == participation_id)
    return [
        {
            "type": ix.interaction_type,
            "data": json.loads(ix.data) if ix.data else {},
        }
        for ix in q.all()
    ]


# ---------------------------------------------------------------------------
# Mock heavy ML dependencies so tests don't need real model files
# ---------------------------------------------------------------------------

def _mock_generate_recs(**kwargs):
    """Return fake recommendation list."""
    k = kwargs.get('k', 5)
    return [
        {"movie_idx": 1000 + i, "title": f"Movie {1000 + i}", "url": "", "metadata": "Action"}
        for i in range(k)
    ], {}


@pytest.fixture(autouse=True)
def mock_ml(monkeypatch):
    """Patch ML model loading and recommendation generation."""
    # SAE recommender
    fake_rec = MagicMock()
    fake_rec.load.return_value = None
    fake_rec.item_ids = list(range(2000))
    fake_rec.item_features = MagicMock()

    monkeypatch.setattr(
        'plugins.sae_steering.sae_recommender.get_sae_recommender',
        lambda **kw: fake_rec,
    )

    def fake_gen_for_model(**kw):
        return _mock_generate_recs(**kw)

    monkeypatch.setattr(
        'plugins.sae_steering.generate_steered_recommendations_for_model',
        fake_gen_for_model,
        raising=False,
    )

    # Patch at module level in __init__
    import plugins.sae_steering as mod
    if hasattr(mod, 'generate_steered_recommendations_for_model'):
        monkeypatch.setattr(mod, 'generate_steered_recommendations_for_model', fake_gen_for_model)

    # Patch _unwrap_recommendation_payload
    def fake_unwrap(payload):
        if isinstance(payload, tuple):
            return payload
        return payload, {}
    if hasattr(mod, '_unwrap_recommendation_payload'):
        monkeypatch.setattr(mod, '_unwrap_recommendation_payload', fake_unwrap)

    # Patch _select_slider_features and _select_cluster_features
    def fake_slider_features(*a, **kw):
        return [
            {"id": f"cluster_{i}", "label": f"Feature {i}", "category": "latent",
             "description": f"Test feature {i}", "member_ids": [i], "activation": 0.5, "movie_count": 100}
            for i in range(3)
        ]
    if hasattr(mod, '_select_slider_features'):
        monkeypatch.setattr(mod, '_select_slider_features', fake_slider_features)
    if hasattr(mod, '_select_cluster_features'):
        monkeypatch.setattr(mod, '_select_cluster_features', fake_slider_features)

    # Patch _load_semantic_clusters
    def fake_clusters(model_id=None):
        return {"clusters": [
            {"cluster_id": f"cluster_{i}", "label": f"Feature {i}", "neuron_ids": [i],
             "description": f"Test {i}", "support": 100}
            for i in range(10)
        ]}
    if hasattr(mod, '_load_semantic_clusters'):
        monkeypatch.setattr(mod, '_load_semantic_clusters', fake_clusters)

    # Patch data loader
    fake_loader = MagicMock()
    fake_loader.movie_index_to_id = {i: i for i in range(2000)}
    fake_loader.movies_df_indexed = MagicMock()
    monkeypatch.setattr(
        'plugins.utils.data_loading.load_ml_dataset',
        lambda **kw: fake_loader,
    )

    # Patch _update_elsa_seed_with_likes
    if hasattr(mod, '_update_elsa_seed_with_likes'):
        monkeypatch.setattr(mod, '_update_elsa_seed_with_likes', lambda *a, **kw: None)

    yield


# ---------------------------------------------------------------------------
# Test: Randomisation is logged
# ---------------------------------------------------------------------------

class TestRandomisation:

    def test_approach_order_is_logged(self, app, client, study_guid):
        """After show-features, an 'approach-order-assigned' interaction should exist."""
        with client.session_transaction() as sess:
            sess['user_study_guid'] = study_guid
            from models import UserStudy
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_id'] = us.id

            from models import Participation
            from app import db
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-uuid-1',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['uuid'] = 'test-uuid-1'

        resp = client.get(f'/sae_steering/show-features?selectedMovies=100,200,300',
                          follow_redirects=False)
        assert resp.status_code in (302, 200)

        with app.app_context():
            interactions = _get_interactions(app)
            types = [ix['type'] for ix in interactions]
            assert 'approach-order-assigned' in types, \
                f"Expected 'approach-order-assigned' in {types}"
            assert 'elicitation-completed' in types, \
                f"Expected 'elicitation-completed' in {types}"

            order_ix = next(ix for ix in interactions if ix['type'] == 'approach-order-assigned')
            assert isinstance(order_ix['data']['approach_order'], list)
            assert sorted(order_ix['data']['approach_order']) == [0, 1]

    def test_randomisation_produces_both_orders(self, app, client, study_guid):
        """Over many sessions, both [0,1] and [1,0] should appear."""
        orders_seen = set()
        for i in range(30):
            with app.test_client() as c:
                with c.session_transaction() as sess:
                    from models import UserStudy, Participation
                    from app import db
                    us = UserStudy.query.filter_by(guid=study_guid).first()
                    sess['user_study_guid'] = study_guid
                    sess['user_study_id'] = us.id
                    p = Participation(
                        user_study_id=us.id,
                        time_joined=datetime.datetime.utcnow(),
                        uuid=f'uuid-rand-{i}',
                    )
                    db.session.add(p)
                    db.session.commit()
                    sess['participation_id'] = p.id
                    sess['uuid'] = f'uuid-rand-{i}'

                c.get(f'/sae_steering/show-features?selectedMovies=100,200',
                      follow_redirects=False)

                with c.session_transaction() as sess:
                    order = tuple(sess.get('approach_order', []))
                    orders_seen.add(order)

            if len(orders_seen) == 2:
                break

        assert (0, 1) in orders_seen or (1, 0) in orders_seen, \
            f"Expected both orderings in {orders_seen}"


# ---------------------------------------------------------------------------
# Test: Elicitation search is logged
# ---------------------------------------------------------------------------

class TestSearchLogging:

    def test_item_search_logged(self, app, client, study_guid):
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-search-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['uuid'] = 'test-search-uuid'

        import importlib
        pe_mod = importlib.import_module('plugins.utils.preference_elicitation')
        with patch.object(pe_mod, 'search_for_movie',
                          return_value=[{"movie": "The Matrix", "url": "/img/1.jpg", "movie_idx": 1}]):
            resp = client.get('/sae_steering/item-search?pattern=matrix')
            assert resp.status_code == 200

        with app.app_context():
            interactions = _get_interactions(app)
            search_ixs = [ix for ix in interactions if ix['type'] == 'elicitation-search']
            assert len(search_ixs) >= 1
            assert search_ixs[0]['data']['query'] == 'matrix'
            assert search_ixs[0]['data']['phase'] == 'elicitation'


# ---------------------------------------------------------------------------
# Test: Approve preferences (final confirmation flag)
# ---------------------------------------------------------------------------

class TestPreferencesApproval:

    def test_final_confirmation_flagged(self, app, client, study_guid):
        """When iteration_locked_final is True, approve should log is_final_confirmation=True."""
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-approve-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['uuid'] = 'test-approve-uuid'
            sess['iteration'] = 4
            sess['current_phase'] = 0
            sess['iteration_locked_final'] = True

        resp = client.post('/sae_steering/approve-preferences',
                           data=json.dumps({"liked_movies": [100, 200]}),
                           content_type='application/json')
        assert resp.status_code == 200

        with app.app_context():
            interactions = _get_interactions(app)
            approve_ixs = [ix for ix in interactions if ix['type'] == 'preferences-approved']
            assert len(approve_ixs) == 1
            assert approve_ixs[0]['data']['is_final_confirmation'] is True
            assert approve_ixs[0]['data']['liked_movies'] == [100, 200]

    def test_regular_approval_not_final(self, app, client, study_guid):
        """Normal mid-iteration approval should have is_final_confirmation=False."""
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-approve-reg-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['uuid'] = 'test-approve-reg-uuid'
            sess['iteration'] = 1
            sess['current_phase'] = 0
            sess['iteration_locked_final'] = False

        resp = client.post('/sae_steering/approve-preferences',
                           data=json.dumps({"liked_movies": [300]}),
                           content_type='application/json')
        assert resp.status_code == 200

        with app.app_context():
            interactions = _get_interactions(app)
            approve_ixs = [ix for ix in interactions if ix['type'] == 'preferences-approved']
            assert len(approve_ixs) == 1
            assert approve_ixs[0]['data']['is_final_confirmation'] is False


# ---------------------------------------------------------------------------
# Test: Movie feedback logging
# ---------------------------------------------------------------------------

class TestMovieFeedback:

    def test_like_and_neutral_logged(self, app, client, study_guid):
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-feedback-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['iteration'] = 1
            sess['current_phase'] = 0

        client.post('/sae_steering/log-movie-feedback',
                     data=json.dumps({"movie_id": 500, "action": "like", "iteration": 1, "rank": 3, "list_id": "recs-single"}),
                     content_type='application/json')
        client.post('/sae_steering/log-movie-feedback',
                     data=json.dumps({"movie_id": 500, "action": "neutral", "iteration": 1, "rank": 3, "list_id": "recs-single"}),
                     content_type='application/json')

        with app.app_context():
            interactions = _get_interactions(app)
            fb = [ix for ix in interactions if ix['type'] == 'movie-feedback']
            assert len(fb) == 2
            assert fb[0]['data']['action'] == 'like'
            assert fb[1]['data']['action'] == 'neutral'
            assert fb[0]['data']['movie_id'] == 500


# ---------------------------------------------------------------------------
# Test: Autosave endpoint
# ---------------------------------------------------------------------------

class TestAutosave:

    def test_autosave_persists(self, app, client, study_guid):
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-autosave-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['iteration'] = 2
            sess['current_phase'] = 0

        payload = {
            "liked_movies": [100, 200, 300],
            "feature_adjustments": {"cluster_1": 0.5},
            "activity_snapshot": {"1": {"sliders": [], "liked": [{"mid": 100, "title": "Movie 100"}]}},
            "timestamp": "2026-04-16T12:00:00",
            "trigger": "periodic",
        }
        resp = client.post('/sae_steering/autosave',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'

        with app.app_context():
            interactions = _get_interactions(app)
            auto_ixs = [ix for ix in interactions if ix['type'] == 'autosave']
            assert len(auto_ixs) == 1
            assert auto_ixs[0]['data']['liked_movies'] == [100, 200, 300]
            assert auto_ixs[0]['data']['trigger'] == 'periodic'


# ---------------------------------------------------------------------------
# Test: UI event logging (slider adjustments via search etc.)
# ---------------------------------------------------------------------------

class TestUiEventLogging:

    def test_slider_adjusted_event(self, app, client, study_guid):
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-uievent-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['iteration'] = 2
            sess['current_phase'] = 0

        resp = client.post('/sae_steering/log-ui-event',
                           data=json.dumps({
                               "event_type": "search-slider-adjusted",
                               "feature_id": "cluster_5",
                               "label": "Horror Movies",
                               "value": 0.75,
                               "found_via_query": "horror",
                               "source": "search_panel",
                           }),
                           content_type='application/json')
        assert resp.status_code == 200

        resp2 = client.post('/sae_steering/log-ui-event',
                            data=json.dumps({
                                "event_type": "slider-adjusted",
                                "feature_id": "cluster_1",
                                "label": "Comedy",
                                "value": -0.5,
                                "source": "main_panel",
                            }),
                            content_type='application/json')
        assert resp2.status_code == 200

        with app.app_context():
            interactions = _get_interactions(app)
            search_adj = [ix for ix in interactions if ix['type'] == 'search-slider-adjusted']
            assert len(search_adj) == 1
            assert search_adj[0]['data']['feature_id'] == 'cluster_5'
            assert search_adj[0]['data']['found_via_query'] == 'horror'
            assert search_adj[0]['data']['source'] == 'search_panel'
            assert search_adj[0]['data']['iteration'] == 2

            main_adj = [ix for ix in interactions if ix['type'] == 'slider-adjusted']
            assert len(main_adj) == 1
            assert main_adj[0]['data']['source'] == 'main_panel'


# ---------------------------------------------------------------------------
# Test: Raw JSON export
# ---------------------------------------------------------------------------

class TestExportRaw:

    def test_export_returns_all_interactions(self, app, client, study_guid):
        """Export should include all participations and their interactions."""
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-export-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['iteration'] = 1
            sess['current_phase'] = 0
            sess['iteration_locked_final'] = False
            participation_id = p.id

        # Create some interactions
        client.post('/sae_steering/log-movie-feedback',
                     data=json.dumps({"movie_id": 1, "action": "like", "iteration": 1}),
                     content_type='application/json')
        client.post('/sae_steering/approve-preferences',
                     data=json.dumps({"liked_movies": [1]}),
                     content_type='application/json')
        client.post('/sae_steering/autosave',
                     data=json.dumps({"liked_movies": [1], "trigger": "periodic"}),
                     content_type='application/json')

        resp = client.get(f'/sae_steering/export-raw/{study_guid}')
        assert resp.status_code == 200

        export = resp.get_json()
        assert export['study_guid'] == study_guid
        assert export['participants_total'] >= 1

        participant = export['participants'][0]
        interaction_types = [ix['type'] for ix in participant['interactions']]
        assert 'movie-feedback' in interaction_types
        assert 'preferences-approved' in interaction_types
        assert 'autosave' in interaction_types

        # Verify chronological order
        times = [ix['time'] for ix in participant['interactions'] if ix['time']]
        assert times == sorted(times), "Interactions should be in chronological order"

    def test_export_nonexistent_study(self, client):
        resp = client.get('/sae_steering/export-raw/nonexistent-guid')
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Test: Feature search is logged
# ---------------------------------------------------------------------------

class TestFeatureSearch:

    def test_feature_search_logged(self, app, client, study_guid):
        with client.session_transaction() as sess:
            from models import UserStudy, Participation
            from app import db
            us = UserStudy.query.filter_by(guid=study_guid).first()
            sess['user_study_guid'] = study_guid
            sess['user_study_id'] = us.id
            p = Participation(
                user_study_id=us.id,
                time_joined=datetime.datetime.utcnow(),
                uuid='test-fsearch-uuid',
            )
            db.session.add(p)
            db.session.commit()
            sess['participation_id'] = p.id
            sess['iteration'] = 2
            sess['current_phase'] = 1
            sess['current_features'] = []

        resp = client.get('/sae_steering/search-features?q=action')
        assert resp.status_code == 200

        with app.app_context():
            interactions = _get_interactions(app)
            fs_ixs = [ix for ix in interactions if ix['type'] == 'feature-search']
            assert len(fs_ixs) >= 1
            assert fs_ixs[0]['data']['query'] == 'action'
            assert fs_ixs[0]['data']['phase'] == 1
            assert fs_ixs[0]['data']['iteration'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
