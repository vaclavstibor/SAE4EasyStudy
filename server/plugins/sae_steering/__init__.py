"""
SAE Steering Plugin for EasyStudy

Enables users to understand and control neural recommender systems through
Sparse Autoencoder (SAE) derived interpretable features.
"""

import json
import os
import secrets
import sys
import traceback
import datetime
from pathlib import Path
from multiprocessing import Process

import numpy as np
from flask import Blueprint, jsonify, request, redirect, url_for, render_template, session
from flask_login import login_required
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

# Add parent directories to path for imports
[sys.path.append(i) for i in ['.', '..']]
[sys.path.append(i) for i in ['../.', '../..', '../../.']]

from models import UserStudy, Participation, Interaction
from common import get_tr, load_languages, multi_lang, load_user_study_config, load_user_study_config_by_guid
from plugins.utils.interaction_logging import log_interaction, study_ended
try:
    from .model_store import (
        DEFAULT_BOOTSTRAP_COMMAND,
        DEFAULT_TOPK_SAE_MODEL_ID,
        find_local_model_path,
    )
except ImportError:
    from model_store import (
        DEFAULT_BOOTSTRAP_COMMAND,
        DEFAULT_TOPK_SAE_MODEL_ID,
        find_local_model_path,
    )

# Plugin metadata
__plugin_name__ = "sae_steering"
__version__ = "0.1.0"
__author__ = "Research Team"
__author_contact__ = "research@example.com"
__description__ = "SAE-based interpretable and steerable neural recommendations"

# Create blueprint
bp = Blueprint(__plugin_name__, __plugin_name__, url_prefix=f"/{__plugin_name__}")

# Load translations
languages = load_languages(os.path.dirname(__file__))


def get_lang():
    """Get current language from session with fallback to English"""
    default_lang = "en"
    if "lang" in session and session["lang"] and session["lang"] in languages:
        return session["lang"]
    return default_lang


@bp.context_processor
def plugin_name():
    """Make plugin name available to templates"""
    return {"plugin_name": __plugin_name__}


_tmdb_cache = None

DEFAULT_STEERING_MODE = "sliders"
DEFAULT_FEATURE_SELECTION_ALGORITHM = "personalized_grouped_topk"
DEFAULT_BASE_MODEL_ID = "elsa"
DEFAULT_DATASET_VARIANT = "ml-32m-filtered"
SUPPORTED_DATASET_VARIANTS = {"ml-32m-filtered"}
SUPPORTED_STEERING_MODES = {"sliders", "toggles", "text", "both", "none"}
SUPPORTED_FEATURE_SELECTION_ALGORITHMS = {
    "personalized_grouped_topk",
    "global_label_topk",
}


def _approach_label(idx: int) -> str:
    if 0 <= idx < 26:
        return f"Approach {chr(65 + idx)}"
    return f"Approach {idx + 1}"


def _normalize_steering_mode(mode: str) -> str:
    mode = (mode or DEFAULT_STEERING_MODE).strip().lower()
    if mode in SUPPORTED_STEERING_MODES:
        return mode
    return DEFAULT_STEERING_MODE


def _normalize_feature_selection_algorithm(algorithm: str) -> str:
    algorithm = (algorithm or DEFAULT_FEATURE_SELECTION_ALGORITHM).strip().lower()
    if algorithm in SUPPORTED_FEATURE_SELECTION_ALGORITHMS:
        return algorithm
    return DEFAULT_FEATURE_SELECTION_ALGORITHM


def _normalize_dataset_variant(dataset_id: str) -> str:
    dataset_id = (dataset_id or DEFAULT_DATASET_VARIANT).strip().lower()
    if dataset_id in SUPPORTED_DATASET_VARIANTS:
        return dataset_id
    return DEFAULT_DATASET_VARIANT


def _get_study_dataset_variant(conf: dict) -> str:
    return _normalize_dataset_variant((conf or {}).get("dataset"))


def _persist_approach_order_on_participation(raw_order, effective_names, model_names):
    """Copy the approach order into ``Participation.extra_data`` as a belt-and-suspenders
    backup so it survives even if the ``approach-order-assigned`` interaction log
    row is lost, not written (dev reloads / session-reuse races), or pruned."""
    from app import db

    participation_id = session.get("participation_id")
    if not participation_id:
        return
    try:
        participation = Participation.query.filter(Participation.id == participation_id).first()
        if participation is None:
            return
        try:
            extra = json.loads(participation.extra_data) if participation.extra_data else {}
            if not isinstance(extra, dict):
                extra = {}
        except Exception:
            extra = {}
        extra["approach_order"] = list(raw_order)
        extra["effective_order"] = list(effective_names)
        extra["model_names"] = list(model_names)
        participation.extra_data = json.dumps(extra)
        db.session.commit()
    except Exception as exc:  # pragma: no cover — pure defensive logging
        print(f"[_persist_approach_order_on_participation] Failed to persist order: {exc}")


def _log_approach_order_once(raw_order, models):
    """Emit the ``approach-order-assigned`` interaction if (a) we have a
    participation id and (b) we haven't already logged it for this session.

    Historically we only logged this when the order was *first chosen*; if the
    session already had ``approach_order`` set from an earlier request (e.g. the
    developer landed on ``/join`` twice), the log row was skipped and the
    Participants table ended up showing ``-``.  We now also mirror the order
    into ``Participation.extra_data`` so results are self-healing."""
    if session.get("approach_order_logged"):
        return
    participation_id = session.get("participation_id")
    if not participation_id:
        return
    model_names = [m.get("name", f"Model {i}") for i, m in enumerate(models)]
    effective_names = [models[idx].get("name", f"Model {idx}") for idx in raw_order]
    log_interaction(
        participation_id,
        "approach-order-assigned",
        approach_order=list(raw_order),
        model_names=model_names,
        effective_order=effective_names,
    )
    session["approach_order_logged"] = True
    _persist_approach_order_on_participation(raw_order, effective_names, model_names)


def _get_effective_models(conf):
    conf = _normalize_study_config(conf)
    models = list(conf.get("models", []))
    if len(models) != 2 or not conf.get("enable_comparison", False):
        return models

    if not conf.get("randomize_approach_order", True):
        session["approach_order"] = [0, 1]
        _log_approach_order_once([0, 1], models)
        return models

    raw_order = session.get("approach_order")
    if (
        not isinstance(raw_order, list)
        or len(raw_order) != 2
        or sorted(raw_order) != [0, 1]
    ):
        raw_order = [0, 1] if secrets.randbelow(2) == 0 else [1, 0]
        session["approach_order"] = raw_order
        session.pop("approach_order_logged", None)
        print(f"[_get_effective_models] Assigned per-participant approach order: {raw_order}")

    _log_approach_order_once(raw_order, models)
    return [models[idx] for idx in raw_order]


def _get_required_sae_model_option():
    if find_local_model_path(DEFAULT_TOPK_SAE_MODEL_ID):
        return {
            "name": "WWW TopKSAE-8192 (k=32)",
            "id": DEFAULT_TOPK_SAE_MODEL_ID,
            "input_dim": 512,
            "feature_dim": 8192,
            "k": 32,
            "description": "Required SAE checkpoint for the SAE steering study",
        }
    return None


def _normalize_study_config(conf):
    conf = dict(conf or {})
    conf["skip_participation_details"] = conf.get("skip_participation_details", True)
    conf["disable_demographics"] = conf.get("disable_demographics", True)
    conf["show_general_features"] = conf.get("show_general_features", False)
    conf["dataset"] = _normalize_dataset_variant(conf.get("dataset"))
    conf["randomize_approach_order"] = bool(conf.get("randomize_approach_order", True))
    conf["feature_selection_algorithm"] = _normalize_feature_selection_algorithm(
        conf.get("feature_selection_algorithm")
    )

    legacy_mode = _normalize_steering_mode(conf.get("steering_mode", DEFAULT_STEERING_MODE))
    raw_models = conf.get("models") or []
    if not raw_models:
        raw_models = get_default_models()
        if not conf.get("enable_comparison", False):
            raw_models = raw_models[:1]

    models = []
    for idx, raw_model in enumerate(raw_models):
        model = dict(raw_model or {})
        model["id"] = model.get("id") or f"approach_{idx + 1}"
        model["name"] = model.get("name") or _approach_label(idx)
        model["base"] = model.get("base") or DEFAULT_BASE_MODEL_ID
        model["sae"] = model.get("sae") or DEFAULT_TOPK_SAE_MODEL_ID
        model["steering_mode"] = _normalize_steering_mode(
            model.get("steering_mode", legacy_mode)
        )
        model["feature_selection_algorithm"] = _normalize_feature_selection_algorithm(
            model.get("feature_selection_algorithm", conf["feature_selection_algorithm"])
        )
        models.append(model)

    conf["models"] = models
    conf["approach_count"] = len(models)
    conf["enable_comparison"] = len(models) > 1

    comparison_mode = (conf.get("comparison_mode") or "").strip().lower()
    if len(models) <= 1:
        comparison_mode = "none"
    elif len(models) > 2:
        comparison_mode = "sequential"
    elif comparison_mode not in {"side_by_side", "sequential"}:
        comparison_mode = "sequential"
    conf["comparison_mode"] = comparison_mode

    if models:
        conf["steering_mode"] = models[0]["steering_mode"]
    else:
        conf["steering_mode"] = legacy_mode

    return conf


def _get_active_model_config(conf, phase_idx=None):
    conf = _normalize_study_config(conf)
    models = _get_effective_models(conf)
    if not models:
        return {
            "id": "single",
            "name": "Approach A",
            "base": DEFAULT_BASE_MODEL_ID,
            "sae": DEFAULT_TOPK_SAE_MODEL_ID,
            "steering_mode": DEFAULT_STEERING_MODE,
            "feature_selection_algorithm": DEFAULT_FEATURE_SELECTION_ALGORITHM,
        }
    if phase_idx is None:
        phase_idx = session.get("current_phase", 0)
    return models[min(max(int(phase_idx), 0), len(models) - 1)]


def _get_active_sae_model_id(conf, phase_idx=None):
    model_id = _get_active_model_config(conf, phase_idx).get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    if not model_id or str(model_id).strip().lower() == "none":
        return DEFAULT_TOPK_SAE_MODEL_ID
    return model_id


def _get_phase_questionnaire_filename(conf, phase_idx=None):
    conf = _normalize_study_config(conf)
    model = _get_active_model_config(conf, phase_idx)
    return model.get("phase_questionnaire_file") or conf.get("phase_questionnaire_file")


def _get_steering_subtitle(steering_mode: str) -> str:
    steering_mode = _normalize_steering_mode(steering_mode)
    if steering_mode == "text":
        return "Describe what you want to steer your recommendations."
    if steering_mode == "both":
        return "Write text or adjust features to steer your recommendations."
    if steering_mode == "none":
        return "Review recommendations and select movies you would watch."
    return "Adjust features to steer your recommendations."


def _get_steering_guidance(steering_mode: str) -> str:
    steering_mode = _normalize_steering_mode(steering_mode)
    if steering_mode == "text":
        return "Start by reviewing the current recommendations, then describe the kind of change you want in your own words before getting updated recommendations."
    if steering_mode == "both":
        return "Start by reviewing the current recommendations, then either write what you want or adjust the discovered concepts before getting updated recommendations."
    if steering_mode == "none":
        return "Start by reviewing the current recommendations, select what you would watch, and then continue to the next recommendation update."
    return "Start by reviewing the current recommendations, adjust the discovered concepts below, and then get updated recommendations."


def _sync_prolific_session_from_request():
    if "PROLIFIC_PID" in request.args:
        session["PROLIFIC_PID"] = request.args.get("PROLIFIC_PID")
        session["PROLIFIC_STUDY_ID"] = request.args.get("STUDY_ID")
        session["PROLIFIC_SESSION_ID"] = request.args.get("SESSION_ID")
    else:
        session.pop("PROLIFIC_PID", None)
        session.pop("PROLIFIC_STUDY_ID", None)
        session.pop("PROLIFIC_SESSION_ID", None)


def _ensure_participation_for_guid(guid: str):
    if session.get("participation_id") and session.get("user_study_guid") == guid:
        return

    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        raise ValueError(f"Unknown study guid: {guid}")

    if "uuid" not in session:
        session["uuid"] = secrets.token_urlsafe(16)

    extra_data = {}
    if "PROLIFIC_PID" in session:
        extra_data["PROLIFIC_PID"] = session["PROLIFIC_PID"]
        extra_data["PROLIFIC_STUDY_ID"] = session.get("PROLIFIC_STUDY_ID")
        extra_data["PROLIFIC_SESSION_ID"] = session.get("PROLIFIC_SESSION_ID")

    participant_email = session.get("PROLIFIC_PID") or ""
    participation = Participation(
        participant_email=participant_email,
        user_study_id=user_study.id,
        time_joined=datetime.datetime.utcnow(),
        time_finished=None,
        age_group=None,
        gender=None,
        education=None,
        ml_familiar=None,
        language=get_lang(),
        uuid=session["uuid"],
        extra_data=json.dumps(extra_data),
    )
    from app import db

    db.session.add(participation)
    db.session.commit()

    session["participation_id"] = participation.id
    session["user_study_id"] = user_study.id
    session["user_study_guid"] = guid
    # Reset so _log_approach_order_once fires once for the new participation even
    # if the session already carried an approach_order from a prior run.
    session.pop("approach_order_logged", None)


def _load_tmdb_overviews():
    """Load movie plots keyed by movieId from plots.csv. Cached after first call."""
    global _tmdb_cache
    if _tmdb_cache is not None:
        return _tmdb_cache
    plots_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'static', 'datasets', 'ml-32m-filtered', 'plots.csv'
    )
    _tmdb_cache = {}
    if os.path.exists(plots_path):
        try:
            import csv
            with open(plots_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mid = row.get('movieId', '')
                    plot = row.get('plot', '')
                    if mid and plot:
                        _tmdb_cache[int(mid)] = plot
            print(f"[Plots] Loaded {len(_tmdb_cache)} movie plots")
        except Exception as e:
            print(f"[Plots] Could not load plots: {e}")
    return _tmdb_cache


def get_default_models():
    """Default A/B models if study config is missing."""
    return [
        {
            "id": "approach_a",
            "name": "Approach A",
            "base": DEFAULT_BASE_MODEL_ID,
            "sae": DEFAULT_TOPK_SAE_MODEL_ID,
            "steering_mode": "sliders",
            "feature_selection_algorithm": DEFAULT_FEATURE_SELECTION_ALGORITHM,
        },
        {
            "id": "approach_b",
            "name": "Approach B",
            "base": "elsa",
            "sae": DEFAULT_TOPK_SAE_MODEL_ID,
            "steering_mode": "none",
            "feature_selection_algorithm": DEFAULT_FEATURE_SELECTION_ALGORITHM,
        },
    ]


def get_cache_path(guid, name=""):
    """Get cache directory path for this plugin"""
    return os.path.join("cache", __plugin_name__, guid, name)


def _get_phase_id_map(session_key: str) -> dict:
    raw = session.get(session_key, {})
    if isinstance(raw, dict):
        return raw
    return {}


def _get_phase_movie_set(session_key: str, phase_idx: int) -> set:
    phase_map = _get_phase_id_map(session_key)
    raw_list = phase_map.get(str(int(phase_idx)), [])
    if not isinstance(raw_list, list):
        return set()
    return {int(mid) for mid in raw_list if mid is not None}


def _set_phase_movie_set(session_key: str, phase_idx: int, movie_ids: set) -> None:
    phase_map = _get_phase_id_map(session_key)
    phase_map[str(int(phase_idx))] = sorted({int(mid) for mid in movie_ids if mid is not None})
    session[session_key] = phase_map


def _get_phase_token_set(session_key: str, phase_idx: int) -> set:
    """Return phase-local set of string tokens (e.g., slider cluster IDs)."""
    phase_map = _get_phase_id_map(session_key)
    raw_list = phase_map.get(str(int(phase_idx)), [])
    if not isinstance(raw_list, list):
        return set()
    return {str(token) for token in raw_list if token is not None}


def _set_phase_token_set(session_key: str, phase_idx: int, tokens: set) -> None:
    """Store phase-local set of string tokens."""
    phase_map = _get_phase_id_map(session_key)
    phase_map[str(int(phase_idx))] = sorted({str(token) for token in tokens if token is not None})
    session[session_key] = phase_map


def _remember_shown_movies(phase_idx: int, movie_ids: list) -> None:
    if not movie_ids:
        return
    seen = _get_phase_movie_set("seen_movies_per_phase", phase_idx)
    seen.update({int(mid) for mid in movie_ids if mid is not None})
    _set_phase_movie_set("seen_movies_per_phase", phase_idx, seen)


# ============================================================================
# Feature De-duplication Helpers
# ============================================================================

_FUZZY_LABEL_JACCARD_THRESHOLD = 0.65


def _normalize_label(label: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return ' '.join(label.lower().split())


def _label_word_set(label: str) -> set:
    """Extract meaningful word tokens from a label for Jaccard comparison."""
    import re
    return {w for w in re.split(r'[\s·\-–—/,&]+', label.lower()) if len(w) > 2}


def _is_near_duplicate_label(new_label: str, existing_labels: set) -> bool:
    """Check if *new_label* is a near-duplicate of any label in *existing_labels*.

    Uses two criteria:
      1. Exact normalized match (fast path)
      2. Word-level Jaccard similarity > threshold (catches "Dark Comedy · Satire"
         vs "Dark Comedy · Social Satire")
    """
    norm = _normalize_label(new_label)
    if norm in existing_labels:
        return True
    new_words = _label_word_set(new_label)
    if not new_words:
        return False
    for existing in existing_labels:
        ex_words = _label_word_set(existing)
        if not ex_words:
            continue
        intersection = new_words & ex_words
        union = new_words | ex_words
        if len(intersection) / len(union) >= _FUZZY_LABEL_JACCARD_THRESHOLD:
            return True
    return False


def _expand_feature_adjustments(
    raw_adjustments: dict,
    cluster_map: dict = None,
) -> dict:
    """Expand cluster-level slider deltas into neuron-level deltas.

    Every cluster slider value is applied **equally** to all member neurons.
    """
    feature_adjustments: dict = {}
    cluster_map = cluster_map or {}

    for key, val in (raw_adjustments or {}).items():
        delta = float(val)
        if abs(delta) < 0.0001:
            continue
        neuron_ids = cluster_map.get(key)
        if neuron_ids:
            for nid in neuron_ids:
                skey = str(nid)
                feature_adjustments[skey] = feature_adjustments.get(skey, 0.0) + delta
        else:
            feature_adjustments[key] = feature_adjustments.get(key, 0.0) + delta

    return feature_adjustments


# ============================================================================
# Semantic Cluster Registry (from offline labeling pipeline)
# ============================================================================

_semantic_clusters_cache: dict = {}  # model_id -> parsed data


def _load_semantic_clusters(model_id: str = None) -> dict:
    """Load pre-computed semantic clusters from semantic_merged JSON.

    Returns dict with:
        clusters: list of {cluster_id, label, description, neuron_ids, support}
        cluster_map: {cluster_id: [neuron_ids]}
        neuron_to_cluster: {neuron_id: cluster_id}
    """
    resolved = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    if resolved in _semantic_clusters_cache:
        return _semantic_clusters_cache[resolved]

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    path = os.path.join(data_dir, f"semantic_merged_{resolved}.json")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Semantic clusters not found: {path}. "
            f"Copy from labeling/artifacts/."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    clusters = []
    cluster_map = {}
    neuron_to_cluster = {}
    for c in raw.get("clusters", []):
        cid = c["cluster_id"]
        nids = [int(n) for n in c["neuron_ids"]]
        clusters.append({
            "cluster_id": cid,
            "label": c["label"],
            "description": c.get("description", ""),
            "neuron_ids": nids,
            "support": c.get("support", len(nids)),
        })
        cluster_map[cid] = nids
        for nid in nids:
            neuron_to_cluster[nid] = cid

    result = {
        "clusters": clusters,
        "cluster_map": cluster_map,
        "neuron_to_cluster": neuron_to_cluster,
    }
    _semantic_clusters_cache[resolved] = result
    print(f"[clusters] Loaded {len(clusters)} clusters ({sum(len(v) for v in cluster_map.values())} neurons) from {path}")
    return result


# ============================================================================
# Study Creation & Configuration
# ============================================================================

@bp.route("/create")
@login_required
def create():
    """Display study creation interface"""
    tr = get_tr(languages, get_lang())
    
    params = {
        "title": tr("sae_create_title"),
        "select_dataset": tr("sae_create_select_dataset"),
        "select_base_model": tr("sae_create_select_base_model"),
        "select_sae_model": tr("sae_create_select_sae_model"),
        "select_steering_mode": tr("sae_create_select_steering_mode"),
        "num_features_display": tr("sae_create_num_features_display"),
        "num_recommendations": tr("sae_create_num_recommendations"),
        "num_iterations": tr("sae_create_num_iterations"),
        "create": tr("create"),
        "cancel": tr("cancel"),
    }
    
    return render_template("sae_steering_create.html", **params)


@bp.route("/available-datasets")
def available_datasets():
    """Return list of supported datasets"""
    return jsonify([
        {
            "name": "MovieLens 32M Filtered",
            "id": "ml-32m-filtered",
            "description": "Curated MovieLens dataset (8328 movies)"
        }
    ])


@bp.route("/available-sae-models")
def available_sae_models():
    """Return list of available SAE models"""
    required_model = _get_required_sae_model_option()
    if required_model:
        return jsonify([required_model])

    return jsonify([{
        "name": "Required SAE model missing",
        "id": "none",
        "input_dim": 512,
        "feature_dim": 8192,
        "k": 32,
        "description": f"Run `{DEFAULT_BOOTSTRAP_COMMAND}` to download the required checkpoint.",
    }])


@bp.route("/available-steering-modes")
def available_steering_modes():
    """Return list of steering modalities"""
    return jsonify([
        {
            "name": "No Steering (movie selection only)",
            "id": "none",
            "description": "No explicit steering controls, only movie feedback"
        },
        {
            "name": "Feature Sliders",
            "id": "sliders",
            "description": "Continuous adjustment of feature strengths"
        },
        {
            "name": "Feature Toggles",
            "id": "toggles",
            "description": "Binary on/off for features"
        },
        {
            "name": "Natural Language (Text)",
            "id": "text",
            "description": "Describe preferences in natural language (requires sentence-transformers)"
        }
    ])


@bp.route("/available-feature-selection-algorithms")
def available_feature_selection_algorithms():
    return jsonify([
        {
            "id": "personalized_grouped_topk",
            "name": "Personalized grouped Top-K",
            "description": "Personalize sliders from elicitation picks and deduplicate similar concepts."
        },
        {
            "id": "global_label_topk",
            "name": "Global label-diverse Top-K",
            "description": "Show globally strong, label-diverse features independent of elicitation."
        },
    ])


@bp.route("/available-neurons")
def available_neurons():
    """Return list of all available clusters for manual selection in study creation."""
    model_id = request.args.get("model_id") or DEFAULT_TOPK_SAE_MODEL_ID
    if not model_id or str(model_id).strip().lower() == "none":
        model_id = DEFAULT_TOPK_SAE_MODEL_ID

    try:
        sc = _load_semantic_clusters(model_id)
        clusters = []
        for c in sc["clusters"]:
            clusters.append({
                "id": c["cluster_id"],
                "label": c["label"],
                "category": "latent",
                "description": c.get("description", ""),
                "score": c["support"],
            })
        clusters.sort(key=lambda n: (-n["score"], n["label"].lower()))
        return jsonify(clusters)
    except Exception as e:
        print(f"[available_neurons] Error: {e}")
        traceback.print_exc()
        return jsonify([])


# ============================================================================
# Study Initialization
# ============================================================================

def _resolve_db_url():
    """Resolve the SQLAlchemy DB URL the way `server/app.py` does, but without
    needing a Flask app context (this function runs in a fresh subprocess).

    Mirrors the postgres:// -> postgresql:// rewrite for Railway/Heroku-style
    add-ons so a child process never accidentally writes to a different
    database than the main app."""
    db_url = os.environ.get('DATABASE_URL', 'sqlite:///instance/db.sqlite')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    return db_url


def long_initialization(guid):
    """
    Long-running initialization process for SAE steering study.
    Runs in separate process to avoid blocking.
    """
    engine = create_engine(_resolve_db_url())
    db_session = Session(engine)
    
    try:
        user_study = db_session.query(UserStudy).filter(UserStudy.guid == guid).first()
        conf = _normalize_study_config(json.loads(user_study.settings))
        
        # Create cache directories
        Path(get_cache_path(guid)).mkdir(parents=True, exist_ok=True)
        Path(get_cache_path(guid, "sae_model")).mkdir(parents=True, exist_ok=True)
        Path(get_cache_path(guid, "embeddings")).mkdir(parents=True, exist_ok=True)
        
        import shutil

        # Default questionnaire files bundled with the plugin
        _DEFAULT_QUESTIONNAIRE_DIR = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'data', 'recsys2026'
        )

        def _resolve_questionnaire(filename):
            """Move an uploaded file into cache, or fall back to bundled default."""
            if not filename:
                return
            dest = get_cache_path(guid, filename)
            if os.path.exists(dest):
                return
            uploaded = os.path.join("cache", __plugin_name__, "uploads", filename)
            if os.path.exists(uploaded):
                shutil.move(uploaded, dest)
                return
            bundled = os.path.join(_DEFAULT_QUESTIONNAIRE_DIR, filename)
            if os.path.exists(bundled):
                shutil.copy2(bundled, dest)
                print(f"[init] Copied bundled questionnaire {filename}")

        # Final questionnaire
        _resolve_questionnaire(conf.get("questionnaire_file"))

        # Phase questionnaires (global + per-model)
        phase_files = set()
        if conf.get("phase_questionnaire_file"):
            phase_files.add(conf["phase_questionnaire_file"])
        for model in conf.get("models", []):
            if model.get("phase_questionnaire_file"):
                phase_files.add(model["phase_questionnaire_file"])
        for phase_file in phase_files:
            _resolve_questionnaire(phase_file)

        print(f"Initialized SAE steering study with GUID: {guid}")
        
        user_study.initialized = True
        user_study.active = True
        
    except Exception as e:
        user_study.initialization_error = traceback.format_exc()
        print(f"Error during SAE steering initialization: {user_study.initialization_error}")
        print(str(e))
    
    db_session.commit()
    db_session.expunge_all()
    db_session.close()


@bp.route("/initialize", methods=["GET"])
@login_required
def initialize():
    """Start initialization process"""
    guid = request.args.get("guid")
    
    heavy_process = Process(
        target=long_initialization,
        daemon=True,
        args=(guid,)
    )
    heavy_process.start()
    
    return redirect(request.args.get("continuation_url"))


# ============================================================================
# User Study Participation
# ============================================================================

@bp.route("/join", methods=["GET"])
@multi_lang
def join():
    """Public endpoint for users to join study"""
    assert "guid" in request.args, "GUID must be provided"
    guid = request.args.get("guid")
    conf = _normalize_study_config(load_user_study_config_by_guid(guid))
    if conf.get("skip_participation_details", True):
        _sync_prolific_session_from_request()
        _ensure_participation_for_guid(guid)
        return redirect(url_for(f"{__plugin_name__}.on_joined", **request.args))
    return redirect(url_for("utils.join", 
                          continuation_url=url_for(f"{__plugin_name__}.on_joined"),
                          **request.args))


@bp.route("/on-joined", methods=["GET", "POST"])
def on_joined():
    """Callback after user has joined, redirect to study intro page"""
    return redirect(url_for(f"{__plugin_name__}.study_intro"))


def _get_min_resolution_settings(conf):
    """Return normalized min-resolution settings for client-side gating."""
    min_resolution_cfg = conf.get("min_resolution") if isinstance(conf.get("min_resolution"), dict) else {}

    def _safe_int(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    width = _safe_int(
        conf.get("min_resolution_width", conf.get("min_width", min_resolution_cfg.get("width"))),
        1280,
    )
    height = _safe_int(
        conf.get("min_resolution_height", conf.get("min_height", min_resolution_cfg.get("height"))),
        720,
    )
    error_message = conf.get(
        "min_resolution_error",
        (
            f"This study requires at least {width}x{height} resolution. "
            "Please resize your browser window (or switch to a larger screen) before continuing."
        ),
    )
    return width, height, error_message


@bp.route("/study-intro", methods=["GET"])
def study_intro():
    """Show intro page explaining what the study involves before proceeding."""
    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    tr = get_tr(languages, get_lang())

    models = _get_effective_models(conf)
    comparison_mode = conf.get("comparison_mode", "side_by_side")
    num_phases = len(models) if comparison_mode == "sequential" else 1
    num_iterations = conf.get("num_iterations", 3)
    min_resolution_width, min_resolution_height, min_resolution_error = _get_min_resolution_settings(conf)
    has_questionnaire = bool(conf.get("questionnaire_file")) or bool(conf.get("phase_questionnaire_file")) or any(
        model.get("phase_questionnaire_file") or conf.get("phase_questionnaire_file")
        for model in models
    )

    steering_label_a = ""
    steering_label_b = ""
    if num_phases > 1 and len(models) >= 2:
        mode_a = models[0].get("steering_mode", conf.get("steering_mode", "sliders"))
        mode_b = models[1].get("steering_mode", conf.get("steering_mode", "sliders"))
        steering_labels = {
            "sliders": "sliders",
            "text": "text input",
            "both": "sliders + text",
            "none": "movie selection only",
        }
        steering_label_a = steering_labels.get(mode_a, mode_a)
        steering_label_b = steering_labels.get(mode_b, mode_b)

    custom_intro_html = None
    if "text_overrides" in conf and "study_intro" in conf["text_overrides"]:
        custom_intro_html = conf["text_overrides"]["study_intro"]

    params = {
        "title": conf.get("study_title", tr("sae_steering_title")),
        "subtitle": conf.get("study_subtitle", "Interactive Recommendation Study"),
        "custom_intro_html": custom_intro_html,
        "time_estimate": conf.get("time_estimate", "10-15 minutes"),
        "start_button_text": conf.get("start_button_text", "I give my consent, let's continue"),
        "num_phases": num_phases,
        "num_iterations": num_iterations,
        "has_questionnaire": has_questionnaire,
        "min_resolution_width": min_resolution_width,
        "min_resolution_height": min_resolution_height,
        "min_resolution_error": min_resolution_error,
        "steering_label_a": steering_label_a,
        "steering_label_b": steering_label_b,
        "notes": conf.get("intro_notes", [
            "There are no right or wrong answers (except for attention checks).",
            "Your data is anonymous and used for research purposes only.",
        ]),
        "study_parts": [
            {
                "title": "Preference elicitation",
                "description": "Choose a few movies you like so the system can estimate your starting taste profile.",
            },
            {
                "title": "Implicit feedback phase",
                "description": "Refine recommendations without sliders; the system learns from your interactions with shown movies.",
            },
            {
                "title": "Explicit feedback phase",
                "description": "Refine recommendations by directly adjusting steering sliders representing interpretable features.",
            },
            {
                "title": "Questionnaires",
                "description": "When an approach or the full study ends, you will continue to the relevant questionnaire automatically.",
            },
        ],
        "continuation_url": url_for(
            "utils.preference_elicitation",
            continuation_url=url_for(f"{__plugin_name__}.show_features"),
            consuming_plugin=__plugin_name__,
            initial_data_url=url_for(f"{__plugin_name__}.get_initial_data"),
            search_item_url=url_for(f"{__plugin_name__}.item_search")
        ),
    }
    return render_template("study_intro.html", **params)


@bp.route("/get-initial-data", methods=["GET"])
def get_initial_data():
    """Get initial items for preference elicitation"""
    try:
        # Initialize elicitation_movies in session if not present
        if "elicitation_movies" not in session:
            session["elicitation_movies"] = []
        
        el_movies = session["elicitation_movies"]
        
        # Use the working approach from load_data_2
        from plugins.utils.preference_elicitation import load_data_2
        
        x = load_data_2(el_movies)
        
        # Translate genre names
        tr = get_tr(languages, get_lang())
        for i in range(len(x)):
            if "genres" in x[i]:
                x[i]["movie"] = x[i]["movie"] + " " + "|".join([tr(f"genre_{y.lower()}") for y in x[i]["genres"]])
        
        # Add to elicitation_movies
        el_movies.extend(x)
        session["elicitation_movies"] = el_movies
        
        return jsonify(el_movies)
        
    except Exception as e:
        print(f"Error in get_initial_data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@bp.route("/item-search", methods=["GET"])
def item_search():
    """Search for items during preference elicitation"""
    pattern = request.args.get("pattern")
    if not pattern:
        return jsonify([])
    
    try:
        from plugins.utils.preference_elicitation import search_for_movie
        
        lang = get_lang()
        tr = get_tr(languages, lang) if lang != "en" else None
        results = search_for_movie("movie", pattern, tr)

        participation_id = session.get("participation_id")
        if participation_id:
            log_interaction(
                participation_id,
                "elicitation-search",
                query=pattern,
                result_count=len(results),
                result_titles=[r.get("movie", "")[:80] for r in results[:10]],
                phase="elicitation",
            )
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in item_search: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Feature Display & Steering
# ============================================================================

def _select_cluster_features(model_id: str = None, top_k: int = 21) -> list:
    """Pick the best ``top_k`` clusters from semantic_merged for the UI.

    Each returned feature represents one cluster (1+ neurons).
    Scoring per cluster: mean(selectivity) * log(sum(activation_count) + 1).
    Diversity: limit repeated words across cluster labels.
    """
    import re as _re

    STOP_WORDS = {
        'the', 'and', 'of', 'in', 'a', 'an', 'to', 'for', 'with', 'on',
        'at', 'by', 'from', 'its', 'that', 'this', 'but', 'or', 'as',
    }

    effective_model_id = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    sc = _load_semantic_clusters(effective_model_id)
    from .llm_labeling import get_llm_labels
    neuron_stats = get_llm_labels(model_id=effective_model_id)

    candidates = []
    for cluster in sc["clusters"]:
        label = cluster["label"]
        if not label:
            continue
        nids = cluster["neuron_ids"]
        total_act = 0
        sels = []
        for nid in nids:
            info = neuron_stats.get(nid, {})
            total_act += info.get("activation_count", 0)
            sel = info.get("selectivity", 0)
            if sel > 0:
                sels.append(sel)
        mean_sel = np.mean(sels) if sels else 0
        if total_act < 50 or mean_sel < 0.3:
            continue
        score = mean_sel * np.log(total_act + 1)
        candidates.append({
            "cluster_id": cluster["cluster_id"],
            "label": label,
            "description": cluster.get("description", ""),
            "neuron_ids": nids,
            "score": score,
            "total_act": total_act,
        })

    candidates.sort(key=lambda x: -x["score"])

    selected = []
    used_words: dict = {}
    MAX_WORD_USES = 2

    for c in candidates:
        if len(selected) >= top_k:
            break
        words = {
            w for w in _re.split(r'[\s·\-–—/,&]+', c["label"].lower())
            if len(w) > 2 and w not in STOP_WORDS
        }
        if any(used_words.get(w, 0) >= MAX_WORD_USES for w in words):
            continue
        selected.append(c)
        for w in words:
            used_words[w] = used_words.get(w, 0) + 1

    features = []
    for s in selected:
        features.append({
            "id": s["cluster_id"],
            "label": s["label"],
            "category": "latent",
            "description": s["description"],
            "member_ids": s["neuron_ids"],
            "activation": 0.5,
            "movie_count": s["total_act"],
        })
    features.sort(key=lambda f: -f["movie_count"])
    return features


def get_sae_features(top_k: int = 21, model_id: str = None) -> list:
    """Select the most interpretable cluster-level features for the UI."""
    effective_model_id = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    features = _select_cluster_features(model_id=effective_model_id, top_k=top_k)
    print(f"[get_sae_features] Selected {len(features)} cluster features")
    return features


def _personalized_features(
    selected_movies: list,
    model_id: str = None,
    num_sliders: int = 21,
) -> list:
    """Select clusters personalized to the user's elicitation picks.

    For each cluster, compute mean SAE activation across member neurons
    on the user's selected movies.  Rank clusters by that score,
    apply diversity filtering, return top-k.
    """
    import re as _re
    import torch as _torch

    if not selected_movies:
        return get_sae_features(top_k=num_sliders, model_id=model_id)

    effective_model_id = model_id or DEFAULT_TOPK_SAE_MODEL_ID

    from .sae_recommender import get_sae_recommender
    recommender = get_sae_recommender(model_id=effective_model_id)
    recommender.load()

    if recommender.item_features is None or recommender.item_ids is None:
        return get_sae_features(top_k=num_sliders, model_id=model_id)

    id_to_idx = {int(mid): i for i, mid in enumerate(recommender.item_ids)}
    acts = []
    for mid in selected_movies:
        idx = id_to_idx.get(int(mid))
        if idx is not None:
            a = recommender.item_features[idx]
            if isinstance(a, _torch.Tensor):
                a = a.cpu().numpy()
            acts.append(a)

    if not acts:
        return get_sae_features(top_k=num_sliders, model_id=model_id)

    mean_act = np.mean(acts, axis=0)
    print(f"[_personalized_features] {len(acts)}/{len(selected_movies)} movies matched")

    sc = _load_semantic_clusters(effective_model_id)

    STOP_WORDS = {
        'the', 'and', 'of', 'in', 'a', 'an', 'to', 'for', 'with', 'on',
        'at', 'by', 'from', 'its', 'that', 'this', 'but', 'or', 'as',
    }

    candidates = []
    for cluster in sc["clusters"]:
        nids = cluster["neuron_ids"]
        cluster_score = float(np.mean([mean_act[n] for n in nids if n < len(mean_act)]))
        if cluster_score <= 0:
            continue
        total_act = sum(mean_act[n] for n in nids if n < len(mean_act))
        candidates.append({
            "cluster_id": cluster["cluster_id"],
            "label": cluster["label"],
            "description": cluster.get("description", ""),
            "neuron_ids": nids,
            "score": cluster_score,
            "total_act": int(total_act * 100),
        })

    candidates.sort(key=lambda x: -x["score"])

    selected = []
    used_words: dict = {}
    MAX_WORD_USES = 2

    for c in candidates:
        if len(selected) >= num_sliders:
            break
        words = {
            w for w in _re.split(r'[\s·\-–—/,&]+', c["label"].lower())
            if len(w) > 2 and w not in STOP_WORDS
        }
        if any(used_words.get(w, 0) >= MAX_WORD_USES for w in words):
            continue
        selected.append(c)
        for w in words:
            used_words[w] = used_words.get(w, 0) + 1

    if selected:
        print(f"[_personalized_features] {len(selected)} clusters "
              f"(top: {selected[0]['label']} score={selected[0]['score']:.4f})")

    features = []
    for s in selected:
        features.append({
            "id": s["cluster_id"],
            "label": s["label"],
            "category": "latent",
            "description": s["description"],
            "member_ids": s["neuron_ids"],
            "activation": 0.5,
            "movie_count": s["total_act"],
        })
    return features


def _select_slider_features(selected_movies: list, conf: dict, active_model_cfg: dict, num_sliders: int) -> list:
    algorithm = _normalize_feature_selection_algorithm(
        active_model_cfg.get("feature_selection_algorithm", conf.get("feature_selection_algorithm"))
    )
    active_sae_model_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)

    if algorithm == "global_label_topk":
        return get_sae_features(top_k=num_sliders, model_id=active_sae_model_id)

    return _personalized_features(
        selected_movies=selected_movies,
        model_id=active_sae_model_id,
        num_sliders=num_sliders,
    )


@bp.route("/show-features", methods=["GET"])
def show_features():
    """Initialize session state and redirect to steering interface."""
    selected_movies_raw = request.args.get("selectedMovies", "")
    selected_indices = [int(m) for m in selected_movies_raw.split(",") if m]

    # The elicitation frontend sends loader-internal positional indices,
    # but the SAE pipeline works with movieIds.  Convert here once.
    from plugins.utils.data_loading import load_ml_dataset
    loader = load_ml_dataset()
    selected_movies = []
    for idx in selected_indices:
        mid = loader.movie_index_to_id.get(idx)
        if mid is not None:
            selected_movies.append(int(mid))
        else:
            selected_movies.append(idx)

    session["elicitation_selected_movies"] = selected_movies
    session["iteration"] = 1
    session["cumulative_adjustments"] = {}
    session["feature_adjustments"] = {}
    session["boosted_liked_ids"] = []
    session["current_phase"] = 0
    session["phase_data"] = {}
    session["seen_movies_per_phase"] = {}
    session["persistent_liked_by_phase"] = {}
    session["shown_sliders_per_phase"] = {}
    session["steered_sliders_per_phase"] = {}
    session["last_shown_movies_per_phase"] = {}
    session["iteration_preferences_approved"] = False
    session["iteration_locked_final"] = False

    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    _get_effective_models(conf)

    participation_id = session.get("participation_id")
    if participation_id:
        log_interaction(
            participation_id,
            "elicitation-completed",
            selected_movies=selected_movies,
            approach_order=session.get("approach_order"),
        )

    return redirect(url_for(f"{__plugin_name__}.steering_interface"))


@bp.route("/next-phase", methods=["GET"])
def next_phase():
    """Transition to the next sequential phase (Model B after Model A, etc.).
    
    Flow: Phase A done → per-phase questionnaire → Phase B → per-phase questionnaire → overall questionnaire → finish.
    """
    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    if not conf:
        return redirect(url_for(f"{__plugin_name__}.finish_user_study"))

    models = _get_effective_models(conf)
    current_phase = session.get("current_phase", 0)
    next_phase_idx = current_phase + 1
    current_phase_model_name = _get_active_model_config(conf, current_phase).get(
        "name", _approach_label(current_phase)
    )
    phase_questionnaire_title = f"Questionnaire for {current_phase_model_name}"

    participation_id = session.get("participation_id")
    if participation_id:
        approach_name = (
            models[current_phase].get("name", f"Model {current_phase}")
            if current_phase < len(models)
            else "unknown"
        )
        active_model = _get_active_model_config(conf, current_phase)
        # Roll up likes / iterations / slider changes for the closing phase so
        # the journey card can say "Phase 0 complete: model=…, 14 liked, 4
        # iterations, 7 slider changes" instead of the previous "? iterations,
        # ? liked" placeholders.  We read from the canonical Interaction log
        # because session-level counters aren't comprehensive across refreshes.
        total_liked = None
        iterations_used = None
        total_slider_changes = None
        try:
            phase_rows = Interaction.query.filter(
                Interaction.participation == participation_id
            ).all()
            likes_set = set()
            iters_seen = set()
            sliders = 0
            for row in phase_rows:
                try:
                    d = json.loads(row.data) if row.data else {}
                except Exception:
                    continue
                if not isinstance(d, dict):
                    continue
                if d.get("phase") != current_phase and str(d.get("phase")) != str(current_phase):
                    continue
                if row.interaction_type == "movie-feedback" and d.get("action") == "like":
                    mid = d.get("movie_id")
                    if mid is not None:
                        likes_set.add(str(mid))
                if d.get("iteration") is not None:
                    try:
                        iters_seen.add(int(d["iteration"]))
                    except (TypeError, ValueError):
                        pass
                if row.interaction_type == "feature-adjustment":
                    sliders += len(d.get("adjustments", {}) or {})
            total_liked = len(likes_set)
            iterations_used = max(iters_seen) if iters_seen else None
            total_slider_changes = sliders
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[next_phase] phase-complete rollup failed: {exc}")

        log_interaction(
            participation_id,
            "phase-complete",
            phase=current_phase,
            model=approach_name,
            approach_name=approach_name,
            model_id=active_model.get("sae", DEFAULT_TOPK_SAE_MODEL_ID),
            iterations_used=iterations_used,
            total_liked=total_liked,
            total_slider_changes=total_slider_changes,
        )

    if next_phase_idx >= len(models):
        # Last phase completed — show phase questionnaire for this phase, then finish
        phase_questionnaire_file = _get_phase_questionnaire_filename(conf, current_phase)
        if _phase_questionnaire_exists(conf, current_phase):
            session["pending_next_phase"] = None  # Signal: no more phases, go to finish
            return redirect(url_for(
                "utils.final_questionnaire",
                questionnaire_file=phase_questionnaire_file,
                continuation_url=url_for(f"{__plugin_name__}._advance_phase"),
                title_override=phase_questionnaire_title,
                header_override=phase_questionnaire_title,
                hint_override="",
                finish_override="Continue to the rest of the study",
                hide_embedded_questionnaire_heading=1
            ))
        return redirect(url_for(f"{__plugin_name__}.finish_user_study"))

    # Show per-phase questionnaire before moving to next phase (if configured)
    phase_questionnaire_file = _get_phase_questionnaire_filename(conf, current_phase)
    if _phase_questionnaire_exists(conf, current_phase):
        # Store next phase index so _advance_phase can pick it up after questionnaire
        session["pending_next_phase"] = next_phase_idx
        return redirect(url_for(
            "utils.final_questionnaire",
            questionnaire_file=phase_questionnaire_file,
            continuation_url=url_for(f"{__plugin_name__}._advance_phase"),
            title_override=phase_questionnaire_title,
            header_override=phase_questionnaire_title,
            hint_override="",
            finish_override="Continue to the rest of the study",
            hide_embedded_questionnaire_heading=1
        ))

    # No questionnaire — advance immediately
    return _do_advance_phase(next_phase_idx)


@bp.route("/_advance-phase", methods=["GET", "POST"])
def _advance_phase():
    """Called after the per-phase questionnaire completes to actually switch phases.
    
    The questionnaire form POSTs here.  We log the answers, then redirect.
    """
    # Log questionnaire answers if this came from a POST form submission
    if request.method == "POST":
        participation_id = session.get("participation_id")
        if participation_id and "final_questionnaire_data" in request.form:
            data = {}
            for key, val in request.form.items():
                if key in ("final_questionnaire_data", "csrf_token"):
                    continue
                data[key] = val
            current_phase = session.get("current_phase", 0)
            log_interaction(
                participation_id,
                "phase-questionnaire",
                phase=current_phase,
                **data,
            )

    next_phase_idx = session.pop("pending_next_phase", "missing")
    if next_phase_idx is None:
        # Last phase questionnaire done — proceed to final questionnaire / finish
        return redirect(url_for(f"{__plugin_name__}.finish_user_study"))
    if next_phase_idx == "missing":
        return redirect(url_for(f"{__plugin_name__}.steering_interface"))
    return _do_advance_phase(next_phase_idx)


def _do_advance_phase(next_phase_idx):
    """Reset session state and redirect to the steering interface for the next phase."""
    session["current_phase"] = next_phase_idx
    session["iteration"] = 1
    session["cumulative_adjustments"] = {}
    session["feature_adjustments"] = {}
    session["boosted_liked_ids"] = []
    session["iteration_preferences_approved"] = False
    session["iteration_locked_final"] = False
    session.pop("excluded_movies_from_text", None)

    return redirect(url_for(f"{__plugin_name__}.steering_interface"))


@bp.route("/steering-interface", methods=["GET"])
def steering_interface():
    """Main steering interface where users interact with features"""
    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    if not conf:
        conf = _normalize_study_config({
            "enable_comparison": True,
            "models": get_default_models(),
            "interaction_mode": "cumulative",
            "num_iterations": 3,
            "num_recommendations": 20,
            "steering_mode": DEFAULT_STEERING_MODE,
        })
    min_resolution_width, min_resolution_height, min_resolution_error = _get_min_resolution_settings(conf)
    tr = get_tr(languages, get_lang())

    selected_movies = session.get("elicitation_selected_movies", [])

    # Determine the active SAE model id from config
    current_phase_tmp = session.get("current_phase", 0)
    active_model_cfg = _get_active_model_config(conf, current_phase_tmp)
    active_sae_model_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)

    NUM_SLIDERS = conf.get("num_sliders", 16)   # default to two full rows on desktop

    features = _select_slider_features(
        selected_movies=selected_movies,
        conf=conf,
        active_model_cfg=active_model_cfg,
        num_sliders=NUM_SLIDERS,
    )

    features = features[:NUM_SLIDERS]

    sc = _load_semantic_clusters(active_sae_model_id)
    cluster_map = sc["cluster_map"]

    all_clusters_catalog = [
        {
            "id": c["cluster_id"],
            "label": c["label"],
            "description": c.get("description", ""),
            "member_ids": c["neuron_ids"],
            "movie_count": c.get("support", len(c["neuron_ids"])),
        }
        for c in sc["clusters"]
    ]

    session["current_features"] = features
    session["cluster_map"] = cluster_map
    shown_phase = current_phase_tmp if (conf.get("comparison_mode", "side_by_side") == "sequential" and len(_get_effective_models(conf)) >= 2) else 0
    shown_ids = _get_phase_token_set("shown_sliders_per_phase", shown_phase)
    shown_ids.update({str(f.get("id")) for f in features if f.get("id") is not None})
    _set_phase_token_set("shown_sliders_per_phase", shown_phase, shown_ids)

    max_iterations = conf.get("num_iterations", 3)
    comparison_mode = conf.get("comparison_mode", "side_by_side")
    enable_comparison = conf.get("enable_comparison", False)
    interaction_mode = conf.get("interaction_mode", "reset")
    models = _get_effective_models(conf)
    num_recommendations = max(1, int(conf.get("num_recommendations", 20)))

    # --- Sequential mode: show one model at a time ---
    is_sequential = comparison_mode == "sequential" and len(models) >= 2
    current_phase = session.get("current_phase", 0)
    total_phases = len(models) if is_sequential else 1

    if is_sequential:
        enable_comparison = False
        active_model = models[current_phase] if current_phase < len(models) else models[0]
        session["active_model_config"] = active_model
        phase_label = active_model.get("name", f"Approach {chr(65 + current_phase)}")
    model_a_name = models[0].get("name", "Model A") if len(models) > 0 else "Model A"
    model_b_name = models[1].get("name", "Model B") if len(models) > 1 else "Model B"

    next_phase_name = ""
    if is_sequential and current_phase + 1 < len(models):
        next_phase_name = models[current_phase + 1].get("name", f"Model {chr(66 + current_phase)}")
    has_phase_questionnaire_for_current_phase = bool(
        is_sequential and _phase_questionnaire_exists(conf, current_phase)
    )

    # ------------------------------------------------------------------
    # Generate initial recommendations using the user's PROJECTED
    # preference from elicitation — NOT a flat seed for everyone.
    # For each displayed feature (neuron), compute the mean activation
    # of the user's selected movies on that neuron.  This gives each
    # user a unique starting profile that reflects their taste.
    # ------------------------------------------------------------------
    initial_recs_a = []
    initial_recs_b = []
    initial_recs = []
    try:
        import torch as _torch_init
        from plugins.utils.data_loading import load_ml_dataset
        loader = load_ml_dataset(ml_variant=_get_study_dataset_variant(conf))

        # ------------------------------------------------------------------
        # Compute ELSA seed embedding (dense 512-dim collaborative filtering
        # space) from the user's selected movies.  This is the primary
        # "find similar movies" signal — much more effective than sparse
        # SAE activations (k=32) which share almost no neurons even between
        # sequels.
        # ------------------------------------------------------------------
        elsa_seed = None
        seed_genres = set()
        if selected_movies:
            try:
                from .sae_recommender import get_sae_recommender
                _rec = get_sae_recommender(model_id=active_sae_model_id)
                _rec.load()
                if _rec.item_embeddings is not None and _rec.item_ids is not None:
                    _id2idx = {int(mid): i for i, mid in enumerate(_rec.item_ids)}
                    _embs = []
                    for mid in selected_movies:
                        idx = _id2idx.get(int(mid))
                        if idx is not None:
                            emb = _rec.item_embeddings[idx]
                            if isinstance(emb, _torch_init.Tensor):
                                emb = emb.cpu().numpy()
                            _embs.append(emb)
                        try:
                            row = loader.movies_df_indexed.loc[int(mid)]
                            for g in str(row.genres).split("|"):
                                g = g.strip()
                                if g and g != "(no genres listed)":
                                    seed_genres.add(g)
                        except (KeyError, AttributeError):
                            pass
                    if _embs:
                        elsa_seed = np.mean(_embs, axis=0).astype(np.float32)
                        print(f"[steering_interface] ELSA seed: "
                              f"{len(_embs)}/{len(selected_movies)} movies matched, "
                              f"dim={len(elsa_seed)}, genres={seed_genres}")
            except Exception as e:
                print(f"[steering_interface] Could not compute ELSA seed: {e}")
                traceback.print_exc()

        session["elsa_seed"] = elsa_seed.tolist() if elsa_seed is not None else None
        session["elsa_seed_movie_count"] = len(selected_movies) if elsa_seed is not None else 0
        session["seed_genres"] = list(seed_genres)
        session["cumulative_adjustments"] = {}
        session["feature_adjustments"] = {}

        _empty_adj = {}
        if is_sequential:
            seen_for_phase = _get_phase_movie_set("seen_movies_per_phase", current_phase)
            all_excluded_movies = list(set(selected_movies + list(seen_for_phase)))
            _payload = generate_steered_recommendations_for_model(
                loader=loader, selected_movies=all_excluded_movies,
                feature_adjustments=_empty_adj,
                model_config=active_model, k=num_recommendations)
            initial_recs, _initial_debug = _unwrap_recommendation_payload(_payload)
        elif enable_comparison and len(models) >= 2:
            seen_a = _get_phase_movie_set("seen_movies_per_phase", 0)
            seen_b = _get_phase_movie_set("seen_movies_per_phase", 1)
            _payload_a = generate_steered_recommendations_for_model(
                loader=loader, selected_movies=list(set(selected_movies + list(seen_a))),
                feature_adjustments=_empty_adj,
                model_config=models[0], k=num_recommendations)
            _payload_b = generate_steered_recommendations_for_model(
                loader=loader, selected_movies=list(set(selected_movies + list(seen_b))),
                feature_adjustments=_empty_adj,
                model_config=models[1], k=num_recommendations)
            initial_recs_a, _initial_debug_a = _unwrap_recommendation_payload(_payload_a)
            initial_recs_b, _initial_debug_b = _unwrap_recommendation_payload(_payload_b)
        else:
            if models:
                seen_single = _get_phase_movie_set("seen_movies_per_phase", 0)
                _payload = generate_steered_recommendations_for_model(
                    loader=loader, selected_movies=list(set(selected_movies + list(seen_single))),
                    feature_adjustments=_empty_adj,
                    model_config=models[0], k=num_recommendations)
                initial_recs, _initial_debug = _unwrap_recommendation_payload(_payload)
            else:
                initial_recs = generate_steered_recommendations(
                    loader=loader, selected_movies=selected_movies,
                    feature_adjustments=_empty_adj, k=num_recommendations)

        shown_map = _get_phase_id_map("last_shown_movies_per_phase")
        if is_sequential:
            shown_map[str(int(current_phase))] = [int(r.get("movie_idx")) for r in initial_recs if r.get("movie_idx") is not None]
        elif enable_comparison and len(models) >= 2:
            shown_map["0"] = [int(r.get("movie_idx")) for r in initial_recs_a if r.get("movie_idx") is not None]
            shown_map["1"] = [int(r.get("movie_idx")) for r in initial_recs_b if r.get("movie_idx") is not None]
        else:
            shown_map["0"] = [int(r.get("movie_idx")) for r in initial_recs if r.get("movie_idx") is not None]
        session["last_shown_movies_per_phase"] = shown_map
    except Exception as e:
        print(f"[steering_interface] Could not generate initial recs: {e}")
        traceback.print_exc()

    # Determine steering_mode: in sequential mode each model can have its own
    if is_sequential:
        steering_mode = active_model.get("steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE))
    else:
        steering_mode = active_model_cfg.get("steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE))

    title = active_model_cfg.get("name", tr("sae_steering_title"))

    params = {
        "title": title,
        "features": features,
        "iteration": session.get("iteration", 1),
        "max_iterations": max_iterations,
        "steering_mode": steering_mode,
        "submit": tr("get_recommendations"),
        "enable_comparison": enable_comparison,
        "interaction_mode": interaction_mode,
        "model_a_name": model_a_name,
        "model_b_name": model_b_name,
        "initial_recs_a": initial_recs_a,
        "initial_recs_b": initial_recs_b,
        "initial_recs": initial_recs,
        "is_sequential": is_sequential,
        "current_phase": current_phase,
        "total_phases": total_phases,
        "next_phase_name": next_phase_name,
        "has_phase_questionnaire_for_current_phase": has_phase_questionnaire_for_current_phase,
        "seed_adjustments": {},
        "cluster_map": cluster_map,
        "all_clusters_catalog": all_clusters_catalog,
        "feature_selection_algorithm": active_model_cfg.get("feature_selection_algorithm", conf.get("feature_selection_algorithm")),
        "preferences_approved": bool(session.get("iteration_preferences_approved", False)),
        "iteration_locked_final": bool(session.get("iteration_locked_final", False)),
        "num_recommendations": num_recommendations,
        "header_subtitle": _get_steering_subtitle(steering_mode),
        "header_guidance": _get_steering_guidance(steering_mode),
        "min_resolution_width": min_resolution_width,
        "min_resolution_height": min_resolution_height,
        "min_resolution_error": min_resolution_error,
    }

    return render_template("steering_interface.html", **params)


@bp.route("/adjust-features", methods=["POST"])
def adjust_features():
    """
    Apply feature adjustments and get new recommendations.
    
    Supports:
    - Single model mode: returns 'recommendations'
    - A/B comparison mode: returns 'recommendations_a' and 'recommendations_b'
    - Interaction modes: 'reset' (fresh each time) or 'cumulative' (stack adjustments)
    """
    try:
        data = request.get_json(force=True)
        if data is None:
            data = {}
        
        raw_adjustments = data.get("adjustments", {})
        request_interaction_mode = data.get("interaction_mode", "cumulative")
        excluded_movies_from_text = data.get("excluded_movies", [])
        client_liked = [m for m in data.get("liked_movies", []) if m is not None]
        suppressed_genres = data.get("suppressed_features", data.get("suppressed_genres", []))
        search_context = data.get("search_context", {})
        preferences_approved = bool(
            data.get("preferences_approved", session.get("iteration_preferences_approved", False))
        )

        # Expand cluster-level slider deltas into neuron-level deltas
        cluster_map = session.get("cluster_map", {})
        feature_adjustments = _expand_feature_adjustments(
            raw_adjustments=raw_adjustments,
            cluster_map=cluster_map,
        )
        if suppressed_genres:
            print(f"[adjust_features] Suppressed features: {suppressed_genres}")

        conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
        if not conf:
            conf = _normalize_study_config({
                "enable_comparison": True,
                "models": get_default_models(),
                "interaction_mode": request_interaction_mode,
                "num_iterations": 3,
                "num_recommendations": 20,
            })
        max_iterations = conf.get("num_iterations", 3)
        enable_comparison = conf.get("enable_comparison", False)
        interaction_mode = conf.get("interaction_mode", request_interaction_mode)
        models = _get_effective_models(conf)

        # Sequential mode overrides enable_comparison
        comparison_mode_cfg = conf.get("comparison_mode", "side_by_side")
        is_sequential_cfg = comparison_mode_cfg == "sequential" and len(models) >= 2
        current_phase = session.get("current_phase", 0)
        if is_sequential_cfg:
            enable_comparison = False

        num_recommendations = max(1, int(conf.get("num_recommendations", 20)))

        # Determine active SAE model_id for this request
        active_model_cfg = _get_active_model_config(conf)
        active_sae_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
        steering_mode_for_iteration = active_model_cfg.get(
            "steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE)
        )

        if not preferences_approved:
            return jsonify({
                "status": "error",
                "message": "Please confirm your movie selections before continuing.",
                "recommendations": [],
                "recommendations_a": [],
                "recommendations_b": [],
            }), 200

        shown_map = _get_phase_id_map("last_shown_movies_per_phase")
        if is_sequential_cfg:
            _remember_shown_movies(current_phase, shown_map.get(str(int(current_phase)), []))
        elif enable_comparison and len(models) >= 2:
            _remember_shown_movies(0, shown_map.get("0", []))
            _remember_shown_movies(1, shown_map.get("1", []))
        else:
            _remember_shown_movies(0, shown_map.get("0", []))

        # Track cluster-level touches (for slider rotation) from raw_adjustments
        # and accumulate neuron-level weights for the model from expanded feature_adjustments.
        # Scoring is a raw dot-product (item_features @ profile), so the
        # amplification factor controls how strongly a full-range slider
        # move shifts the ranking relative to the elicitation seed.
        SLIDER_AMP = 2.0
        previous_adjustments = session.get("cumulative_adjustments", {})
        user_touched = set(session.get("user_touched_features", []))

        for key, val in (raw_adjustments or {}).items():
            if abs(float(val)) > 0.001:
                user_touched.add(str(key))

        for key, val in feature_adjustments.items():
            skey = str(key)
            prev = float(previous_adjustments.get(skey, 0))
            raw_delta = float(val)
            new = raw_delta * SLIDER_AMP
            if abs(raw_delta) > 0.001:
                previous_adjustments[skey] = round(prev + new, 4)
            elif skey in previous_adjustments and abs(prev) < 0.001:
                del previous_adjustments[skey]

        session["cumulative_adjustments"] = previous_adjustments
        session["user_touched_features"] = list(user_touched)

        # For the model: pass ALL non-zero adjustments (seed + slider deltas).
        model_adjustments = {
            k: v for k, v in previous_adjustments.items() if abs(float(v)) > 0.001
        }
        feature_adjustments = model_adjustments
        session["feature_adjustments"] = previous_adjustments

        # ---- Persistent selections (per approach/phase) ----
        # Store the *current* liked set for this phase (replace semantics),
        # not a monotonic union. This ensures de-selections are respected.
        active_phase_for_profile = current_phase if is_sequential_cfg else 0
        current_liked_set = {int(m) for m in client_liked if m is not None}
        _set_phase_movie_set("persistent_liked_by_phase", active_phase_for_profile, current_liked_set)

        # ---- Liked-movie boost via ELSA seed update ----
        # Instead of boosting individual SAE neurons (which are too sparse
        # to be meaningful), we shift the ELSA seed embedding toward liked
        # movies.  This directly improves the collaborative-filtering signal.
        current_liked = set(int(m) for m in client_liked)
        already_boosted = set(int(m) for m in session.get("boosted_liked_ids", []))
        new_likes = [m for m in current_liked if m not in already_boosted]
        removed_likes = [m for m in already_boosted if m not in current_liked]

        # Calibrate like influence by interface:
        # - slider-based phases already have an explicit steering channel
        #   => reduce like contribution to keep channels separable
        # - non-steering phases rely on likes as the main explicit signal
        effective_mode = _normalize_steering_mode(steering_mode_for_iteration)
        like_weight = 0.25 if effective_mode in {"sliders", "both", "toggles"} else 0.5

        if new_likes or removed_likes:
            _update_elsa_seed_with_likes(
                current_liked, active_sae_id, LIKE_WEIGHT=like_weight, LIKE_CAP=10,
            )

        session["boosted_liked_ids"] = list(current_liked)

        # Also update genre set from liked movies
        # Recompute from elicitation + current likes to honor de-selections.
        from plugins.utils.data_loading import load_ml_dataset as _load_ds
        _ldr = _load_ds(ml_variant=_get_study_dataset_variant(conf))
        seed_genres = set()
        for mid in session.get("elicitation_selected_movies", []):
            try:
                row = _ldr.movies_df_indexed.loc[int(mid)]
                for g in str(row.genres).split("|"):
                    g = g.strip()
                    if g and g != "(no genres listed)":
                        seed_genres.add(g)
            except (KeyError, AttributeError):
                pass
        for mid in current_liked:
            try:
                row = _ldr.movies_df_indexed.loc[int(mid)]
                for g in str(row.genres).split("|"):
                    g = g.strip()
                    if g and g != "(no genres listed)":
                        seed_genres.add(g)
            except (KeyError, AttributeError):
                pass
        session["seed_genres"] = list(seed_genres)

        current_iteration = session.get("iteration", 1)
        participation_id = session.get("participation_id")
        if participation_id:
            active_phase = session.get("current_phase", 0)
            approach_name = _get_active_model_config(conf, active_phase).get(
                "name", f"Model {active_phase}"
            )
            log_interaction(
                participation_id,
                "feature-adjustment",
                iteration=current_iteration,
                phase=active_phase,
                model_id=active_sae_id,
                approach_name=approach_name,
                steering_mode=steering_mode_for_iteration,
                adjustments=feature_adjustments,
                interaction_mode=interaction_mode,
                enable_comparison=enable_comparison,
                excluded_movies=excluded_movies_from_text,
                liked_movies=client_liked,
                search_context=search_context,
                negative_adjustment_ids=[int(k) for k, v in feature_adjustments.items() if float(v) < 0],
                negative_adjustment_count=sum(1 for v in feature_adjustments.values() if float(v) < 0),
            )

        selected_movies = session.get("elicitation_selected_movies", [])
        # Exclude explicit text blocks and keep per-approach recommendation
        # history disjoint across iterations.
        excluded_movie_ids = list(set((excluded_movies_from_text or [])))
        if excluded_movie_ids:
            session["excluded_movies_from_text"] = excluded_movie_ids

        from plugins.utils.data_loading import load_ml_dataset
        loader = load_ml_dataset(ml_variant=_get_study_dataset_variant(conf))

        # --- Sequential mode detection ---
        comparison_mode = conf.get("comparison_mode", "side_by_side")
        is_sequential = comparison_mode == "sequential" and len(models) >= 2
        current_phase = session.get("current_phase", 0)
        total_phases = len(models) if is_sequential else 1

        is_final_iteration_in_phase = (current_iteration + 1) > max_iterations
        if is_sequential:
            is_final_of_study = is_final_iteration_in_phase and (current_phase + 1) >= total_phases
        else:
            is_final_of_study = is_final_iteration_in_phase

        response_data = {
            "status": "success",
            "iteration": current_iteration + 1,
            "max_iterations": max_iterations,
            "is_final_iteration": is_final_iteration_in_phase,
            "is_final_of_study": is_final_of_study,
            "is_sequential": is_sequential,
            "current_phase": current_phase,
            "total_phases": total_phases,
            "interaction_mode": interaction_mode
        }
        debug_insights = {}

        if is_sequential:
            active_model = session.get("active_model_config", models[current_phase])
            seen_in_phase = _get_phase_movie_set("seen_movies_per_phase", current_phase)
            all_excluded_movies = list(set(selected_movies + excluded_movie_ids + list(seen_in_phase)))
            _payload = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=all_excluded_movies,
                feature_adjustments=feature_adjustments,
                model_config=active_model,
                k=num_recommendations,
                suppressed_genres=suppressed_genres,
            )
            recommendations, debug_insights = _unwrap_recommendation_payload(_payload)
            response_data["recommendations"] = recommendations

            if participation_id:
                log_interaction(
                    participation_id,
                    "recommendations-shown",
                    iteration=current_iteration,
                    phase=current_phase,
                    model=active_model.get("name", f"Phase {current_phase}"),
                    movies=[r.get('movie_idx') for r in recommendations]
                )
            shown_map = _get_phase_id_map("last_shown_movies_per_phase")
            shown_map[str(int(current_phase))] = [int(r.get("movie_idx")) for r in recommendations if r.get("movie_idx") is not None]
            session["last_shown_movies_per_phase"] = shown_map

        elif enable_comparison and len(models) >= 2:
            # A/B Comparison Mode - generate recommendations for both models
            model_a_config = models[0]
            model_b_config = models[1]
            
            print(f"[A/B Comparison] Model A config: {model_a_config}")
            print(f"[A/B Comparison] Model B config: {model_b_config}")
            
            seen_a = _get_phase_movie_set("seen_movies_per_phase", 0)
            seen_b = _get_phase_movie_set("seen_movies_per_phase", 1)
            all_excluded_movies_a = list(set(selected_movies + excluded_movie_ids + list(seen_a)))
            all_excluded_movies_b = list(set(selected_movies + excluded_movie_ids + list(seen_b)))
            
            _payload_a = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=all_excluded_movies_a,
                feature_adjustments=feature_adjustments,
                model_config=model_a_config,
                k=num_recommendations,
                suppressed_genres=suppressed_genres,
            )
            
            _payload_b = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=all_excluded_movies_b,
                feature_adjustments=feature_adjustments,
                model_config=model_b_config,
                k=num_recommendations,
                suppressed_genres=suppressed_genres,
            )
            recommendations_a, debug_a = _unwrap_recommendation_payload(_payload_a)
            recommendations_b, debug_b = _unwrap_recommendation_payload(_payload_b)
            debug_insights = {"model_a": debug_a, "model_b": debug_b}
            
            print(f"[A/B Comparison] Model A produced {len(recommendations_a)} recommendations (requested: {num_recommendations})")
            print(f"[A/B Comparison] Model B produced {len(recommendations_b)} recommendations (requested: {num_recommendations})")
            
            # Ensure we return the requested number
            if len(recommendations_a) < num_recommendations:
                print(f"[A/B Comparison] WARNING: Model A returned only {len(recommendations_a)} recommendations, expected {num_recommendations}")
            if len(recommendations_b) < num_recommendations:
                print(f"[A/B Comparison] WARNING: Model B returned only {len(recommendations_b)} recommendations, expected {num_recommendations}")
            
            response_data["recommendations_a"] = recommendations_a
            response_data["recommendations_b"] = recommendations_b
            response_data["recommendations"] = recommendations_a  # Fallback for backward compatibility
            
            # Log which recommendations were shown
            if participation_id:
                log_interaction(
                    participation_id,
                    "recommendations-shown",
                    iteration=current_iteration,
                    model_a=[r.get('movie_idx') for r in recommendations_a],
                    model_b=[r.get('movie_idx') for r in recommendations_b]
                )
            shown_map = _get_phase_id_map("last_shown_movies_per_phase")
            shown_map["0"] = [int(r.get("movie_idx")) for r in recommendations_a if r.get("movie_idx") is not None]
            shown_map["1"] = [int(r.get("movie_idx")) for r in recommendations_b if r.get("movie_idx") is not None]
            session["last_shown_movies_per_phase"] = shown_map
        else:
            # Single model mode
            seen_single = _get_phase_movie_set("seen_movies_per_phase", 0)
            all_excluded_movies = list(set(selected_movies + excluded_movie_ids + list(seen_single)))
            if models:
                recommendations = generate_steered_recommendations_for_model(
                    loader=loader,
                    selected_movies=all_excluded_movies,
                    feature_adjustments=feature_adjustments,
                    model_config=models[0],
                    k=num_recommendations,
                    suppressed_genres=suppressed_genres,
                )
            else:
                recommendations = generate_steered_recommendations(
                    loader=loader,
                    selected_movies=all_excluded_movies,
                    feature_adjustments=feature_adjustments,
                    k=num_recommendations
                )
                debug_insights = {}
            response_data["recommendations"] = recommendations
            shown_map = _get_phase_id_map("last_shown_movies_per_phase")
            shown_map["0"] = [int(r.get("movie_idx")) for r in recommendations if r.get("movie_idx") is not None]
            session["last_shown_movies_per_phase"] = shown_map
        
        # Build updated_features depending on steering mode
        _sm = "sliders"
        if is_sequential_cfg and models:
            cp = session.get("current_phase", 0)
            _sm = models[min(cp, len(models) - 1)].get("steering_mode", conf.get("steering_mode", "sliders"))
        elif models:
            _sm = models[0].get("steering_mode", conf.get("steering_mode", "sliders"))
        else:
            _sm = conf.get("steering_mode", "sliders")

        if _sm == "text":
            updated_features = []
            response_data["updated_features"] = updated_features
        elif _sm in ("sliders", "both", "toggles"):
            NUM_SLIDERS = conf.get("num_sliders", 16)
            current_features = session.get("current_features", [])
            touched_cluster_adjustments = {
                cid: 1.0 for cid in user_touched if cid.startswith("cluster_")
            }
            updated_features = _compute_updated_sliders(
                current_features=current_features,
                cumulative_adjustments=touched_cluster_adjustments,
                liked_movie_ids=list(current_liked_set),
                model_id=active_sae_id,
                num_sliders=NUM_SLIDERS,
                phase_idx=active_phase_for_profile,
            )
            if updated_features and updated_features != current_features:
                session["current_features"] = updated_features
                response_data["updated_features"] = updated_features
                old_ids = {f['id'] for f in current_features}
                new_count = len([f for f in updated_features if f['id'] not in old_ids])
                print(f"[adjust_features] Sliders refreshed: {new_count} new features")

        # Optional debug payload for research interpretability.
        response_data["debug_insights"] = debug_insights or {}
        if debug_insights:
            if "influence_level" in debug_insights:
                print(
                    "[adjust_features] influence="
                    f"{debug_insights.get('influence_level')} "
                    f"(gamma={debug_insights.get('adaptive_gamma')}, "
                    f"ratio={debug_insights.get('steering_ratio')})"
                )
            else:
                for model_key in ("model_a", "model_b"):
                    dbg = (debug_insights or {}).get(model_key, {})
                    if dbg:
                        print(
                            f"[adjust_features] {model_key} influence="
                            f"{dbg.get('influence_level')} "
                            f"(gamma={dbg.get('adaptive_gamma')}, "
                            f"ratio={dbg.get('steering_ratio')})"
                        )

        # Increment iteration
        session["iteration"] = current_iteration + 1
        session["iteration_preferences_approved"] = False
        session["iteration_locked_final"] = bool(is_final_iteration_in_phase)
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in adjust_features: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "recommendations": [],
            "recommendations_a": [],
            "recommendations_b": []
        }), 200


def _compute_genre_bonus(recommender, loader, seed_genres: set) -> np.ndarray:
    """Pre-compute Jaccard genre similarity for every item vs the seed genres."""
    n_items = len(recommender.item_ids)
    bonus = np.zeros(n_items, dtype=np.float32)
    if not seed_genres:
        return bonus
    for i, mid in enumerate(recommender.item_ids):
        mid = int(mid)
        try:
            row = loader.movies_df_indexed.loc[mid]
            item_genres = {
                g.strip() for g in str(row.genres).split("|")
                if g.strip() and g.strip() != "(no genres listed)"
            }
            if item_genres:
                overlap = len(item_genres & seed_genres)
                union = len(item_genres | seed_genres)
                bonus[i] = overlap / union
        except (KeyError, AttributeError):
            pass
    return bonus


def _unwrap_recommendation_payload(payload):
    """Normalize recommendation function output to (recs, debug)."""
    if isinstance(payload, dict):
        return payload.get("recommendations", []), payload.get("debug", {})
    return payload or [], {}


def generate_steered_recommendations_for_model(loader, selected_movies, feature_adjustments, model_config, k=20, suppressed_genres=None):
    """
    Generate recommendations for a specific model configuration (for A/B testing).
    
    Uses hybrid scoring: ELSA cosine (collaborative filtering) + genre Jaccard
    + SAE neuron adjustments (from sliders/likes).
    """
    suppressed_genres = suppressed_genres or []
    sae_model_id = model_config.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    
    try:
        from .sae_recommender import get_sae_recommender
        
        recommender = get_sae_recommender(model_id=sae_model_id)
        recommender.load()

        if recommender.item_features is None or recommender.item_ids is None:
            print(
                "[generate_steered_recs] SAE runtime activations "
                "missing; falling back to metadata-based recommendations"
            )
            return _fallback_genre_recommendations(loader, selected_movies, feature_adjustments, k)
        
        neuron_adjustments = {
            int(key): float(value) for key, value in feature_adjustments.items()
        }
        n_adj = sum(1 for v in neuron_adjustments.values() if abs(v) > 0.001)
        print(f"[generate_steered_recs] SAE model={sae_model_id}, "
              f"non-zero adjustments={n_adj}")
        
        exclude_movie_ids = []
        for movie_ref in selected_movies:
            try:
                exclude_movie_ids.append(int(movie_ref))
            except (ValueError, TypeError):
                continue
        
        allowed_ids = set(loader.movies_df_indexed.index.tolist())

        # Retrieve ELSA seed embedding and genre set from session
        elsa_seed_list = session.get("elsa_seed")
        elsa_seed = np.array(elsa_seed_list, dtype=np.float32) if elsa_seed_list else None
        seed_genres = set(session.get("seed_genres", []))

        genre_bonus = _compute_genre_bonus(recommender, loader, seed_genres) if seed_genres else None

        if elsa_seed is not None:
            print(f"[generate_steered_recs] ELSA seed active, genres={seed_genres}")

        rec_payload = recommender.get_recommendations(
            feature_adjustments=neuron_adjustments,
            n_items=max(k * 15, 300),
            exclude_items=exclude_movie_ids,
            allowed_ids=allowed_ids,
            seed_embedding=elsa_seed,
            genre_bonus=genre_bonus,
            return_debug=True,
        )
        raw_recommendations = rec_payload.get("results", []) if isinstance(rec_payload, dict) else rec_payload
        debug_payload = rec_payload.get("debug", {}) if isinstance(rec_payload, dict) else {}
        print(f"[generate_steered_recs] Raw recommendations: {len(raw_recommendations)}")
        if debug_payload:
            print(
                "[generate_steered_recs] influence="
                f"{debug_payload.get('influence_level')} "
                f"(gamma={debug_payload.get('adaptive_gamma')}, "
                f"clamp={debug_payload.get('steering_clamp')}, "
                f"ratio={debug_payload.get('steering_ratio')})"
            )
            for item in (debug_payload.get("top_up") or [])[:5]:
                print(
                    "[generate_steered_recs] top_up "
                    f"movie_id={item.get('movie_id')} "
                    f"delta={item.get('rank_delta')} "
                    f"base={item.get('base_rank')}->final={item.get('final_rank')} "
                    f"(cf={item.get('cf_score')}, genre={item.get('genre_score')}, "
                    f"steer={item.get('steering_score')})"
                )
            for item in (debug_payload.get("top_down") or [])[:5]:
                print(
                    "[generate_steered_recs] top_down "
                    f"movie_id={item.get('movie_id')} "
                    f"delta={item.get('rank_delta')} "
                    f"base={item.get('base_rank')}->final={item.get('final_rank')} "
                    f"(cf={item.get('cf_score')}, genre={item.get('genre_score')}, "
                    f"steer={item.get('steering_score')})"
                )
        
        # Format results - include image URLs; skip items with missing metadata or unknown IDs
        overviews = _load_tmdb_overviews()
        results = []
        skipped_missing_meta = []
        skipped_unknown_id = []
        for rec in raw_recommendations:
            movie_id = rec.get('movie_id')
            title = None
            genres = []
            image_url = None
            
            # Skip if movie_id not in known dataset
            if movie_id not in loader.movies_df_indexed.index:
                skipped_unknown_id.append(movie_id)
                continue
            
            try:
                movie_info = loader.movies_df_indexed.loc[movie_id]
                title = movie_info.title
                genres = movie_info.genres.split("|")
                try:
                    movie_idx = loader.movie_id_to_index[movie_id]
                    image_url = loader.get_image(movie_idx)
                except (KeyError, AttributeError):
                    image_url = None
            except (KeyError, AttributeError):
                # Try to recover from the full movies_df by movieId
                try:
                    fallback_row = loader.movies_df[loader.movies_df.movieId == movie_id]
                    if not fallback_row.empty:
                        title = fallback_row.iloc[0].title
                        genres = fallback_row.iloc[0].genres.split("|")
                        image_url = None
                except Exception:
                    pass
            
            if not title:
                skipped_missing_meta.append(movie_id)
                continue

            overview_text = overviews.get(movie_id, "")

            # Skip if movie has a suppressed genre
            if suppressed_genres:
                has_suppressed = any(sg in genres for sg in suppressed_genres)
                if has_suppressed:
                    continue  # Filter out movies with suppressed genres
            
            results.append({
                "title": title,
                "movie_idx": movie_id,
                "score": rec.get('score', 0.5),
                "metadata": " | ".join([g for g in genres if g != "(no genres listed)"]),
                "matched_features": rec.get('matched_features', {}),
                "model": model_config.get("id", "unknown"),
                "url": image_url,
                "overview": overview_text
            })
            
            if len(results) >= k:
                break
        
        if skipped_unknown_id:
            print(f"[generate_steered_recommendations_for_model] Skipped {len(skipped_unknown_id)} items with unknown IDs, sample: {skipped_unknown_id[:10]}")
            unknown_ratio = len(skipped_unknown_id) / max(len(raw_recommendations), 1)
            if unknown_ratio > 0.25:
                print(
                    "[generate_steered_recommendations_for_model] WARNING: high unknown-ID drop ratio "
                    f"({unknown_ratio:.1%}); this usually indicates dataset/model cache mismatch."
                )
        if skipped_missing_meta:
            sample_ids = skipped_missing_meta[:10]
            print(f"[generate_steered_recommendations_for_model] Skipped {len(skipped_missing_meta)} items with missing metadata, sample: {sample_ids}")
        if len(results) < k:
            print(f"[generate_steered_recommendations_for_model] WARNING: only {len(results)} results after filtering (target {k})")
        print(f"[generate_steered_recommendations_for_model] Returning {len(results)} recommendations (target {k})")
        return {
            "recommendations": results[:k],
            "debug": debug_payload,
        }
        
    except Exception as e:
        print(f"[generate_steered_recommendations_for_model] Error: {e}")
        traceback.print_exc()
        # Fallback to genre-based
        return _fallback_genre_recommendations(loader, selected_movies, feature_adjustments, k)


def generate_steered_recommendations(loader, selected_movies, feature_adjustments, k=20):
    """
    Legacy wrapper — delegates to the model-aware version with the default
    TopK WWW SAE config.  Kept for backward-compatibility with call
    sites that don't have a model_config dict handy.
    """
    default_config = {"sae": DEFAULT_TOPK_SAE_MODEL_ID}
    payload = generate_steered_recommendations_for_model(
        loader=loader,
        selected_movies=selected_movies,
        feature_adjustments=feature_adjustments,
        model_config=default_config,
        k=k,
    )
    if isinstance(payload, dict):
        return payload.get("recommendations", [])
    return payload


def _fallback_genre_recommendations(loader, selected_movies, feature_adjustments, k=20):
    """Fallback genre-based recommendations when SAE is not available."""
    # Map feature IDs to genre preferences
    feature_to_genres = {
        "0": ["Action", "Adventure"],
        "1": ["Drama"],
        "2": ["Comedy"],
        "3": ["Sci-Fi", "Fantasy"],
        "4": ["Thriller", "Mystery"],
        "5": ["Romance"],
        "6": ["Horror"],
        "7": ["Animation", "Children"],
        "8": ["Documentary"],
        "9": ["War", "Western", "Film-Noir"],
    }
    
    # Calculate genre weights based on adjustments
    genre_weights = {}
    for feature_id, adjustment in feature_adjustments.items():
        genres = feature_to_genres.get(str(feature_id), [])
        for genre in genres:
            genre_weights[genre] = genre_weights.get(genre, 1.0) * float(adjustment)
    
    # Get all movie indices
    all_movie_indices = list(loader.movie_index_to_id.keys())
    
    # Filter out already selected movies
    candidate_indices = [idx for idx in all_movie_indices if idx not in selected_movies]
    
    # Score each candidate movie
    scored_movies = []
    for movie_idx in candidate_indices[:500]:  # Limit for performance
        try:
            movie_id = loader.movie_index_to_id[movie_idx]
            movie_genres = loader.movies_df_indexed.loc[movie_id].genres.split("|")
            title = loader.movies_df_indexed.loc[movie_id].title
            
            # Base score from genre overlap with selected movies
            base_score = 0.5
            
            # Adjust score based on genre weights
            genre_score = 0.0
            for genre in movie_genres:
                if genre in genre_weights:
                    genre_score += genre_weights[genre] - 1.0
            
            final_score = base_score + genre_score * 0.3
            final_score = max(0.0, min(1.0, final_score))

            # Try to get image URL
            try:
                image_url = loader.get_image(movie_idx)
            except (KeyError, AttributeError):
                image_url = None

            scored_movies.append({
                "title": title,
                "movie_idx": movie_id,
                "score": final_score,
                "metadata": " | ".join([g for g in movie_genres if g != "(no genres listed)"]),
                "url": image_url,
            })
        except (KeyError, AttributeError):
            continue
    
    scored_movies.sort(key=lambda x: -x["score"])
    return scored_movies[:k]


def _compute_updated_sliders(
    current_features: list,
    cumulative_adjustments: dict,
    liked_movie_ids: list,
    model_id: str = None,
    num_sliders: int = 21,
    phase_idx: int = 0,
) -> list:
    """Recompute cluster slider list after an iteration.

    Simple queue policy:
    - sliders touched by user are marked as "steered" and removed from
      future auto-pools (unless explicitly searched/edited),
    - sliders already shown are not re-shown,
    - exploit panel is derived primarily from last shown recommendations in the
      active phase (plus likes), then global fallback.
    """
    touched_ids = set()
    for fid_str, val in cumulative_adjustments.items():
        if fid_str.startswith("cluster_") and abs(float(val)) > 0.001:
            touched_ids.add(fid_str)

    shown_ids = _get_phase_token_set("shown_sliders_per_phase", phase_idx)
    steered_ids = _get_phase_token_set("steered_sliders_per_phase", phase_idx)

    if touched_ids:
        steered_ids.update({str(cid) for cid in touched_ids})
        _set_phase_token_set("steered_sliders_per_phase", phase_idx, steered_ids)

    # Exploit source: strictly current recommendation context in this phase.
    last_shown_map = _get_phase_id_map("last_shown_movies_per_phase")
    last_shown_phase = last_shown_map.get(str(int(phase_idx)), [])
    profile_movies = sorted({int(mid) for mid in last_shown_phase if mid is not None})

    profile_pool = []
    if profile_movies:
        profile_pool = _personalized_features(
            selected_movies=profile_movies,
            model_id=model_id,
            num_sliders=max(num_sliders * 4, num_sliders + 24),
        )
    # Global pool order is already a top-ranked ordering from
    # cluster selection (interpretable/high-coverage clusters first).
    global_pool = get_sae_features(top_k=num_sliders * 8, model_id=model_id)

    selected = []
    used_ids = set()
    seen_labels = set()
    source_counts = {"exploit": 0, "explore": 0}

    # Only-exploit policy: all visible sliders come from phase-local
    # recommendation context. Global pool is used only as a safety fallback.
    exploit_target = num_sliders
    explore_target = 0

    def _append_from_pool(pool: list, target_size: int, source: str, allow_shown: bool = False):
        if not pool:
            return
        for gf in pool:
            if len(selected) >= target_size:
                break
            cid = str(gf.get("id"))
            if cid in used_ids:
                continue
            if (not allow_shown) and cid in shown_ids:
                continue
            if cid in steered_ids:
                continue
            if _is_near_duplicate_label(gf.get("label", ""), seen_labels):
                continue
            selected.append(gf)
            used_ids.add(cid)
            seen_labels.add(_normalize_label(gf.get("label", "")))
            source_counts[source] = source_counts.get(source, 0) + 1

    # Stage 1: exploit (profile-conditioned sliders)
    _append_from_pool(profile_pool, exploit_target, source="exploit", allow_shown=False)

    # Stage 2: top-up unseen exploit frontier if needed
    _append_from_pool(profile_pool, num_sliders, source="exploit", allow_shown=False)

    # Safety fallback: if exploit frontier is exhausted, use global unseen
    # non-steered clusters to avoid empty UI.
    if len(selected) < num_sliders:
        _append_from_pool(global_pool, num_sliders, source="explore", allow_shown=True)
    if len(selected) < num_sliders:
        _append_from_pool(profile_pool, num_sliders, source="exploit", allow_shown=True)

    shown_ids.update({str(f.get("id")) for f in selected if f.get("id") is not None})
    _set_phase_token_set("shown_sliders_per_phase", phase_idx, shown_ids)

    print(
        f"[_compute_updated_sliders] phase={phase_idx} "
        f"touched={len(touched_ids)} shown_pool={len(shown_ids)} "
        f"steered_pool={len(steered_ids)} returned={len(selected)} "
        f"(exploit={source_counts.get('exploit', 0)}, explore={source_counts.get('explore', 0)})"
    )
    return selected[:num_sliders]


def _update_elsa_seed_with_likes(
    current_liked_ids: set,
    model_id: str = None,
    LIKE_WEIGHT: float = 0.5,
    LIKE_CAP: int = 10,
):
    """Recompute the ELSA seed embedding to incorporate liked movies.

    The seed is a weighted average of:
      * original elicitation movies  (weight = 1.0 each)
      * currently liked movies       (weight = LIKE_WEIGHT each)

    We recompute from scratch each time to cleanly handle un-likes.
    To avoid profile drift, liked contribution is saturated to ``LIKE_CAP``.
    """
    import torch as _torch
    try:
        from .sae_recommender import get_sae_recommender
        rec = get_sae_recommender(model_id=model_id)
        rec.load()
        if rec.item_embeddings is None or rec.item_ids is None:
            return

        id_to_idx = {int(mid): i for i, mid in enumerate(rec.item_ids)}

        # Reconstruct the original elicitation seed
        original_movies = session.get("elicitation_selected_movies", [])
        original_count = session.get("elsa_seed_movie_count", len(original_movies))

        weighted_sum = np.zeros(rec.item_embeddings.shape[1], dtype=np.float32)
        total_weight = 0.0

        for mid in original_movies:
            idx = id_to_idx.get(int(mid))
            if idx is not None:
                emb = rec.item_embeddings[idx]
                if isinstance(emb, _torch.Tensor):
                    emb = emb.cpu().numpy()
                weighted_sum += emb.astype(np.float32)
                total_weight += 1.0

        liked_sorted = sorted(int(x) for x in current_liked_ids)
        effective_liked = liked_sorted[: max(0, int(LIKE_CAP))]

        for mid in effective_liked:
            idx = id_to_idx.get(int(mid))
            if idx is not None:
                emb = rec.item_embeddings[idx]
                if isinstance(emb, _torch.Tensor):
                    emb = emb.cpu().numpy()
                weighted_sum += emb.astype(np.float32) * LIKE_WEIGHT
                total_weight += LIKE_WEIGHT

        if total_weight > 0:
            new_seed = weighted_sum / total_weight
            session["elsa_seed"] = [round(float(v), 6) for v in new_seed]
            n_liked = len(current_liked_ids)
            n_effective = len(effective_liked)
            print(f"[_update_elsa_seed_with_likes] Updated seed: "
                  f"{len(original_movies)} elicitation + {n_liked} liked "
                  f"(effective={n_effective}, cap={LIKE_CAP}, weight={LIKE_WEIGHT})")
    except Exception as e:
        print(f"[_update_elsa_seed_with_likes] Error: {e}")
        traceback.print_exc()


def _boost_from_liked_movies(liked_movie_ids: list, strength: float = 0.30, model_id: str = None) -> dict:
    """Derive neuron boosts from the SAE activations of liked movies.

    For each liked movie we find its SAE activation vector and pick the
    top-5 most active neurons, adding a weighted boost for each.
    """
    import torch as _torch
    try:
        from .sae_recommender import get_sae_recommender
        recommender = get_sae_recommender(model_id=model_id)
        recommender.load()

        if recommender.item_features is None or recommender.item_ids is None:
            return {}

        # Build movieId → positional index lookup (once)
        id_to_idx = {int(mid): i for i, mid in enumerate(recommender.item_ids)}

        boost = {}
        for mid in liked_movie_ids:
            mid = int(mid)
            idx = id_to_idx.get(mid)
            if idx is None:
                continue
            acts = recommender.item_features[idx]
            if isinstance(acts, _torch.Tensor):
                acts = acts.cpu().numpy()
            top_neurons = np.argsort(acts)[-5:]
            for nid in top_neurons:
                nid = int(nid)
                act_val = float(acts[nid])
                if act_val > 0:
                    boost[nid] = boost.get(nid, 0) + act_val * strength
        return boost
    except Exception as e:
        print(f"[_boost_from_liked_movies] Error: {e}")
        traceback.print_exc()
        return {}


@bp.route("/text-to-adjustments", methods=["POST"])
def text_to_adjustments_endpoint():
    """Convert natural language text to SAE neuron adjustments.

    Matches the user's query against LLM-labeled neuron concepts using
    sentence-BERT.  Returns the neuron adjustments **and** the top-K
    most impacted features for display in the UI as editable sliders.
    """
    try:
        from .text_steering import text_to_concept_adjustments, get_matched_tags, get_neuron_labels

        data = request.get_json(force=True)
        text = data.get("text", "")

        conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
        active_sae_id = _get_active_sae_model_id(conf)

        if not text.strip():
            return jsonify({
                "status": "error",
                "message": "No text provided",
                "adjustments": {},
                "matched_tags": [],
                "top_features": [],
            })

        raw_adjustments, top_features = text_to_concept_adjustments(
            text, model_id=active_sae_id, top_k=7, sensitivity=1.0,
        )

        adjustments = {
            str(nid): float(round(v, 3))
            for nid, v in raw_adjustments.items()
        }

        neuron_labels = {}
        try:
            neuron_labels = {
                str(k): v
                for k, v in get_neuron_labels(list(raw_adjustments.keys()), model_id=active_sae_id).items()
            }
        except Exception:
            pass

        matched_tags = []
        try:
            matched_tags = [
                {"tag": tag, "score": float(round(sc, 2)), "direction": d}
                for tag, sc, d in get_matched_tags(text, top_k=8, model_id=active_sae_id)
            ]
        except Exception:
            pass

        participation_id = session.get("participation_id")
        if participation_id:
            log_interaction(
                participation_id,
                "text-query",
                iteration=session.get("iteration", 1),
                phase=session.get("current_phase", 0),
                model_id=active_sae_id,
                text=text,
                matched_tags=matched_tags,
                top_features=[f.get("neuron_id") for f in top_features],
            )

        return jsonify({
            "status": "success",
            "adjustments": adjustments,
            "matched_tags": matched_tags,
            "neuron_labels": neuron_labels,
            "top_features": top_features,
            "excluded_movies": [],
            "parsed_text": text,
        })

    except FileNotFoundError as e:
        return jsonify({
            "status": "error",
            "message": f"LLM label file not found: {e}",
            "adjustments": {},
            "matched_tags": [],
            "top_features": [],
        }), 200
    except ImportError:
        return jsonify({
            "status": "error",
            "message": "sentence-transformers not installed. Run: pip install sentence-transformers",
            "adjustments": {},
            "matched_tags": [],
            "top_features": [],
        }), 200
    except Exception as e:
        print(f"Error in text_to_adjustments: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e),
            "adjustments": {},
            "matched_tags": [],
            "top_features": [],
        }), 200


# ============================================================================
# Feature Search (search all SAE neurons by label)
# ============================================================================

@bp.route("/search-features", methods=["GET"])
def search_features():
    """Search all clusters by label (substring match).

    Returns JSON list of {id, label, description, member_ids}.
    """
    query = request.args.get("q", "").strip().lower()
    if len(query) < 2:
        return jsonify([])

    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    active_sae_id = _get_active_sae_model_id(conf)

    try:
        sc = _load_semantic_clusters(active_sae_id)
    except Exception as e:
        print(f"[search-features] Error: {e}")
        return jsonify([])

    current_feature_ids = {f['id'] for f in session.get("current_features", [])}

    results = []
    for cluster in sc["clusters"]:
        label = cluster["label"]
        if query in label.lower():
            label_lower = label.lower()
            if label_lower == query:
                match_rank = 0
            elif label_lower.startswith(query):
                match_rank = 1
            elif f" {query}" in label_lower:
                match_rank = 2
            else:
                match_rank = 3
            results.append({
                'id': cluster["cluster_id"],
                'label': label,
                'description': cluster.get("description", ""),
                'member_ids': cluster["neuron_ids"],
                'movie_count': cluster.get("support", len(cluster["neuron_ids"])),
                'already_shown': cluster["cluster_id"] in current_feature_ids,
                'match_rank': match_rank,
            })
    results.sort(key=lambda x: (x['already_shown'], x['match_rank'], -x['movie_count'], len(x['label'])))

    participation_id = session.get("participation_id")
    if participation_id:
        log_interaction(
            participation_id,
            "feature-search",
            iteration=session.get("iteration", 1),
            phase=session.get("current_phase", 0),
            model_id=active_sae_id,
            query=query,
            result_count=len(results),
        )
    return jsonify(results[:20])


# ============================================================================
# Iteration Approval
# ============================================================================

@bp.route("/approve-preferences", methods=["POST"])
def approve_preferences():
    data = request.get_json(force=True) or {}
    session["iteration_preferences_approved"] = True

    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    active_model = _get_active_model_config(conf)
    participation_id = session.get("participation_id")
    is_final_confirmation = bool(session.get("iteration_locked_final", False))
    if participation_id:
        log_interaction(
            participation_id,
            "preferences-approved",
            iteration=session.get("iteration", 1),
            phase=session.get("current_phase", 0),
            model_id=active_model.get("sae", DEFAULT_TOPK_SAE_MODEL_ID),
            steering_mode=active_model.get("steering_mode", DEFAULT_STEERING_MODE),
            liked_movies=data.get("liked_movies", []),
            is_final_confirmation=is_final_confirmation,
        )

    return jsonify({"status": "ok", "approved": True})


# ============================================================================
# Movie Feedback (selection events)
# ============================================================================

@bp.route("/log-movie-feedback", methods=["POST"])
def log_movie_feedback():
    """Log a selection action on a recommended movie."""
    try:
        data = request.get_json(force=True)
        movie_id = data.get("movie_id")
        action = data.get("action", "neutral")
        iteration = data.get("iteration", session.get("iteration", 1))
        rank = data.get("rank")
        list_id = data.get("list_id")
        current_phase = session.get("current_phase", 0)

        participation_id = session.get("participation_id")
        if participation_id:
            log_interaction(
                participation_id,
                "movie-feedback",
                movie_id=movie_id,
                action=action,
                iteration=iteration,
                phase=current_phase,
                rank=rank,
                list_id=list_id,
            )
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[log_movie_feedback] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 200


# ============================================================================
# Generic client-side UI event logging
# ============================================================================

@bp.route("/log-ui-event", methods=["POST"])
def log_ui_event():
    """Log a fine-grained client-side UI event (slider touch, search, etc.)."""
    try:
        data = request.get_json(force=True) or {}
        participation_id = session.get("participation_id")
        if not participation_id:
            return jsonify({"status": "skip"}), 200

        event_type = data.pop("event_type", "ui-event")
        log_interaction(
            participation_id,
            event_type,
            iteration=session.get("iteration", 1),
            phase=session.get("current_phase", 0),
            **data,
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[log_ui_event] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 200


# ============================================================================
# Autosave – periodic client-side state snapshots for crash resilience
# ============================================================================

@bp.route("/autosave", methods=["POST"])
def autosave():
    """Persist a client-side state snapshot so progress can be recovered."""
    try:
        data = request.get_json(force=True) or {}
        participation_id = session.get("participation_id")
        if not participation_id:
            return jsonify({"status": "skip", "reason": "no participation"}), 200

        log_interaction(
            participation_id,
            "autosave",
            iteration=session.get("iteration", 1),
            phase=session.get("current_phase", 0),
            liked_movies=data.get("liked_movies", []),
            feature_adjustments=data.get("feature_adjustments", {}),
            activity_snapshot=data.get("activity_snapshot", {}),
            timestamp=data.get("timestamp"),
            trigger=data.get("trigger", "periodic"),
        )
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[autosave] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 200


# ============================================================================
# Study Completion
# ============================================================================

def _questionnaire_exists(conf):
    """Check whether a final questionnaire HTML file has been set up for this study."""
    if not conf or "questionnaire_file" not in conf:
        return False
    guid = session.get("user_study_guid", "")
    if not guid:
        # Fallback: look up guid from study id
        try:
            us = UserStudy.query.filter(UserStudy.id == session.get("user_study_id")).first()
            if us:
                guid = us.guid
        except Exception:
            pass
    q_path = get_cache_path(guid, conf["questionnaire_file"])
    return os.path.exists(q_path)


def _phase_questionnaire_exists(conf, phase_idx=None):
    """Check whether a phase questionnaire HTML file has been set up for this study."""
    phase_questionnaire_file = _get_phase_questionnaire_filename(conf, phase_idx)
    if not phase_questionnaire_file:
        return False
    guid = session.get("user_study_guid", "")
    if not guid:
        try:
            us = UserStudy.query.filter(UserStudy.id == session.get("user_study_id")).first()
            if us:
                guid = us.guid
        except Exception:
            pass
    q_path = get_cache_path(guid, phase_questionnaire_file)
    return os.path.exists(q_path)


@bp.route("/finish-user-study")
@multi_lang
def finish_user_study():
    """Complete the user study — show final questionnaire first if one exists."""
    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))

    has_final_q = conf and _questionnaire_exists(conf)
    print(f"[finish_user_study] conf has questionnaire_file: {bool(conf and 'questionnaire_file' in conf)}, "
          f"file exists: {has_final_q}, "
          f"guid: {session.get('user_study_guid', '(none)')}")

    if has_final_q:
        return redirect(url_for(
            "utils.final_questionnaire",
            continuation_url=url_for("utils.finish"),
            header_override="Final Comparison Questionnaire",
            hint_override="",
            hide_embedded_questionnaire_heading=1,
        ))
    study_ended(session["participation_id"])
    return redirect(url_for("utils.finish"))


# ============================================================================
# Results & Analysis
# ============================================================================

@bp.route("/results")
def results():
    """Display results dashboard"""
    guid = request.args.get("guid")
    return render_template(
        "sae_steering_results.html",
        guid=guid,
        fetch_results_url=url_for(f"{__plugin_name__}.fetch_results", guid=guid),
        journey_url_base=url_for(f"{__plugin_name__}.participant_journey", participation_id=0).rstrip("0"),
        export_raw_url=url_for(f"{__plugin_name__}.export_raw_data", guid=guid),
    )


def _safe_parse_json(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


_PROLIFIC_BASE_URL = "https://app.prolific.com/submissions/complete"


def _build_prolific_block(extra_data, study_config=None):
    """Return a normalised Prolific identity block.

    Pulls Prolific IDs from the participant's ``extra_data`` JSON and stitches
    in the study-level completion code so admins can copy a working completion
    URL straight from the participants table for payment reconciliation.
    """
    extra = _safe_parse_json(extra_data) if not isinstance(extra_data, dict) else extra_data
    pid = (extra or {}).get("PROLIFIC_PID")
    study_id = (extra or {}).get("PROLIFIC_STUDY_ID")
    session_id = (extra or {}).get("PROLIFIC_SESSION_ID")
    completion_code = (study_config or {}).get("prolific_code") if isinstance(study_config, dict) else None
    completion_url = None
    if completion_code:
        completion_url = f"{_PROLIFIC_BASE_URL}?cc={completion_code}"
    return {
        "pid": pid,
        "study_id": study_id,
        "session_id": session_id,
        "completion_code": completion_code,
        "completion_url": completion_url,
    }


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _build_distribution(values):
    counts = {}
    total = len(values)
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    percentages = {}
    for key, cnt in counts.items():
        percentages[key] = round((cnt / total) * 100, 2) if total else 0
    return {"counts": counts, "percentages": percentages, "n": total}


def _round_or_none(value, digits=3):
    if value is None:
        return None
    return round(value, digits)


@bp.route("/fetch-results/<guid>")
def fetch_results(guid):
    """Aggregated analytics for the SAE steering results dashboard.

    This is the *second* of the two export endpoints backing the admin UI
    (see :func:`export_raw_data` for the first).  Both read from the same
    SQLAlchemy tables (``UserStudy`` / ``Participation`` / ``Interaction``);
    they just differ in what they compute on the way out:

    - :func:`export_raw_data` dumps every ``Interaction`` row verbatim —
      the event log used for payment reconciliation and journey replay.
    - :func:`fetch_results` (this function) produces the precomputed
      per-arm / per-participant analytics consumed by
      ``sae_steering_results.html``: Likert means, attention pass rates,
      moderator effects, order/satisfaction splits, plus a
      ``participants_table`` with Prolific IDs, status, and duration.

    Downloading the returned JSON as-is (``Download analytics (JSON)``
    button) is the fastest way to reproduce dashboard figures offline.
    """
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return jsonify({"error": "Study not found"}), 404

    study_config = _normalize_study_config(load_user_study_config_by_guid(guid))
    config_models = list(study_config.get("models", []))

    participants = Participation.query.filter(
        (Participation.time_finished != None) &
        (Participation.user_study_id == user_study.id)
    ).all()

    final_score_maps = {
        "f1_preference": {
            "without_control_strongly": -2,
            "without_control_slightly": -1,
            "no_preference": 0,
            "with_control_slightly": 1,
            "with_control_strongly": 2,
        },
        "f2_better_recs": {
            "without_control_clearly": -2,
            "without_control_slightly": -1,
            "same": 0,
            "with_control_slightly": 1,
            "with_control_clearly": 2,
        },
        "f3_more_control": {
            "without_control_clearly": -2,
            "without_control_slightly": -1,
            "same": 0,
            "with_control_slightly": 1,
            "with_control_clearly": 2,
        },
        "f4_more_responsive": {
            "without_control_clearly": -2,
            "without_control_slightly": -1,
            "same": 0,
            "with_control_slightly": 1,
            "with_control_clearly": 2,
        },
    }

    phase_shared_items = [
        "p1a_accuracy",
        "p1b_novelty",
        "p1c_diversity",
        "p2a_control",
        "p2b_convergence",
        "p2c_liked_movies_sufficient",
        "p2d_correction_ease",
        "p5a_satisfaction",
        "p5b_reuse",
        "p5c_recommend",
    ]
    quality_items = ["p1a_accuracy", "p1b_novelty", "p1c_diversity"]
    satisfaction_items = ["p5a_satisfaction", "p5b_reuse", "p5c_recommend"]
    control_items = ["p2a_control", "p2b_convergence", "p2c_liked_movies_sufficient", "p2d_correction_ease"]
    steering_likert_items = [
        "p2e_responsiveness",
        "p3a_label_clarity",
        "p3b_predictability",
        "p4a_ease",
        "p4b_cognitive_load",
        "p4c_displayed_features_sufficient",
        "p4d_search_needed",
        "p4g_granularity",
        "p4h_overlap",
    ]
    reverse_scored_items = {"p4b_cognitive_load", "p4h_overlap"}
    steering_categorical_items = ["p4e_displayed_feature_count", "p4f_boost_suppress_balance"]

    results = {
        "study_guid": guid,
        "sample": {
            "participants_completed": len(participants),
            "participants_with_phase_questionnaires": 0,
            "participants_with_final_questionnaire": 0,
            "participants_with_both": 0,
            "attention_checks": {
                "phase_passed": 0,
                "phase_total": 0,
                "final_passed": 0,
                "final_total": 0,
                "overall_passed": 0,
                "overall_total": len(participants),
            },
        },
        "approaches": {"labels": {}, "overview": {}, "comparison": {}},
        "questionnaire": {
            "final": {"responses": 0, "items": {}, "text_feedback": {}},
            "phase_shared": {"participants_with_pairs": 0, "items": {}, "composites": {}},
            "steering_specific": {"likert_items": {}, "categorical_items": {}},
            "links": {},
        },
        "moderators": {},
        "insights": [],
        "participants": [],
    }

    participant_analytics = []
    final_distributions = {}
    final_score_values = {item: [] for item in final_score_maps}
    final_text_feedback = {"f24_liked_most": [], "f25_improvement": [], "f26_other": []}
    # Two paired-difference buckets per shared item:
    #   * phase_shared_diffs  — chronological (phase2 − phase1); detects
    #     order/fatigue effects, arm-agnostic.
    #   * phase_shared_arm_diffs — condition-based (steered − baseline);
    #     the primary hypothesis test "did the steering arm score higher?".
    #     Only populated when we can identify which phase carried the
    #     steering UI (via `steered_markers` heuristic below).
    phase_shared_diffs = {item: [] for item in phase_shared_items}
    phase_shared_arm_diffs = {item: [] for item in phase_shared_items}
    quality_diffs = []
    satisfaction_diffs = []
    control_diffs = []
    steering_likert_values = {item: [] for item in steering_likert_items}
    steering_categorical_values = {item: [] for item in steering_categorical_items}
    moderator_pref_by_ml = {"low": [], "mid": [], "high": []}
    moderator_pref_by_movie = {"low": [], "mid": [], "high": []}
    moderator_pref_by_freq = {}
    link_pref_quality = []
    link_control = []
    link_responsiveness = []
    preference_model_votes = {}
    selection_dynamics = {}

    for idx, cfg_model in enumerate(config_models):
        key = str(idx)
        results["approaches"]["labels"][key] = cfg_model.get("name") or _approach_label(idx)

    for p in participants:
        interactions = Interaction.query.filter(
            Interaction.participation == p.id
        ).order_by(Interaction.time.asc()).all()

        prolific_block = _build_prolific_block(p.extra_data, study_config)
        duration_sec = None
        if p.time_joined and p.time_finished:
            duration_sec = int((p.time_finished - p.time_joined).total_seconds())

        # Seed approach_order / effective_order from extra_data as a defensive
        # fallback.  `_get_effective_models` now mirrors the chosen order onto
        # the Participation row so results never show "-" just because the
        # interaction log row was missed (e.g. session was pre-seeded).
        extra_seed = {}
        try:
            extra_seed = json.loads(p.extra_data) if p.extra_data else {}
            if not isinstance(extra_seed, dict):
                extra_seed = {}
        except Exception:
            extra_seed = {}

        participant_data = {
            "participant_id": p.id,
            "uuid": p.uuid,
            "email": p.participant_email,
            "prolific": prolific_block,
            "time_joined": p.time_joined.isoformat() if p.time_joined else None,
            "time_finished": p.time_finished.isoformat() if p.time_finished else None,
            "duration_sec": duration_sec,
            "language": p.language,
            "approach_order": extra_seed.get("approach_order"),
            "effective_order": extra_seed.get("effective_order"),
            "elicitation_selected_movies": None,
            "search_events": [],
            "phase_models": {},
            "phase_behavior": {},
            "phase_movie_feedback": {},
            "phase_questionnaires": {},
            "final_questionnaire": {},
            "attention_checks": {
                "phase_pass": None,
                "phase_passed_count": 0,
                "phase_total_count": 0,
                "final_pass": None,
                "overall_pass": None,
            },
        }

        phase_behavior = {}
        phase_movie_feedback = {}
        phase_questionnaires = {}
        final_questionnaire = {}
        phase_models = {}

        for interaction in interactions:
            data = _safe_parse_json(interaction.data)

            if interaction.interaction_type == "approach-order-assigned":
                if data.get("approach_order") is not None:
                    participant_data["approach_order"] = data.get("approach_order")
                if data.get("effective_order") is not None:
                    participant_data["effective_order"] = data.get("effective_order")

            elif interaction.interaction_type == "elicitation-completed":
                participant_data["elicitation_selected_movies"] = data.get("selected_movies")

            elif interaction.interaction_type in ("elicitation-search", "feature-search"):
                participant_data["search_events"].append({
                    "type": interaction.interaction_type,
                    "time": interaction.time.isoformat() if interaction.time else None,
                    "query": data.get("query"),
                    "result_count": data.get("result_count"),
                    "phase": data.get("phase"),
                    "iteration": data.get("iteration"),
                })

            elif interaction.interaction_type == "feature-adjustment":
                phase_idx = str(int(data.get("phase", 0)))
                phase_entry = phase_behavior.setdefault(
                    phase_idx,
                    {"iterations": [], "abs_values": [], "nonzero_adjustments": 0, "events": 0},
                )
                iteration = int(data.get("iteration", 0) or 0)
                if iteration > 0:
                    phase_entry["iterations"].append(iteration)
                adjustments = data.get("adjustments", {}) or {}
                numeric_values = []
                for _, raw_val in adjustments.items():
                    val = _to_float(raw_val)
                    if val is None:
                        continue
                    numeric_values.append(val)
                    phase_entry["abs_values"].append(abs(val))
                    if abs(val) > 1e-9:
                        phase_entry["nonzero_adjustments"] += 1
                if numeric_values:
                    phase_entry["events"] += 1

            elif interaction.interaction_type == "phase-questionnaire":
                phase_idx = str(int(data.get("phase", 0)))
                answers = {k: v for k, v in data.items() if k not in {"phase"}}
                phase_questionnaires[phase_idx] = answers

            elif interaction.interaction_type == "movie-feedback":
                phase_idx = str(int(data.get("phase", 0)))
                phase_entry = phase_movie_feedback.setdefault(
                    phase_idx,
                    {"events": [], "iteration_like_counts": {}, "position_counts": {}, "like_events": 0, "neutral_events": 0},
                )
                action = str(data.get("action", "neutral")).strip().lower()
                iteration = int(data.get("iteration", 0) or 0)
                rank = _to_float(data.get("rank"))
                rank_int = int(rank) if rank is not None else None
                list_id = data.get("list_id")
                phase_entry["events"].append(
                    {
                        "movie_id": data.get("movie_id"),
                        "action": action,
                        "iteration": iteration,
                        "rank": rank_int,
                        "list_id": list_id,
                    }
                )
                if action == "like":
                    phase_entry["like_events"] += 1
                    iter_bucket = phase_entry["iteration_like_counts"]
                    iter_key = str(iteration)
                    iter_bucket[iter_key] = iter_bucket.get(iter_key, 0) + 1
                    if rank_int is not None and rank_int > 0:
                        pos_bucket = phase_entry["position_counts"]
                        rank_key = str(rank_int)
                        pos_bucket[rank_key] = pos_bucket.get(rank_key, 0) + 1
                elif action == "neutral":
                    phase_entry["neutral_events"] += 1

            elif interaction.interaction_type == "final-questionnaire":
                final_questionnaire = data

            elif interaction.interaction_type == "phase-complete":
                phase_idx = str(int(data.get("phase", 0)))
                if data.get("model"):
                    phase_models[phase_idx] = data.get("model")

        has_phase_q = bool(phase_questionnaires)
        has_final_q = bool(final_questionnaire)
        if has_phase_q:
            results["sample"]["participants_with_phase_questionnaires"] += 1
        if has_final_q:
            results["sample"]["participants_with_final_questionnaire"] += 1
        if has_phase_q and has_final_q:
            results["sample"]["participants_with_both"] += 1

        phase_attention_passes = []
        for phase_idx, answers in phase_questionnaires.items():
            raw_attention = answers.get("p_attention_check")
            if raw_attention is None:
                continue
            is_steered_phase = any(
                key in answers for key in ("p2e_responsiveness", "p3a_label_clarity", "p4a_ease")
            )
            attention_value = str(raw_attention).strip()
            if is_steered_phase:
                passed = attention_value in {"1", "2", "3"}
            else:
                passed = attention_value == "7"
            phase_attention_passes.append(passed)
            results["sample"]["attention_checks"]["phase_total"] += 1
            if passed:
                results["sample"]["attention_checks"]["phase_passed"] += 1

        final_attention_value = final_questionnaire.get("f_attention_check")
        final_attention_pass = None
        if final_attention_value is not None:
            final_attention_pass = str(final_attention_value).strip() == "same"
            results["sample"]["attention_checks"]["final_total"] += 1
            if final_attention_pass:
                results["sample"]["attention_checks"]["final_passed"] += 1

        phase_pass_all = (all(phase_attention_passes) if phase_attention_passes else None)
        if phase_pass_all is True and (final_attention_pass in (True, None)):
            participant_data["attention_checks"]["overall_pass"] = True
        elif phase_pass_all is False or final_attention_pass is False:
            participant_data["attention_checks"]["overall_pass"] = False

        if participant_data["attention_checks"]["overall_pass"] is True:
            results["sample"]["attention_checks"]["overall_passed"] += 1
        participant_data["attention_checks"]["phase_pass"] = phase_pass_all
        participant_data["attention_checks"]["phase_passed_count"] = sum(
            1 for v in phase_attention_passes if v
        )
        participant_data["attention_checks"]["phase_total_count"] = len(phase_attention_passes)
        participant_data["attention_checks"]["final_pass"] = final_attention_pass

        normalized_phase_behavior = {}
        for phase_idx, raw in phase_behavior.items():
            max_iteration = max(raw["iterations"]) if raw["iterations"] else 0
            normalized_phase_behavior[phase_idx] = {
                "events": raw["events"],
                "max_iteration": max_iteration,
                "mean_abs_adjustment": _round_or_none(_mean(raw["abs_values"]), 4),
                "nonzero_adjustments": raw["nonzero_adjustments"],
            }

        participant_data["phase_models"] = phase_models
        participant_data["phase_behavior"] = normalized_phase_behavior
        participant_data["phase_movie_feedback"] = phase_movie_feedback
        participant_data["phase_questionnaires"] = phase_questionnaires
        participant_data["final_questionnaire"] = final_questionnaire
        results["participants"].append(participant_data)

        participant_analytics.append(
            {
                "participant_id": p.id,
                "phase_questionnaires": phase_questionnaires,
                "final_questionnaire": final_questionnaire,
                "phase_models": phase_models,
                "phase_behavior": normalized_phase_behavior,
                "phase_movie_feedback": phase_movie_feedback,
            }
        )

        for item_key in list(final_score_maps.keys()) + [
            "f19_movie_familiarity",
            "f20_rs_frequency",
            "f21_ml_familiarity",
        ]:
            value = final_questionnaire.get(item_key)
            if value in (None, ""):
                continue
            final_distributions.setdefault(item_key, []).append(value)

        for item_key, mapping in final_score_maps.items():
            choice = final_questionnaire.get(item_key)
            if choice in mapping:
                score = mapping[choice]
                final_score_values[item_key].append(score)

        for text_key in final_text_feedback:
            text_value = (final_questionnaire.get(text_key) or "").strip()
            if text_value:
                final_text_feedback[text_key].append(text_value)

        phase_keys = sorted(phase_questionnaires.keys(), key=lambda x: int(x))
        if len(phase_keys) >= 2:
            first_phase, second_phase = phase_keys[0], phase_keys[1]
            first_answers = phase_questionnaires.get(first_phase, {})
            second_answers = phase_questionnaires.get(second_phase, {})
            steered_markers = ("p2e_responsiveness", "p3a_label_clarity", "p4a_ease")
            first_is_steered = any(marker in first_answers for marker in steered_markers)
            second_is_steered = any(marker in second_answers for marker in steered_markers)

            with_control_phase = None
            without_control_phase = None
            with_control_answers = None
            without_control_answers = None
            if first_is_steered and not second_is_steered:
                with_control_phase, with_control_answers = first_phase, first_answers
                without_control_phase, without_control_answers = second_phase, second_answers
            elif second_is_steered and not first_is_steered:
                with_control_phase, with_control_answers = second_phase, second_answers
                without_control_phase, without_control_answers = first_phase, first_answers

            for item_key in phase_shared_items:
                left = _to_float(first_answers.get(item_key))
                right = _to_float(second_answers.get(item_key))
                if left is not None and right is not None:
                    phase_shared_diffs[item_key].append(right - left)

            # Condition-based (arm) delta: for each Likert item, compare the
            # score in the steered phase to the score in the baseline phase
            # within the same participant.  Positive ⇒ steering lifted that
            # dimension relative to the vanilla recommender.  Skipped when
            # we can't confidently classify the arms.
            if with_control_answers is not None and without_control_answers is not None:
                for item_key in phase_shared_items:
                    steered_val = _to_float(with_control_answers.get(item_key))
                    baseline_val = _to_float(without_control_answers.get(item_key))
                    if steered_val is not None and baseline_val is not None:
                        phase_shared_arm_diffs[item_key].append(steered_val - baseline_val)

            quality_with = [v for v in [_to_float((with_control_answers or {}).get(x)) for x in quality_items] if v is not None]
            quality_without = [v for v in [_to_float((without_control_answers or {}).get(x)) for x in quality_items] if v is not None]
            if quality_with and quality_without:
                quality_diff = _mean(quality_with) - _mean(quality_without)
                quality_diffs.append(quality_diff)

            sat_with = [v for v in [_to_float((with_control_answers or {}).get(x)) for x in satisfaction_items] if v is not None]
            sat_without = [v for v in [_to_float((without_control_answers or {}).get(x)) for x in satisfaction_items] if v is not None]
            if sat_with and sat_without:
                satisfaction_diffs.append(_mean(sat_with) - _mean(sat_without))

            control_with = [v for v in [_to_float((with_control_answers or {}).get(x)) for x in control_items] if v is not None]
            control_without = [v for v in [_to_float((without_control_answers or {}).get(x)) for x in control_items] if v is not None]
            if control_with and control_without:
                control_diffs.append(_mean(control_with) - _mean(control_without))

            preference_score = None
            pref_choice = final_questionnaire.get("f1_preference")
            if pref_choice in final_score_maps["f1_preference"]:
                preference_score = final_score_maps["f1_preference"][pref_choice]

            if preference_score is not None and quality_with and quality_without:
                link_pref_quality.append(
                    {"preference_score": preference_score, "quality_diff": _mean(quality_with) - _mean(quality_without)}
                )
            if preference_score is not None and control_with and control_without:
                link_control.append(
                    {"preference_score": preference_score, "control_diff": _mean(control_with) - _mean(control_without)}
                )

            responsiveness_without = _to_float((without_control_answers or {}).get("p2e_responsiveness"))
            responsiveness_with = _to_float((with_control_answers or {}).get("p2e_responsiveness"))
            responsiveness_diff = None
            if responsiveness_without is not None and responsiveness_with is not None:
                responsiveness_diff = responsiveness_with - responsiveness_without
            elif responsiveness_with is not None:
                responsiveness_diff = responsiveness_with
            elif responsiveness_without is not None:
                responsiveness_diff = -responsiveness_without
            if preference_score is not None and responsiveness_diff is not None:
                link_responsiveness.append(
                    {"preference_score": preference_score, "responsiveness_diff": responsiveness_diff}
                )

            if pref_choice:
                if pref_choice.startswith("with_control_"):
                    target_phase = with_control_phase
                elif pref_choice.startswith("without_control_"):
                    target_phase = without_control_phase
                else:
                    target_phase = None
                if pref_choice == "no_preference":
                    target_phase = None
                if target_phase is not None:
                    model_name = (
                        phase_models.get(target_phase)
                        or results["approaches"]["labels"].get(target_phase)
                        or f"Approach {target_phase}"
                    )
                    preference_model_votes[model_name] = preference_model_votes.get(model_name, 0) + 1

            ml_score = _to_float(final_questionnaire.get("f21_ml_familiarity"))
            movie_score = _to_float(final_questionnaire.get("f19_movie_familiarity"))
            if preference_score is not None and ml_score is not None:
                if ml_score <= 2:
                    moderator_pref_by_ml["low"].append(preference_score)
                elif ml_score >= 4:
                    moderator_pref_by_ml["high"].append(preference_score)
                else:
                    moderator_pref_by_ml["mid"].append(preference_score)
            if preference_score is not None and movie_score is not None:
                if movie_score <= 2:
                    moderator_pref_by_movie["low"].append(preference_score)
                elif movie_score >= 4:
                    moderator_pref_by_movie["high"].append(preference_score)
                else:
                    moderator_pref_by_movie["mid"].append(preference_score)

            usage_freq = final_questionnaire.get("f20_rs_frequency")
            if preference_score is not None and usage_freq:
                moderator_pref_by_freq.setdefault(usage_freq, []).append(preference_score)

        for phase_idx, answers in phase_questionnaires.items():
            for item_key in phase_shared_items:
                numeric_val = _to_float(answers.get(item_key))
                if numeric_val is not None:
                    bucket = results["questionnaire"]["phase_shared"]["items"].setdefault(
                        item_key, {"by_phase": {}, "delta_b_minus_a": {"n": 0, "mean": None}}
                    )
                    phase_bucket = bucket["by_phase"].setdefault(phase_idx, [])
                    phase_bucket.append(numeric_val)

            for item_key in steering_likert_items:
                numeric_val = _to_float(answers.get(item_key))
                if numeric_val is None:
                    continue
                if item_key in reverse_scored_items:
                    numeric_val = 8 - numeric_val
                steering_likert_values[item_key].append(numeric_val)

            for item_key in steering_categorical_items:
                cat_val = answers.get(item_key)
                if cat_val:
                    steering_categorical_values[item_key].append(cat_val)

    for phase_idx, label in list(results["approaches"]["labels"].items()):
        phase_rows = [x["phase_behavior"].get(phase_idx, {}) for x in participant_analytics if phase_idx in x["phase_behavior"]]
        if not phase_rows:
            continue
        iter_values = [row.get("max_iteration", 0) for row in phase_rows if row.get("max_iteration", 0) > 0]
        abs_values = [row.get("mean_abs_adjustment") for row in phase_rows if row.get("mean_abs_adjustment") is not None]
        nonzero_values = [row.get("nonzero_adjustments", 0) for row in phase_rows]
        event_values = [row.get("events", 0) for row in phase_rows]
        results["approaches"]["overview"][phase_idx] = {
            "label": label,
            "participants": len(phase_rows),
            "mean_iterations": _round_or_none(_mean(iter_values)),
            "mean_abs_adjustment": _round_or_none(_mean(abs_values), 4),
            "mean_nonzero_adjustments": _round_or_none(_mean(nonzero_values), 2),
            "mean_adjustment_events": _round_or_none(_mean(event_values), 2),
        }

    if "0" in results["approaches"]["overview"] and "1" in results["approaches"]["overview"]:
        a_stats = results["approaches"]["overview"]["0"]
        b_stats = results["approaches"]["overview"]["1"]
        results["approaches"]["comparison"] = {
            "delta_iterations_b_minus_a": _round_or_none(
                (b_stats.get("mean_iterations") or 0) - (a_stats.get("mean_iterations") or 0), 3
            ),
            "delta_abs_adjustment_b_minus_a": _round_or_none(
                (b_stats.get("mean_abs_adjustment") or 0) - (a_stats.get("mean_abs_adjustment") or 0), 4
            ),
            "delta_nonzero_adjustments_b_minus_a": _round_or_none(
                (b_stats.get("mean_nonzero_adjustments") or 0) - (a_stats.get("mean_nonzero_adjustments") or 0), 3
            ),
        }

    for participant_row in participant_analytics:
        feedback_by_phase = participant_row.get("phase_movie_feedback", {})
        for phase_idx, phase_feedback in feedback_by_phase.items():
            aggregate = selection_dynamics.setdefault(
                phase_idx,
                {
                    "participants_with_feedback": 0,
                    "total_like_events": 0,
                    "total_neutral_events": 0,
                    "iteration_like_counts": {},
                    "position_counts": {},
                },
            )
            aggregate["participants_with_feedback"] += 1
            aggregate["total_like_events"] += int(phase_feedback.get("like_events", 0))
            aggregate["total_neutral_events"] += int(phase_feedback.get("neutral_events", 0))
            for iter_key, iter_count in (phase_feedback.get("iteration_like_counts", {}) or {}).items():
                aggregate["iteration_like_counts"][iter_key] = (
                    aggregate["iteration_like_counts"].get(iter_key, 0) + int(iter_count)
                )
            for rank_key, rank_count in (phase_feedback.get("position_counts", {}) or {}).items():
                aggregate["position_counts"][rank_key] = (
                    aggregate["position_counts"].get(rank_key, 0) + int(rank_count)
                )

    normalized_selection_dynamics = {}
    for phase_idx, phase_data in selection_dynamics.items():
        participant_count = phase_data.get("participants_with_feedback") or 1
        iteration_counts = phase_data.get("iteration_like_counts", {})
        position_counts = phase_data.get("position_counts", {})
        normalized_selection_dynamics[phase_idx] = {
            "label": results["approaches"]["labels"].get(phase_idx, f"Approach {phase_idx}"),
            "participants_with_feedback": phase_data.get("participants_with_feedback", 0),
            "total_like_events": phase_data.get("total_like_events", 0),
            "total_neutral_events": phase_data.get("total_neutral_events", 0),
            "mean_like_events_per_participant": _round_or_none(
                phase_data.get("total_like_events", 0) / participant_count, 3
            ),
            "iteration_like_counts": iteration_counts,
            "iteration_like_means": {
                iter_key: _round_or_none(count / participant_count, 3)
                for iter_key, count in sorted(iteration_counts.items(), key=lambda item: int(item[0]))
            },
            "position_counts": {
                rank_key: count
                for rank_key, count in sorted(position_counts.items(), key=lambda item: int(item[0]))
            },
        }
    results["approaches"]["selection_dynamics"] = normalized_selection_dynamics

    final_section = results["questionnaire"]["final"]
    final_section["responses"] = len(final_distributions.get("f1_preference", []))
    for item_key, values in final_distributions.items():
        entry = {"distribution": _build_distribution(values)}
        if item_key in final_score_values and final_score_values[item_key]:
            entry["mean_score"] = _round_or_none(_mean(final_score_values[item_key]), 3)
            entry["score_scale"] = "without explicit control=-2 ... with explicit control=+2"
        if item_key in ("f19_movie_familiarity", "f21_ml_familiarity"):
            numeric_values = [_to_float(v) for v in values]
            numeric_values = [x for x in numeric_values if x is not None]
            if numeric_values:
                entry["mean"] = _round_or_none(_mean(numeric_values), 3)
                entry["scale"] = "1-5"
        final_section["items"][item_key] = entry

    final_section["text_feedback"] = {
        key: {"count": len(vals), "samples": vals[:5]} for key, vals in final_text_feedback.items()
    }

    phase_section = results["questionnaire"]["phase_shared"]
    paired_count = 0
    for item_key, diffs in phase_shared_diffs.items():
        if diffs:
            paired_count = max(paired_count, len(diffs))
        item_entry = phase_section["items"].get(item_key)
        if not item_entry:
            item_entry = {
                "by_phase": {},
                "delta_b_minus_a": {"n": 0, "mean": None},
                "delta_steered_minus_baseline": {"n": 0, "mean": None},
            }
            phase_section["items"][item_key] = item_entry
        for phase_idx, scores in list(item_entry["by_phase"].items()):
            item_entry["by_phase"][phase_idx] = {
                "n": len(scores),
                "mean": _round_or_none(_mean(scores), 3),
            }
        item_entry["delta_b_minus_a"] = {"n": len(diffs), "mean": _round_or_none(_mean(diffs), 3)}
        arm_diffs = phase_shared_arm_diffs.get(item_key, [])
        item_entry["delta_steered_minus_baseline"] = {
            "n": len(arm_diffs),
            "mean": _round_or_none(_mean(arm_diffs), 3),
        }
    phase_section["participants_with_pairs"] = paired_count
    phase_section["composites"] = {
        "quality_diff_b_minus_a": {"n": len(quality_diffs), "mean": _round_or_none(_mean(quality_diffs), 3)},
        "satisfaction_diff_b_minus_a": {
            "n": len(satisfaction_diffs),
            "mean": _round_or_none(_mean(satisfaction_diffs), 3),
        },
        "control_diff_b_minus_a": {"n": len(control_diffs), "mean": _round_or_none(_mean(control_diffs), 3)},
    }

    steering_section = results["questionnaire"]["steering_specific"]
    for item_key, values in steering_likert_values.items():
        steering_section["likert_items"][item_key] = {
            "n": len(values),
            "mean": _round_or_none(_mean(values), 3),
            "scale": "1-7",
        }
    for item_key, values in steering_categorical_values.items():
        steering_section["categorical_items"][item_key] = _build_distribution(values)

    def _build_link_summary(rows, value_key):
        if not rows:
            return {"n": 0, "mean_by_preference": {}}
        buckets = {"without_control": [], "neutral": [], "with_control": []}
        for row in rows:
            pref = row.get("preference_score", 0)
            metric_value = row.get(value_key)
            if metric_value is None:
                continue
            if pref < 0:
                buckets["without_control"].append(metric_value)
            elif pref > 0:
                buckets["with_control"].append(metric_value)
            else:
                buckets["neutral"].append(metric_value)
        return {
            "n": len(rows),
            "mean_by_preference": {
                key: _round_or_none(_mean(vals), 3) for key, vals in buckets.items() if vals
            },
        }

    results["questionnaire"]["links"] = {
        "preference_vs_quality": _build_link_summary(link_pref_quality, "quality_diff"),
        "preference_vs_control": _build_link_summary(link_control, "control_diff"),
        "preference_vs_responsiveness": _build_link_summary(link_responsiveness, "responsiveness_diff"),
        "preferred_model_votes": preference_model_votes,
    }

    def _build_moderator_stats(groups):
        return {
            key: {"n": len(vals), "mean_preference_score": _round_or_none(_mean(vals), 3)}
            for key, vals in groups.items()
        }

    results["moderators"] = {
        "ml_familiarity": _build_moderator_stats(moderator_pref_by_ml),
        "movie_familiarity": _build_moderator_stats(moderator_pref_by_movie),
        "rs_usage_frequency": _build_moderator_stats(moderator_pref_by_freq),
    }

    phase_attention = results["sample"]["attention_checks"]["phase_total"]
    final_attention = results["sample"]["attention_checks"]["final_total"]
    if phase_attention:
        results["sample"]["attention_checks"]["phase_pass_rate"] = _round_or_none(
            results["sample"]["attention_checks"]["phase_passed"] / phase_attention, 3
        )
    else:
        results["sample"]["attention_checks"]["phase_pass_rate"] = None
    if final_attention:
        results["sample"]["attention_checks"]["final_pass_rate"] = _round_or_none(
            results["sample"]["attention_checks"]["final_passed"] / final_attention, 3
        )
    else:
        results["sample"]["attention_checks"]["final_pass_rate"] = None
    overall_total = results["sample"]["attention_checks"]["overall_total"] or 0
    if overall_total:
        results["sample"]["attention_checks"]["overall_pass_rate"] = _round_or_none(
            results["sample"]["attention_checks"]["overall_passed"] / overall_total, 3
        )
    else:
        results["sample"]["attention_checks"]["overall_pass_rate"] = None

    pref_mean = _mean(final_score_values["f1_preference"])
    quality_link = results["questionnaire"]["links"]["preference_vs_quality"]["mean_by_preference"]
    control_link = results["questionnaire"]["links"]["preference_vs_control"]["mean_by_preference"]
    resp_link = results["questionnaire"]["links"]["preference_vs_responsiveness"]["mean_by_preference"]
    ml_mod = results["moderators"]["ml_familiarity"]

    if pref_mean is not None:
        direction = (
            "Approach with explicit control"
            if pref_mean > 0
            else "Approach without explicit control"
            if pref_mean < 0
            else "No clear winner"
        )
        results["insights"].append(
            {
                "id": "preference",
                "title": "Overall preference",
                "finding": f"{direction} is preferred on average.",
                "strength": _round_or_none(abs(pref_mean), 3),
                "metric": "f1_preference mean (without control=-2, with control=+2)",
            }
        )
    if quality_link:
        results["insights"].append(
            {
                "id": "quality",
                "title": "Preference vs recommendation quality",
                "finding": "Participants preferring explicit control also report stronger quality gains when with-minus-without quality diff is positive.",
                "strength": _round_or_none(
                    (quality_link.get("with_control", 0) - quality_link.get("without_control", 0)), 3
                ),
                "metric": "Mean quality_diff (with - without) by preference group",
            }
        )
    if control_link or resp_link:
        strength_control = (
            (control_link.get("with_control", 0) - control_link.get("without_control", 0))
            if control_link else 0
        )
        strength_resp = (
            (resp_link.get("with_control", 0) - resp_link.get("without_control", 0))
            if resp_link else 0
        )
        results["insights"].append(
            {
                "id": "control_responsiveness",
                "title": "Control and responsiveness",
                "finding": "Preference trends align with perceived control and responsiveness differences between approaches.",
                "strength": _round_or_none(strength_control + strength_resp, 3),
                "metric": "Control/Responsiveness deltas by preference group",
            }
        )
    if ml_mod:
        high = ml_mod.get("high", {}).get("mean_preference_score")
        low = ml_mod.get("low", {}).get("mean_preference_score")
        if high is not None and low is not None:
            results["insights"].append(
                {
                    "id": "moderator_ml",
                    "title": "ML familiarity as moderator",
                    "finding": "Preference differs between participants with low vs high ML familiarity.",
                    "strength": _round_or_none(high - low, 3),
                    "metric": "Difference in mean preference score (high-low ML familiarity)",
                }
            )

    # ----------------------------------------------------------------
    # Participants table — all participations (including in-progress /
    # abandoned) with Prolific identity + completion status so the
    # admin can reconcile payments without leaving the dashboard.
    # ----------------------------------------------------------------
    completed_by_id = {entry["participant_id"]: entry for entry in results["participants"]}
    all_participations = Participation.query.filter(
        Participation.user_study_id == user_study.id
    ).order_by(Participation.time_joined.asc()).all()

    table_rows = []
    for p in all_participations:
        prolific_block = _build_prolific_block(p.extra_data, study_config)
        completed_entry = completed_by_id.get(p.id)
        if p.time_finished is not None:
            status = "completed"
        elif p.time_joined is not None:
            status = "in_progress"
        else:
            status = "joined"
        duration_sec = None
        if p.time_joined and p.time_finished:
            duration_sec = int((p.time_finished - p.time_joined).total_seconds())
        attention = (completed_entry or {}).get("attention_checks") or {}
        approach_order = (completed_entry or {}).get("approach_order")
        effective_order = (completed_entry or {}).get("effective_order")
        # Defensive fallback: in-progress participants have no completed_entry
        # but we still stamp approach_order into extra_data up-front.
        if approach_order is None or effective_order is None:
            try:
                extra = json.loads(p.extra_data) if p.extra_data else {}
                if isinstance(extra, dict):
                    if approach_order is None:
                        approach_order = extra.get("approach_order")
                    if effective_order is None:
                        effective_order = extra.get("effective_order")
            except Exception:
                pass
        search_count = len((completed_entry or {}).get("search_events", []) or [])
        table_rows.append(
            {
                "participation_id": p.id,
                "uuid": p.uuid,
                "prolific": prolific_block,
                "language": p.language,
                "status": status,
                "time_joined": p.time_joined.isoformat() if p.time_joined else None,
                "time_finished": p.time_finished.isoformat() if p.time_finished else None,
                "duration_sec": duration_sec,
                "approach_order": approach_order,
                "effective_order": effective_order,
                "search_events_count": search_count,
                "attention_checks": {
                    "phase_pass": attention.get("phase_pass"),
                    "phase_passed_count": attention.get("phase_passed_count", 0),
                    "phase_total_count": attention.get("phase_total_count", 0),
                    "final_pass": attention.get("final_pass"),
                    "overall_pass": attention.get("overall_pass"),
                },
            }
        )

    results["participants_table"] = table_rows
    results["sample"]["participants_total"] = len(all_participations)
    results["sample"]["participants_in_progress"] = sum(
        1 for r in table_rows if r["status"] == "in_progress"
    )

    return jsonify(results)


# ============================================================================
# Raw Data Export
# ============================================================================

@bp.route("/export-raw/<guid>")
def export_raw_data(guid):
    """Export complete raw study data as JSON for offline analysis.

    Each participation includes every ``Interaction`` row in chronological order
    so that the full user journey can be reconstructed.

    By default the export is scrubbed of high-frequency hover noise (hover
    events and the bulky ``context.extra.items[]`` arrays inside
    ``changed-viewport`` rows).  Pass ``?include_noise=1`` to opt back into the
    raw stream — useful for debugging client-side instrumentation, but the file
    becomes orders of magnitude larger.
    """
    from .journey import is_noise, scrub_interaction

    include_noise = (request.args.get("include_noise") or "").strip().lower() in {"1", "true", "yes"}

    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return jsonify({"error": "Study not found"}), 404

    study_config = _safe_parse_json(user_study.settings)

    all_participations = Participation.query.filter(
        Participation.user_study_id == user_study.id
    ).order_by(Participation.time_joined.asc()).all()

    participants_data = []
    for p in all_participations:
        interactions = Interaction.query.filter(
            Interaction.participation == p.id
        ).order_by(Interaction.time.asc()).all()

        interaction_rows = []
        for ix in interactions:
            row = {
                "id": ix.id,
                "type": ix.interaction_type,
                "time": ix.time.isoformat() if ix.time else None,
                "data": _safe_parse_json(ix.data),
            }
            if not include_noise and is_noise(row):
                continue
            if not include_noise:
                scrub_interaction(row)
            interaction_rows.append(row)

        prolific_block = _build_prolific_block(p.extra_data, study_config)
        participants_data.append({
            "participation_id": p.id,
            "uuid": p.uuid,
            "email": p.participant_email,
            "language": p.language,
            "time_joined": p.time_joined.isoformat() if p.time_joined else None,
            "time_finished": p.time_finished.isoformat() if p.time_finished else None,
            "prolific": prolific_block,
            "extra_data": _safe_parse_json(p.extra_data),
            "interactions": interaction_rows,
        })

    export = {
        "study_guid": guid,
        "study_id": user_study.id,
        "study_config": study_config,
        "exported_at": datetime.datetime.utcnow().isoformat(),
        "include_noise": include_noise,
        "participants_total": len(participants_data),
        "participants_completed": sum(1 for p in participants_data if p["time_finished"]),
        "participants": participants_data,
    }

    response = jsonify(export)
    response.headers["Content-Disposition"] = f"attachment; filename=study_{guid}_export.json"
    return response


# ============================================================================
# Journey Reconstruction (per participant)
# ============================================================================

@bp.route("/journey/<int:participation_id>")
def participant_journey(participation_id):
    """Return a structured timeline + per-phase summary for one participation.

    Powers the Journey tab on the admin results page.  Uses the shared
    ``journey`` helpers so the on-screen reconstruction matches the offline
    CLI script byte-for-byte.
    """
    from .journey import build_journey

    participation = Participation.query.filter(Participation.id == participation_id).first()
    if not participation:
        return jsonify({"error": "Participation not found"}), 404

    user_study = UserStudy.query.filter(UserStudy.id == participation.user_study_id).first()
    study_config = _safe_parse_json(user_study.settings) if user_study else {}

    interactions = Interaction.query.filter(
        Interaction.participation == participation.id
    ).order_by(Interaction.time.asc()).all()

    rows = [
        {
            "id": ix.id,
            "type": ix.interaction_type,
            "time": ix.time.isoformat() if ix.time else None,
            "data": _safe_parse_json(ix.data),
        }
        for ix in interactions
    ]

    include_noise = (request.args.get("include_noise") or "").strip().lower() in {"1", "true", "yes"}
    journey = build_journey(rows, include_noise=include_noise)

    # Replay just the attention-check logic so the journey card can show the
    # same pass/fail badges as the participants table.
    phase_passes = []
    final_pass = None
    for ix in rows:
        if ix["type"] == "phase-questionnaire":
            d = ix.get("data") or {}
            raw_attention = d.get("p_attention_check")
            if raw_attention is None:
                continue
            is_steered = any(
                k in d for k in ("p2e_responsiveness", "p3a_label_clarity", "p4a_ease")
            )
            attention_value = str(raw_attention).strip()
            phase_passes.append(
                attention_value in {"1", "2", "3"} if is_steered else attention_value == "7"
            )
        elif ix["type"] == "final-questionnaire":
            d = ix.get("data") or {}
            v = d.get("f_attention_check")
            if v is not None:
                final_pass = str(v).strip() == "same"

    phase_pass_all = all(phase_passes) if phase_passes else None
    if phase_pass_all is True and (final_pass in (True, None)):
        overall_pass = True
    elif phase_pass_all is False or final_pass is False:
        overall_pass = False
    else:
        overall_pass = None

    # Look up the approach order from the interaction log first, with a
    # defensive fallback to `Participation.extra_data` (written by
    # `_get_effective_models`).  This way the journey card always knows which
    # phase was the baseline vs. the steered arm, even if the log row is
    # missing.
    approach_order = None
    effective_order = None
    for row in rows:
        if row["type"] == "approach-order-assigned":
            d = row.get("data") or {}
            approach_order = d.get("approach_order") or approach_order
            effective_order = d.get("effective_order") or effective_order
    if approach_order is None or effective_order is None:
        try:
            extra = json.loads(participation.extra_data) if participation.extra_data else {}
            if isinstance(extra, dict):
                approach_order = approach_order or extra.get("approach_order")
                effective_order = effective_order or extra.get("effective_order")
        except Exception:
            pass

    return jsonify(
        {
            "participation_id": participation.id,
            "study_guid": user_study.guid if user_study else None,
            "participant": {
                "uuid": participation.uuid,
                "email": participation.participant_email,
                "language": participation.language,
                "prolific": _build_prolific_block(participation.extra_data, study_config),
                "time_joined": participation.time_joined.isoformat() if participation.time_joined else None,
                "time_finished": participation.time_finished.isoformat() if participation.time_finished else None,
                "approach_order": approach_order,
                "effective_order": effective_order,
            },
            "attention_checks": {
                "phase_pass": phase_pass_all,
                "phase_passed_count": sum(1 for v in phase_passes if v),
                "phase_total_count": len(phase_passes),
                "final_pass": final_pass,
                "overall_pass": overall_pass,
            },
            "timeline": journey["timeline"],
            "summary": journey["summary"],
        }
    )


# ============================================================================
# Cleanup
# ============================================================================

@bp.route("/dispose", methods=["DELETE"])
def dispose():
    """Clean up study data"""
    import shutil
    guid = request.args.get("guid")
    cache_path = get_cache_path(guid)
    
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    
    return "OK"


# ============================================================================
# Plugin Registration
# ============================================================================

def register():
    """Register plugin with EasyStudy framework"""
    return {
        "bep": dict(blueprint=bp, prefix=None)
    }
