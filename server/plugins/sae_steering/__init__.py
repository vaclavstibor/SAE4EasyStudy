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
from plugins.utils.preference_elicitation import load_data, enrich_results
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
    conf["show_general_features"] = conf.get("show_general_features", True)
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
    models = conf.get("models", [])
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
        return "Review recommendations and use the like/dislike buttons."
    return "Adjust features to steer your recommendations."


def _get_steering_guidance(steering_mode: str) -> str:
    steering_mode = _normalize_steering_mode(steering_mode)
    if steering_mode == "text":
        return "Start by reviewing the current recommendations, then describe the kind of change you want in your own words before getting updated recommendations."
    if steering_mode == "both":
        return "Start by reviewing the current recommendations, then either write what you want or adjust the discovered concepts before getting updated recommendations."
    if steering_mode == "none":
        return "Start by reviewing the current recommendations, mark what you like or dislike, and then continue to the next recommendation update."
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


def _load_tmdb_overviews():
    """Load TMDB movie overviews (plots) keyed by movieId. Cached after first call."""
    global _tmdb_cache
    if _tmdb_cache is not None:
        return _tmdb_cache
    tmdb_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'static', 'datasets', 'ml-latest', 'tmdb_data.json'
    )
    _tmdb_cache = {}
    if os.path.exists(tmdb_path):
        try:
            with open(tmdb_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            for entry in raw.values():
                mid = entry.get('movieId')
                overview = entry.get('overview', '')
                if mid and overview:
                    _tmdb_cache[int(mid)] = overview
            print(f"[TMDB] Loaded {len(_tmdb_cache)} movie overviews")
        except Exception as e:
            print(f"[TMDB] Could not load overviews: {e}")
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
            "steering_mode": "text",
            "feature_selection_algorithm": DEFAULT_FEATURE_SELECTION_ALGORITHM,
        },
    ]


def get_cache_path(guid, name=""):
    """Get cache directory path for this plugin"""
    return os.path.join("cache", __plugin_name__, guid, name)


# ============================================================================
# Feature De-duplication Helpers
# ============================================================================

COSINE_DEDUP_THRESHOLD = 0.85
_FUZZY_LABEL_JACCARD_THRESHOLD = 0.65
GROUP_MERGE_COSINE_THRESHOLD = 0.90
GROUP_SUPPORT_COSINE_THRESHOLD = 0.45
GROUP_TEXT_JACCARD_THRESHOLD = 0.25
GROUP_SHARED_TEXT_THRESHOLD = 3


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


def _build_decoder_vecs(recommender) -> dict:
    """Extract L2-normalized decoder weight rows from the SAE for pairwise cosine dedup.

    Returns {neuron_id: normalized_vector} or empty dict on failure.
    """
    try:
        import torch.nn.functional as _F
        sae = recommender.sae_model
        if hasattr(sae, 'decoder_w'):
            _W = sae.decoder_w.detach().cpu()
        elif hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
            _W = sae.decoder.weight.detach().cpu().T
        else:
            print("[dedup] SAE model has no decoder_w or decoder.weight — cosine dedup disabled")
            return {}
        _W = _F.normalize(_W, dim=1)
        return {nid: _W[nid] for nid in range(_W.shape[0])}
    except Exception as e:
        print(f"[dedup] Failed to extract decoder vectors: {e}")
        return {}


def _is_cosine_duplicate(neuron_id: int, selected_ids: list, decoder_vecs: dict) -> bool:
    """Return True if *neuron_id* has cosine > threshold with any already-selected neuron."""
    if not decoder_vecs or neuron_id not in decoder_vecs:
        return False
    vec = decoder_vecs[neuron_id]
    for sel_id in selected_ids:
        if sel_id in decoder_vecs:
            cos = float(vec @ decoder_vecs[sel_id])
            if cos > COSINE_DEDUP_THRESHOLD:
                return True
    return False


_GROUP_TOKEN_STOP = {
    'the', 'and', 'of', 'in', 'a', 'an', 'to', 'for', 'with', 'on', 'at',
    'by', 'from', 'its', 'that', 'this', 'but', 'or', 'as', 'into', 'these',
    'those', 'their', 'while', 'where', 'through', 'often', 'more', 'like',
    'movie', 'movies', 'film', 'films', 'story', 'stories', 'storytelling',
    'narrative', 'narratives', 'cinematic', 'cinema',
}


def _canonicalize_group_token(token: str) -> str:
    """Light stemming for label/description token matching."""
    token = token.strip().lower()
    if len(token) <= 2 or token.isdigit():
        return ''
    if token.endswith('ies') and len(token) > 4:
        token = token[:-3] + 'y'
    elif token.endswith('es') and len(token) > 4 and not token.endswith('sses'):
        token = token[:-2]
    elif token.endswith('s') and len(token) > 4:
        token = token[:-1]
    return token


def _text_token_set(text: str) -> set:
    """Tokenize text into normalized content words for grouping."""
    import re
    tokens = set()
    for raw in re.split(r'[\s·\-–—/,&:;.!?()\[\]{}"\']+', (text or '').lower()):
        token = _canonicalize_group_token(raw)
        if token and token not in _GROUP_TOKEN_STOP:
            tokens.add(token)
    return tokens


def _feature_text_tokens(feature: dict) -> set:
    """Return normalized content tokens from label + description."""
    return _text_token_set(
        f"{feature.get('label', '')} {feature.get('description', '')}"
    )


def _jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    """Compute Jaccard similarity for token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _pairwise_decoder_cosine(neuron_a: int, neuron_b: int, decoder_vecs: dict) -> float:
    """Cosine similarity between two decoder vectors when available."""
    if not decoder_vecs:
        return 0.0
    if neuron_a not in decoder_vecs or neuron_b not in decoder_vecs:
        return 0.0
    return float(decoder_vecs[neuron_a] @ decoder_vecs[neuron_b])


def _ensure_feature_group_metadata(feature: dict) -> dict:
    """Ensure every slider payload has group metadata."""
    feature = dict(feature)
    member_ids = feature.get('member_ids') or [int(feature['id'])]
    member_ids = [int(nid) for nid in member_ids]
    if int(feature['id']) not in member_ids:
        member_ids.insert(0, int(feature['id']))
    seen_ids = []
    for nid in member_ids:
        if nid not in seen_ids:
            seen_ids.append(nid)
    member_labels = feature.get('member_labels') or [feature.get('label', f"Neuron N{feature['id']}")]
    deduped_labels = []
    for label in member_labels:
        if label and label not in deduped_labels:
            deduped_labels.append(label)
    feature['member_ids'] = seen_ids
    feature['member_labels'] = deduped_labels
    feature['group_size'] = len(seen_ids)
    feature.setdefault('group_movie_count', int(feature.get('movie_count', 0) or 0))
    feature.setdefault('_base_description', feature.get('description', ''))
    return feature


def _feature_group_match_score(candidate: dict, anchor: dict, decoder_vecs: dict) -> float:
    """Similarity score used for post-label grouping of related sliders."""
    candidate_label = candidate.get('label', '')
    anchor_label = anchor.get('label', '')
    if _is_near_duplicate_label(candidate_label, {_normalize_label(anchor_label)}):
        return 1.0

    label_sim = _jaccard_similarity(
        _text_token_set(candidate_label),
        _text_token_set(anchor_label),
    )
    text_tokens_a = _feature_text_tokens(candidate)
    text_tokens_b = _feature_text_tokens(anchor)
    text_sim = _jaccard_similarity(text_tokens_a, text_tokens_b)
    shared_text = len(text_tokens_a & text_tokens_b)
    cosine = _pairwise_decoder_cosine(int(candidate['id']), int(anchor['id']), decoder_vecs)

    if cosine >= GROUP_MERGE_COSINE_THRESHOLD:
        return 0.95
    if label_sim >= 0.50:
        return 0.90
    if label_sim >= 0.33 and text_sim >= GROUP_TEXT_JACCARD_THRESHOLD:
        return 0.80
    if shared_text >= GROUP_SHARED_TEXT_THRESHOLD and cosine >= GROUP_SUPPORT_COSINE_THRESHOLD:
        return 0.75
    if text_sim >= 0.35 and shared_text >= 2:
        return 0.70
    return 0.0


def _merge_feature_group(anchor: dict, candidate: dict) -> dict:
    """Attach *candidate* as an additional member of *anchor* slider group."""
    anchor.update(_ensure_feature_group_metadata(anchor))
    candidate = _ensure_feature_group_metadata(candidate)

    for nid in candidate['member_ids']:
        if nid not in anchor['member_ids']:
            anchor['member_ids'].append(int(nid))

    for label in candidate.get('member_labels', []):
        if label and label not in anchor['member_labels']:
            anchor['member_labels'].append(label)

    anchor['group_size'] = len(anchor['member_ids'])
    anchor['group_movie_count'] = max(
        int(anchor.get('group_movie_count', anchor.get('movie_count', 0)) or 0),
        int(candidate.get('group_movie_count', candidate.get('movie_count', 0)) or 0),
    )
    anchor['_group_score'] = float(anchor.get('_group_score', anchor.get('_user_score', 0.0))) + \
        float(candidate.get('_user_score', 0.0))

    base_desc = anchor.get('_base_description', anchor.get('description', '')) or anchor.get('label', '')
    if anchor['group_size'] > 1:
        anchor['description'] = (
            f"{base_desc} This slider groups {anchor['group_size']} closely related latent features."
        ).strip()
    else:
        anchor['description'] = base_desc
    return anchor


def _select_grouped_slider_features(
    labeled: list,
    score_map: dict,
    decoder_vecs: dict,
    num_sliders: int,
    min_neuron_movies: int,
) -> tuple[list, list]:
    """Cluster similar neurons and pick one representative slider per cluster.

    Algorithm:
      1. Filter out placeholders and low-count neurons.
      2. Build a pairwise similarity matrix (text tokens + decoder cosine).
      3. Greedy agglomerative: walk candidates in descending user-score order;
         merge into the most-similar existing cluster if similarity exceeds
         CLUSTER_MERGE_SIM, else start a new cluster.
      4. Pick up to *num_sliders* clusters, sorted by aggregate user-score.
      5. Final diversity gate: reject clusters whose representative label is
         Jaccard-near-duplicate of an already-selected cluster.

    Each selected cluster becomes one UI slider. Its ``member_ids`` list
    contains all merged neuron IDs; ``_member_scores`` maps nid→user_score
    so that ``_expand_feature_adjustments`` can distribute slider deltas
    proportionally.
    """

    CLUSTER_MERGE_SIM = 0.25
    MAX_NEURON_MOVIES = 5000

    rejections: list = []
    valid_candidates: list = []

    for raw_f in labeled:
        f = _ensure_feature_group_metadata(raw_f)
        f['_user_score'] = float(score_map.get(int(f['id']), 0.0))

        lbl = _normalize_label(f.get('label', ''))
        if lbl.startswith('feature ') or lbl.startswith('neuron n') or lbl.startswith('latent concept'):
            rejections.append((f['id'], f.get('label', ''), 'placeholder'))
            continue
        mc = int(f.get('movie_count', 0) or 0)
        if mc < min_neuron_movies:
            rejections.append((f['id'], f.get('label', ''), 'low_movie_count'))
            continue
        if mc > MAX_NEURON_MOVIES:
            rejections.append((f['id'], f.get('label', ''), 'too_broad'))
            continue
        valid_candidates.append(f)

    valid_candidates.sort(key=lambda x: -x['_user_score'])

    def _combined_similarity(a: dict, b: dict) -> float:
        label_j = _jaccard_similarity(
            _text_token_set(a.get('label', '')),
            _text_token_set(b.get('label', '')),
        )
        desc_j = _jaccard_similarity(
            _feature_text_tokens(a),
            _feature_text_tokens(b),
        )
        dec_cos = _pairwise_decoder_cosine(int(a['id']), int(b['id']), decoder_vecs)
        weighted = 0.25 * label_j + 0.45 * desc_j + 0.30 * max(dec_cos, 0.0)
        return max(weighted, desc_j * 0.55)

    clusters: list = []

    for cand in valid_candidates:
        best_cluster = None
        best_sim = 0.0
        for cl in clusters:
            anchor = cl[0]
            sim = _combined_similarity(cand, anchor)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cl
        if best_cluster is not None and best_sim >= CLUSTER_MERGE_SIM:
            best_cluster.append(cand)
        else:
            clusters.append([cand])

    LABEL_DIVERSITY_JACCARD = 0.45

    def _label_too_similar(lbl: str, existing: list) -> bool:
        """Reject if label-token Jaccard >= threshold with ANY selected slider."""
        norm = _normalize_label(lbl)
        tokens_new = _text_token_set(lbl)
        for sel in existing:
            if _normalize_label(sel.get('label', '')) == norm:
                return True
            tokens_old = _text_token_set(sel.get('label', ''))
            if _jaccard_similarity(tokens_new, tokens_old) >= LABEL_DIVERSITY_JACCARD:
                return True
        return False

    selected: list = []

    for cl in sorted(clusters,
                     key=lambda c: -sum(f['_user_score'] for f in c)):
        if len(selected) >= num_sliders:
            break
        anchor = cl[0]
        lbl = anchor.get('label', '')
        if _label_too_similar(lbl, selected):
            for f in cl:
                rejections.append((f['id'], f.get('label', ''), 'cluster_dup_label'))
            continue

        member_ids = []
        member_labels = []
        member_scores: dict = {}
        total_score = 0.0
        max_mc = 0
        for f in cl:
            nid = int(f['id'])
            if nid not in member_ids:
                member_ids.append(nid)
            fl = f.get('label', '')
            if fl and fl not in member_labels:
                member_labels.append(fl)
            sc = f['_user_score']
            member_scores[nid] = sc
            total_score += sc
            max_mc = max(max_mc, int(f.get('movie_count', 0) or 0))

        rep = dict(anchor)
        rep['member_ids'] = member_ids
        rep['member_labels'] = member_labels
        rep['_member_scores'] = member_scores
        rep['group_size'] = len(member_ids)
        rep['group_movie_count'] = max_mc
        rep['_group_score'] = total_score
        rep['_user_score'] = anchor['_user_score']
        if len(member_ids) > 1:
            base_desc = anchor.get('description', anchor.get('label', ''))
            rep['description'] = (
                f"{base_desc} "
                f"This slider groups {len(member_ids)} closely related latent features."
            ).strip()
        selected.append(rep)

    selected.sort(key=lambda f: (
        -float(f.get('_group_score', f.get('_user_score', 0.0))),
        -int(f.get('group_size', 1)),
        -int(f.get('movie_count', 0) or 0),
    ))
    return selected[:num_sliders], rejections


MEMBER_SUPPORT_WEIGHT = 0.20


def _expand_feature_adjustments(
    raw_adjustments: dict,
    current_features=None,
    cluster_map=None,
) -> dict:
    """Expand grouped slider IDs and cluster IDs into neuron-level deltas.

    For grouped sliders the **anchor neuron** (first in ``member_ids``,
    the highest-scoring cluster representative) receives the **full slider
    delta**.  Every other member neuron receives a smaller support fraction
    (``MEMBER_SUPPORT_WEIGHT * delta``).  This keeps the slider signal
    strong enough to visibly shift recommendations even against a seed
    profile with 200+ active neurons totalling ~37 in weight.
    """
    feature_adjustments: dict = {}
    current_features = current_features or []
    cluster_map = cluster_map or {}

    grouped_members: dict = {}
    for feature in current_features:
        gf = _ensure_feature_group_metadata(feature)
        fid = str(gf['id'])
        grouped_members[fid] = [int(nid) for nid in gf['member_ids']]

    for key, val in (raw_adjustments or {}).items():
        if key.startswith("cluster_") and key in cluster_map:
            neuron_ids = cluster_map.get(key) or []
            if neuron_ids:
                per_neuron = float(val)
                for nid in neuron_ids:
                    skey = str(nid)
                    feature_adjustments[skey] = feature_adjustments.get(skey, 0.0) + per_neuron
            continue

        ids = grouped_members.get(str(key))
        if ids:
            anchor = ids[0]
            delta = float(val)
            for nid in ids:
                w = 1.0 if nid == anchor else MEMBER_SUPPORT_WEIGHT
                skey = str(nid)
                feature_adjustments[skey] = feature_adjustments.get(skey, 0.0) + delta * w
        else:
            skey = str(key)
            feature_adjustments[skey] = feature_adjustments.get(skey, 0.0) + float(val)

    return feature_adjustments


# ============================================================================
# Semantic Cluster Profile ("Static User Profile")
# ============================================================================

# Genre buckets that define each cluster's identity.
# These MUST match the bucket definitions in generate_cluster_profile.py.
_CLUSTER_GENRE_CENTROIDS = {
    "Action & Adventure":       {"Action": 1.0, "Adventure": 0.7, "IMAX": 0.3},
    "Sci-Fi & Fantasy":         {"Sci-Fi": 1.0, "Fantasy": 0.8},
    "Thriller & Crime":         {"Thriller": 1.0, "Crime": 0.8, "Mystery": 0.5, "Film-Noir": 0.3},
    "Horror & Suspense":        {"Horror": 1.0, "Thriller": 0.3},
    "Comedy & Lighthearted":    {"Comedy": 1.0, "Musical": 0.3},
    "Romance & Family":         {"Romance": 1.0, "Children": 0.7, "Animation": 0.5},
    "Documentary & Niche":      {"Documentary": 1.0, "War": 0.5, "Western": 0.5},
}


def _compute_user_genre_vector(selected_movies, model_id: str = None):
    """Build a genre vector from the user's selected movies."""
    try:
        from plugins.utils.data_loading import load_ml_dataset
        loader = load_ml_dataset()
        movies_df = loader.movies_df_indexed
    except Exception:
        return None

    genre_vec = {}
    n = 0
    for mid in selected_movies:
        try:
            mid = int(mid)
            if mid not in movies_df.index:
                continue
            row = movies_df.loc[mid]
            gs = row.get("genres", "")
            if isinstance(gs, str):
                for g in gs.split("|"):
                    g = g.strip()
                    if g and g != "(no genres listed)":
                        genre_vec[g] = genre_vec.get(g, 0) + 1.0
            n += 1
        except (ValueError, TypeError, KeyError):
            continue
    if n > 0:
        for g in genre_vec:
            genre_vec[g] /= n
    return genre_vec if genre_vec else None


def _compute_cluster_genre_centroids(cluster_profile, model_id: str = None):
    """Return the ideal genre centroid for each cluster.

    Uses the fixed genre-bucket definitions (not computed from neuron
    activations, which converge to the dataset average and lose
    discriminative power).
    """
    result = {}
    for c in cluster_profile:
        label = c.get("label", "")
        centroid = _CLUSTER_GENRE_CENTROIDS.get(label, {})
        if not centroid:
            for key, val in _CLUSTER_GENRE_CENTROIDS.items():
                if key.split(" & ")[0] in label or key.split(" & ")[-1] in label:
                    centroid = val
                    break
        result[c["id"]] = centroid
    return result


def _build_cluster_profile(model_id: str = None, num_clusters: int = 7):
    """Load pre-generated semantic cluster profile.

    The profile must be generated offline via generate_cluster_profile.py
    (KMeans on sentence embeddings of LLM labels + LLM-based naming).

    Returns a list of cluster dicts:
        [{id, label, description, genres, neuron_ids}, ...]
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    cache_path = os.path.join(data_dir, f"cluster_profile_{model_id or 'default'}.json")

    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached and len(cached) >= 1:
                print(f"[cluster_profile] Loaded {len(cached)} clusters from {cache_path}")
                return cached
        except Exception as e:
            print(f"[cluster_profile] Error reading cache: {e}")

    print(f"[cluster_profile] No pre-generated profile found at {cache_path}")
    print(f"[cluster_profile] Run: python generate_cluster_profile.py --model {model_id or 'default'}")
    return [{"id": f"cluster_{i}", "label": f"Cluster {i+1}", "description": "",
             "genres": [], "neuron_ids": []} for i in range(num_clusters)]


def _get_cluster_for_neuron(neuron_id: int, cluster_profile: list) -> str:
    """Return the cluster_id a neuron belongs to, or '' if none."""
    for c in cluster_profile:
        if neuron_id in c["neuron_ids"]:
            return c["id"]
    return ""


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
    # POC: Start with MovieLens
    return jsonify([
        {
            "name": "MovieLens Latest Small",
            "id": "ml-latest-small",
            "description": "Small MovieLens dataset for testing"
        },
        {
            "name": "MovieLens Latest",
            "id": "ml-latest",
            "description": "Full MovieLens dataset"
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
            "name": "No Steering (likes/dislikes only)",
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
    """
    Return list of all available neurons for manual selection in study creation.
    
    Uses the active SAE model's LLM labels when possible.
    """
    model_id = request.args.get("model_id") or DEFAULT_TOPK_SAE_MODEL_ID
    if not model_id or str(model_id).strip().lower() == "none":
        model_id = DEFAULT_TOPK_SAE_MODEL_ID
    neurons = []

    try:
        from .sae_recommender import get_sae_recommender
        from .llm_labeling import get_llm_labels
        from .dynamic_labeling import get_dynamic_labels

        rec = get_sae_recommender(model_id=model_id)
        rec.load()
        if rec.item_features is None or rec.item_ids is None:
            return jsonify([])

        labels = {}
        try:
            labels = dict(get_llm_labels(
                model_id=model_id,
                item_features=rec.item_features,
                item_ids=rec.item_ids,
            ))
        except Exception:
            labels = {}

        try:
            dyn = get_dynamic_labels(
                model_id=model_id,
                item_features=rec.item_features,
                item_ids=rec.item_ids,
            )
            for nid, info in dyn.items():
                nid_int = int(nid)
                if nid_int not in labels:
                    labels[nid_int] = info
        except Exception:
            pass

        for nid, info in labels.items():
            label = (info or {}).get("label", "")
            description = (info or {}).get("description", "")
            if not label or label.lower().startswith("feature "):
                continue
            neurons.append({
                "id": int(nid),
                "label": label,
                "category": info.get("category", "latent"),
                "description": description or f"Neuron {nid}",
                "score": info.get("activation_count", 0),
            })

        neurons.sort(key=lambda n: (-n.get("score", 0), n["label"].lower()))
        neurons = neurons[:150]

    except Exception as e:
        print(f"[available_neurons] Error: {e}")
        traceback.print_exc()
        # Return fallback neurons
        neurons = [
            {'id': i, 'label': f'Feature {i}', 'category': 'other', 'description': f'Neuron {i}'}
            for i in range(20)
        ]
    
    return jsonify(neurons)


# ============================================================================
# Study Initialization
# ============================================================================

def long_initialization(guid):
    """
    Long-running initialization process for SAE steering study.
    Runs in separate process to avoid blocking.
    """
    engine = create_engine('sqlite:///instance/db.sqlite')
    db_session = Session(engine)
    
    try:
        user_study = db_session.query(UserStudy).filter(UserStudy.guid == guid).first()
        conf = _normalize_study_config(json.loads(user_study.settings))
        
        # Create cache directories
        Path(get_cache_path(guid)).mkdir(parents=True, exist_ok=True)
        Path(get_cache_path(guid, "sae_model")).mkdir(parents=True, exist_ok=True)
        Path(get_cache_path(guid, "embeddings")).mkdir(parents=True, exist_ok=True)
        
        import shutil

        # Move uploaded questionnaire file into cache if provided
        if "questionnaire_file" in conf:
            src = os.path.join("cache", __plugin_name__, "uploads", conf["questionnaire_file"])
            if os.path.exists(src):
                shutil.move(src, get_cache_path(guid, conf["questionnaire_file"]))

        # Move uploaded phase questionnaire files into cache if provided
        phase_files = set()
        if conf.get("phase_questionnaire_file"):
            phase_files.add(conf["phase_questionnaire_file"])
        for model in conf.get("models", []):
            if model.get("phase_questionnaire_file"):
                phase_files.add(model["phase_questionnaire_file"])
        for phase_file in phase_files:
            src = os.path.join("cache", __plugin_name__, "uploads", phase_file)
            if os.path.exists(src):
                shutil.move(src, get_cache_path(guid, phase_file))

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


@bp.route("/study-intro", methods=["GET"])
def study_intro():
    """Show intro page explaining what the study involves before proceeding."""
    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    tr = get_tr(languages, get_lang())

    models = conf.get("models", [])
    comparison_mode = conf.get("comparison_mode", "side_by_side")
    num_phases = len(models) if comparison_mode == "sequential" else 1
    num_iterations = conf.get("num_iterations", 3)
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
            "none": "likes/dislikes only",
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
        "num_phases": num_phases,
        "num_iterations": num_iterations,
        "has_questionnaire": has_questionnaire,
        "steering_label_a": steering_label_a,
        "steering_label_b": steering_label_b,
        "notes": conf.get("intro_notes", [
            "There are no right or wrong answers — just explore what you like.",
            "Your data is anonymous and used for research purposes only.",
        ]),
        "study_parts": [
            {
                "title": "Preference elicitation",
                "description": "Choose a few movies you like so the system can estimate your starting taste profile.",
            },
            {
                "title": "Iteration review",
                "description": "At the start of each round, review the shown recommendations and confirm your likes/dislikes before continuing.",
            },
            {
                "title": "Steering",
                "description": "After approval, refine the next recommendations with the steering controls available in the current approach.",
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
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in item_search: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Feature Display & Steering
# ============================================================================

def _load_cached_model_labels(model_id: str = None) -> dict:
    """Load pre-generated label caches without requiring runtime SAE activations."""
    data_dir = Path(__file__).parent / "data"
    resolved_model_id = model_id or DEFAULT_TOPK_SAE_MODEL_ID
    labels: dict = {}

    def _candidate_paths(pattern: str, specific_name: str) -> list:
        specific = data_dir / specific_name
        if specific.exists():
            return [specific]
        return sorted(data_dir.glob(pattern))

    def _merge_from_paths(paths: list) -> None:
        for path in paths:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                for nid_str, info in raw.items():
                    try:
                        nid = int(nid_str)
                    except (TypeError, ValueError):
                        continue
                    if nid in labels:
                        continue
                    label = (info.get("label") or "").strip()
                    if not label:
                        continue
                    labels[nid] = {
                        "label": label,
                        "description": info.get("description", ""),
                        "activation_count": info.get("activation_count", 0),
                        "selectivity": info.get("selectivity", 0.0),
                        "tags": info.get("tags", []),
                        "genres": info.get("genres", []),
                    }
            except Exception as exc:
                print(f"[_load_cached_model_labels] Failed to read {path.name}: {exc}")

    _merge_from_paths(_candidate_paths("llm_labels_*_llm.json", f"llm_labels_{resolved_model_id}_llm.json"))
    _merge_from_paths(_candidate_paths("dynamic_labels_*_v*.json", f"dynamic_labels_{resolved_model_id}_v2.json"))
    return labels


def _get_cached_sae_features(top_k: int = 21, model_id: str = None) -> list:
    """Build a feature list from cached label catalogs when SAE runtime data is absent."""
    try:
        from .dynamic_labeling import select_features_for_display

        cached_labels = _load_cached_model_labels(model_id=model_id)
        if not cached_labels:
            return []

        features = select_features_for_display(cached_labels, top_k=top_k)
        for feature in features:
            info = cached_labels.get(int(feature["id"]), {})
            if info.get("description"):
                feature["description"] = info["description"]
            if info.get("activation_count"):
                feature["movie_count"] = info["activation_count"]
        if features:
            print(
                f"[_get_cached_sae_features] Using cached labels for "
                f"{model_id or DEFAULT_TOPK_SAE_MODEL_ID}: {len(features)} features"
            )
        return features
    except Exception as exc:
        print(f"[_get_cached_sae_features] Failed: {exc}")
        return []


def get_sae_features(top_k: int = 21, model_id: str = None) -> list:
    """
    Select the most interpretable SAE neurons as user-facing features.

    **Model-agnostic**: works for ANY SAE model (prediction-aware, WWW TopK,
    basic SAE, etc.) by dynamically deriving neuron labels from the model's
    actual activations + MovieLens genome tags.

    Pipeline:
      1. Load the SAE recommender (which has pre-computed item_features).
      2. Call dynamic_labeling.get_dynamic_labels() to derive human-readable
         labels from the activation patterns + MovieLens metadata.
      3. Call dynamic_labeling.select_features_for_display() to pick a
         diverse, interpretable subset for the steering UI.
      4. Falls back to static neuron_labels.json only if the dynamic
         system fails.
    """
    try:
        # --- LLM labels are the ONLY labeling path ---
        from .sae_recommender import get_sae_recommender
        from .dynamic_labeling import select_features_for_display

        recommender = get_sae_recommender(model_id=model_id)
        recommender.load()

        if recommender.item_features is not None and recommender.item_ids is not None:
            effective_model_id = model_id or recommender.model_id or 'default'
            if effective_model_id is None:
                effective_model_id = DEFAULT_TOPK_SAE_MODEL_ID

            dynamic_labels = None
            try:
                from .llm_labeling import get_llm_labels
                dynamic_labels = get_llm_labels(
                    model_id=effective_model_id,
                    item_features=recommender.item_features,
                    item_ids=recommender.item_ids,
                )
                print(f"[get_sae_features] Using LLM labels ({len(dynamic_labels)} neurons)")
            except RuntimeError as e:
                print(f"[get_sae_features] LLM labeling error: {e}")
                raise
            except Exception as e:
                print(f"[get_sae_features] LLM labels failed: {e}")

            if dynamic_labels:
                features = select_features_for_display(dynamic_labels, top_k=top_k)
                if features:
                    # Enrich features with LLM description if available
                    for f in features:
                        nid = f['id']
                        info = dynamic_labels.get(nid, {})
                        if info.get('description'):
                            f['description'] = info['description']
                    return features

        cached_features = _get_cached_sae_features(top_k=top_k, model_id=model_id)
        if cached_features:
            return cached_features

        print("[get_sae_features] Dynamic labeling unavailable, trying static fallback")

    except Exception as e:
        print(f"[get_sae_features] Dynamic labeling failed: {e}")
        cached_features = _get_cached_sae_features(top_k=top_k, model_id=model_id)
        if cached_features:
            return cached_features
        traceback.print_exc()

    # --- Static fallback is only safe for the legacy prediction-aware model ---
    if model_id in (None, "prediction_aware_sae"):
        return _get_sae_features_static(top_k=top_k)
    return _get_fallback_features()


def _get_sae_features_static(top_k: int = 21) -> list:
    """
    Static fallback: uses neuron_labels.json and neuron_analysis.json
    (only valid for the prediction-aware SAE model).
    """
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"
    features = []

    KNOWN_GENRES = {
        'action', 'adventure', 'animation', 'children', 'comedy', 'crime',
        'documentary', 'drama', 'fantasy', 'film-noir', 'horror', 'imax',
        'musical', 'mystery', 'romance', 'sci-fi', 'science fiction',
        'thriller', 'war', 'western', 'family',
    }

    def _is_decade(t):
        return len(t) >= 4 and t[-1] == 's' and t[:-1].isdigit()

    def _is_trivial(t):
        return t in KNOWN_GENRES or _is_decade(t)

    def _label_has_concept(label):
        tokens = {t.strip().lower() for t in label.split('•') if t.strip()}
        return any(not _is_trivial(t) for t in tokens)

    def _clean_label(label):
        parts = [t.strip() for t in label.split('•') if t.strip()]
        has_concept = any(not _is_trivial(p.lower()) for p in parts)
        if has_concept:
            parts = [p for p in parts if not _is_decade(p.strip().lower())]
        return ' · '.join(p.title() for p in parts) if parts else label

    try:
        labels_path = data_dir / "neuron_labels.json"
        analysis_path = data_dir / "neuron_analysis.json"
        if not labels_path.exists() or not analysis_path.exists():
            return _get_fallback_features()

        with open(labels_path) as f:
            neuron_labels = json.load(f)
        with open(analysis_path) as f:
            neuron_analysis = json.load(f)

        blocked_tokens = {
            'nudity', 'sex', 'prostitution', 'erotic', 'erotica',
            'softcore', 'rape', 'silent film', 'transvestism',
        }

        candidates = []
        for nid_str, info in neuron_analysis.items():
            if not info.get('selective'):
                continue
            label = neuron_labels.get(nid_str, '')
            if not label or 'Niche' in label or 'General' in label:
                continue
            if any(tok in label.lower() for tok in blocked_tokens):
                continue

            act_count = info.get('activation_count', 0)
            act_sum = info.get('activation_sum', 0)
            selectivity = info.get('selectivity', 0)

            if act_count < 50 or act_count > 30000:
                continue
            if selectivity < 0.85:
                continue

            avg_act = act_sum / max(act_count, 1)
            has_concept = _label_has_concept(label)
            concept_bonus = 2.0 if has_concept else 1.0
            score = selectivity * np.log(act_count + 1) * avg_act * concept_bonus

            candidates.append({
                'neuron_id': int(nid_str),
                'label': label,
                'clean_label': _clean_label(label),
                'score': score,
                'has_concept': has_concept,
                'activation_count': act_count,
            })

        candidates.sort(key=lambda x: -x['score'])

        selected = []
        used_tokens = set()
        pure_genre_count = 0

        for c in candidates:
            if len(selected) >= top_k:
                break
            tokens = {t.strip().lower() for t in c['label'].split('•') if t.strip()}
            content = {t for t in tokens if not _is_decade(t)}
            if content and content <= used_tokens:
                continue
            if not c['has_concept']:
                if pure_genre_count >= 7:
                    continue
                pure_genre_count += 1
            selected.append(c)
            used_tokens |= content

        for s in selected:
            features.append({
                'id': s['neuron_id'],
                'label': s['clean_label'],
                'category': 'latent',
                'description': f"Latent concept (N{s['neuron_id']}): {s['clean_label']}",
                'top_tags': [],
                'activation': 0.5,
                'movie_count': s['activation_count'],
            })

        features.sort(key=lambda f: -f['movie_count'])
        print(f"[SAE Features/Static] {len(features)} neurons selected")
        return features

    except Exception as e:
        print(f"[SAE Features/Static] Error: {e}")
        traceback.print_exc()
        return _get_fallback_features()


def _get_fallback_features() -> list:
    """Fallback POC features when SAE data not available."""
    feature_labels = [
        ("Action", "genre", "Action and adventure movies"),
        ("Comedy", "genre", "Funny and light-hearted films"),
        ("Drama", "genre", "Character-driven emotional stories"),
        ("Sci-Fi", "genre", "Science fiction and futuristic"),
        ("Horror", "genre", "Scary and suspenseful"),
        ("Romance", "genre", "Love stories and relationships"),
        ("Thriller", "genre", "Tension and mystery"),
        ("Animation", "genre", "Animated films"),
    ]
    
    return [
        {
            'id': i,
            'label': label,
            'category': cat,
            'description': desc,
            'top_tags': [{'name': label, 'category': cat, 'weight': 0.8}],
            'activation': 0.5,
            'specificity': 1.0
        }
        for i, (label, cat, desc) in enumerate(feature_labels)
    ]


def _personalized_features(
    selected_movies: list,
    model_id: str = None,
    num_sliders: int = 21,
) -> list:
    """Select SAE neurons personalized to THIS user's elicitation picks.

    Pipeline (follows P3 SeqSAE / P2 RecSAE correlation approach):
      1. Load the SAE item_features matrix (neuron activations per movie).
      2. Compute mean activation across the user's selected movies.
      3. Rank neurons by mean activation for this user → these are the
         concepts most relevant to what the user already likes.
      4. Pick top neurons ensuring label diversity (no duplicates).
      5. Fall back to the global top-N if the user selected no movies
         or the SAE model is unavailable.

    This ensures each user sees a DIFFERENT set of sliders tailored to
    their taste profile from preference elicitation.
    """
    import torch as _torch

    # Fall back to global features if no elicitation data
    if not selected_movies:
        print("[_personalized_features] No selected movies → global fallback")
        return get_sae_features(top_k=num_sliders, model_id=model_id)

    print(f"[_personalized_features] Starting with {len(selected_movies)} movies, "
          f"model={model_id}, num_sliders={num_sliders}")

    try:
        from .sae_recommender import get_sae_recommender

        recommender = get_sae_recommender(model_id=model_id)
        recommender.load()

        if recommender.item_features is None or recommender.item_ids is None:
            print("[_personalized_features] item_features/item_ids None → global fallback")
            return get_sae_features(top_k=num_sliders, model_id=model_id)

        # Build movieId → index lookup
        id_to_idx = {int(mid): i for i, mid in enumerate(recommender.item_ids)}

        # Compute mean SAE activation across selected movies
        acts = []
        for mid in selected_movies:
            idx = id_to_idx.get(int(mid))
            if idx is not None:
                a = recommender.item_features[idx]
                if isinstance(a, _torch.Tensor):
                    a = a.cpu().numpy()
                acts.append(a)

        if not acts:
            print(f"[_personalized_features] No movies matched item_ids "
                  f"(tried {len(selected_movies)}) → global fallback")
            return get_sae_features(top_k=num_sliders, model_id=model_id)

        print(f"[_personalized_features] {len(acts)}/{len(selected_movies)} movies matched")
        mean_act = np.mean(acts, axis=0)

        # Compute per-neuron activation count (number of movies that activate it)
        features_np = recommender.item_features
        if isinstance(features_np, _torch.Tensor):
            features_np = features_np.cpu().numpy()
        neuron_act_counts = np.sum(features_np > 0, axis=0)  # shape: (n_neurons,)

        # Rank neurons by user-specific activation strength
        neuron_scores = list(enumerate(mean_act))
        neuron_scores.sort(key=lambda x: x[1], reverse=True)

        # Get labels for top candidate neurons (fetch more than needed for diversity filtering)
        # Skip neurons that activate on fewer than 20 movies — too niche / unreliable
        MIN_NEURON_MOVIES = 20
        candidate_ids = [
            int(nid) for nid, sc in neuron_scores
            if sc > 0 and int(neuron_act_counts[nid]) >= MIN_NEURON_MOVIES
        ][:num_sliders * 8]

        if not candidate_ids:
            print("[_personalized_features] No candidate neurons with positive activation "
                  "→ global fallback")
            return get_sae_features(top_k=num_sliders, model_id=model_id)
        print(f"[_personalized_features] {len(candidate_ids)} candidate neurons "
              f"(top: N{candidate_ids[0]} score={mean_act[candidate_ids[0]]:.4f})")

        labeled = get_neurons_by_ids(candidate_ids, model_id=model_id)

        # Supplement with dynamic labels for neurons that got placeholder names
        try:
            from .dynamic_labeling import get_dynamic_labels
            dyn_labels = get_dynamic_labels(
                model_id=model_id or DEFAULT_TOPK_SAE_MODEL_ID,
                item_features=recommender.item_features,
                item_ids=recommender.item_ids,
            )
            for f in labeled:
                lbl = f.get('label', '').lower().strip()
                if (lbl.startswith('feature ') or lbl.startswith('neuron n')
                        or lbl.startswith('latent concept')):
                    dl = dyn_labels.get(f['id']) or dyn_labels.get(str(f['id']))
                    if dl and dl.get('label'):
                        f['label'] = dl['label']
                        f['description'] = dl.get('description', f.get('description', ''))
                        f['movie_count'] = dl.get('activation_count', f.get('movie_count', 0))
        except Exception as e_dyn:
            print(f"[_personalized_features] Dynamic label supplement failed: {e_dyn}")

        score_map = {int(nid): sc for nid, sc in neuron_scores}

        # Build decoder-weight vectors for cosine de-duplication (shared helper)
        decoder_vecs = _build_decoder_vecs(recommender)

        selected, merge_rejections = _select_grouped_slider_features(
            labeled=labeled,
            score_map=score_map,
            decoder_vecs=decoder_vecs,
            num_sliders=num_sliders,
            min_neuron_movies=MIN_NEURON_MOVIES,
        )
        if merge_rejections:
            reason_counts = {}
            for _, _, reason in merge_rejections:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            print(f"[_personalized_features] Candidate post-filter summary: {reason_counts}")
        seen_labels = {_normalize_label(f.get('label', '')) for f in selected}

        # Pad with dynamic labels if still short
        if len(selected) < num_sliders:
            used_ids = {nid for f in selected for nid in f.get('member_ids', [f['id']])}
            try:
                from .dynamic_labeling import get_dynamic_labels, select_features_for_display
                dyn = get_dynamic_labels(
                    model_id=model_id or DEFAULT_TOPK_SAE_MODEL_ID,
                    item_features=recommender.item_features,
                    item_ids=recommender.item_ids,
                )
                dyn_features = select_features_for_display(dyn, top_k=num_sliders * 2)
                for df in dyn_features:
                    if df['id'] not in used_ids:
                        df = _ensure_feature_group_metadata(df)
                        if _is_near_duplicate_label(df.get('label', ''), seen_labels):
                            continue
                        if _is_cosine_duplicate(df['id'], [s['id'] for s in selected], decoder_vecs):
                            continue
                        seen_labels.add(_normalize_label(df.get('label', '')))
                        selected.append(df)
                        used_ids.add(df['id'])
                        if len(selected) >= num_sliders:
                            break
            except Exception:
                pass

        # Final fallback: global LLM-labeled features (with full dedup)
        if len(selected) < num_sliders:
            used_ids = {nid for f in selected for nid in f.get('member_ids', [f['id']])}
            global_features = get_sae_features(top_k=num_sliders * 2, model_id=model_id)
            for gf in global_features:
                if gf['id'] not in used_ids:
                    gf = _ensure_feature_group_metadata(gf)
                    if _is_near_duplicate_label(gf.get('label', ''), seen_labels):
                        continue
                    if _is_cosine_duplicate(gf['id'], [s['id'] for s in selected], decoder_vecs):
                        continue
                    seen_labels.add(_normalize_label(gf.get('label', '')))
                    selected.append(gf)
                    used_ids.add(gf['id'])
                    if len(selected) >= num_sliders:
                        break

        print(f"[_personalized_features] User selected {len(selected_movies)} movies → "
              f"{len(selected)} personalized neurons (top user-score: "
              f"{selected[0].get('_user_score', 0):.3f} for '{selected[0].get('label', '?')}')")

        for f in selected:
            f.pop('_user_score', None)
            f.pop('_group_score', None)
            f.pop('_base_description', None)

        return selected[:num_sliders]

    except Exception as e:
        print(f"[_personalized_features] Error: {e}")
        traceback.print_exc()
        return get_sae_features(top_k=num_sliders, model_id=model_id)


def _select_slider_features(selected_movies: list, conf: dict, active_model_cfg: dict, num_sliders: int) -> list:
    algorithm = _normalize_feature_selection_algorithm(
        active_model_cfg.get("feature_selection_algorithm", conf.get("feature_selection_algorithm"))
    )
    active_sae_model_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)

    if algorithm == "global_label_topk":
        feature_source = lambda n: get_sae_features(top_k=n, model_id=active_sae_model_id)
    else:
        feature_source = lambda n: _personalized_features(
            selected_movies=selected_movies,
            model_id=active_sae_model_id,
            num_sliders=n,
        )

    if conf.get("selected_neurons"):
        features = get_neurons_by_ids(conf["selected_neurons"], model_id=active_sae_model_id)
        if len(features) < num_sliders:
            pinned_ids = {f['id'] for f in features}
            pinned_labels = {_normalize_label(f.get('label', '')) for f in features}
            extra = feature_source(num_sliders * 2)
            for ef in extra:
                if ef['id'] not in pinned_ids:
                    if _is_near_duplicate_label(ef.get('label', ''), pinned_labels):
                        continue
                    pinned_labels.add(_normalize_label(ef.get('label', '')))
                    features.append(ef)
                    if len(features) >= num_sliders:
                        break
        return features[:num_sliders]

    return feature_source(num_sliders)[:num_sliders]


@bp.route("/show-features", methods=["GET"])
def show_features():
    """Initialize session state and redirect to steering interface."""
    selected_movies = request.args.get("selectedMovies", "")
    selected_movies = selected_movies.split(",") if selected_movies else []
    selected_movies = [int(m) for m in selected_movies if m]

    session["elicitation_selected_movies"] = selected_movies
    session["iteration"] = 1
    session["cumulative_adjustments"] = {}
    session["feature_adjustments"] = {}
    session["current_phase"] = 0
    session["phase_data"] = {}
    session["iteration_preferences_approved"] = False
    session["iteration_locked_final"] = False

    return redirect(url_for(f"{__plugin_name__}.steering_interface"))


@bp.route("/next-phase", methods=["GET"])
def next_phase():
    """Transition to the next sequential phase (Model B after Model A, etc.).
    
    Flow: Phase A done → per-phase questionnaire → Phase B → per-phase questionnaire → overall questionnaire → finish.
    """
    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    if not conf:
        return redirect(url_for(f"{__plugin_name__}.finish_user_study"))

    models = conf.get("models", [])
    current_phase = session.get("current_phase", 0)
    next_phase_idx = current_phase + 1

    participation_id = session.get("participation_id")
    if participation_id:
        log_interaction(
            participation_id,
            "phase-complete",
            phase=current_phase,
            model=models[current_phase].get("name", f"Model {current_phase}") if current_phase < len(models) else "unknown",
        )

    if next_phase_idx >= len(models):
        # Last phase completed — show phase questionnaire for this phase, then finish
        phase_questionnaire_file = _get_phase_questionnaire_filename(conf, current_phase)
        if _phase_questionnaire_exists(conf, current_phase):
            session["pending_next_phase"] = None  # Signal: no more phases, go to finish
            return redirect(url_for(
                "utils.final_questionnaire",
                questionnaire_file=phase_questionnaire_file,
                continuation_url=url_for(f"{__plugin_name__}._advance_phase")
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
            continuation_url=url_for(f"{__plugin_name__}._advance_phase")
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
    session["iteration_preferences_approved"] = False
    session["iteration_locked_final"] = False
    session.pop("persistent_disliked", None)
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

    # --- All personalized features are user_activated (rotatable) ---
    for f in features:
        f["role"] = "user_activated"
    # Keep only the top NUM_SLIDERS features (no static split anymore)
    features = features[:NUM_SLIDERS]

    # --- Build semantic cluster profile (replaces old "baseline features") ---
    cluster_profile = _build_cluster_profile(model_id=active_sae_model_id) if conf.get("show_general_features", True) else []
    # Tag each dynamic feature with its cluster membership
    for f in features:
        f["cluster_id"] = _get_cluster_for_neuron(f["id"], cluster_profile)

    session["current_features"] = features
    session["cluster_profile"] = cluster_profile
    session["static_feature_ids"] = []

    max_iterations = conf.get("num_iterations", 3)
    comparison_mode = conf.get("comparison_mode", "side_by_side")
    enable_comparison = conf.get("enable_comparison", False)
    interaction_mode = conf.get("interaction_mode", "reset")
    models = conf.get("models", [])
    num_recommendations = max(1, int(conf.get("num_recommendations", 10)))

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
        loader = load_ml_dataset()

        seed_adjustments = {}
        if selected_movies:
            try:
                from .sae_recommender import get_sae_recommender
                _rec = get_sae_recommender(model_id=active_sae_model_id)
                _rec.load()
                if _rec.item_features is not None and _rec.item_ids is not None:
                    _id2idx = {int(mid): i for i, mid in enumerate(_rec.item_ids)}
                    _acts = []
                    for mid in selected_movies:
                        idx = _id2idx.get(int(mid))
                        if idx is not None:
                            a = _rec.item_features[idx]
                            if isinstance(a, _torch_init.Tensor):
                                a = a.cpu().numpy()
                            _acts.append(a)
                    if _acts:
                        _mean_act = np.mean(_acts, axis=0)
                        # Seed ALL neurons with positive mean activation,
                        # not just displayed features.  The recommendation
                        # engine uses the full weight vector, so including
                        # non-displayed neurons makes initial recs truly
                        # personalized even when displayed features have
                        # zero activation (TopK SAE sparsity).
                        displayed_ids = {f['id'] for f in features}
                        for nid in range(len(_mean_act)):
                            val = float(_mean_act[nid])
                            if val > 0:
                                seed_adjustments[str(nid)] = round(val, 4)
                        # Ensure displayed features are always in seed
                        # (for slider position, even if 0)
                        for f in features:
                            nid = f['id']
                            if str(nid) not in seed_adjustments:
                                seed_adjustments[str(nid)] = 0.0
                        top_val = max(seed_adjustments.values()) if seed_adjustments else 0
                        n_active = sum(1 for v in seed_adjustments.values() if v > 0)
                        print(f"[steering_interface] User preference projected: "
                              f"{len(selected_movies)} movies → {n_active} active neurons, "
                              f"top weight = {top_val:.3f}")
            except Exception as e:
                print(f"[steering_interface] Could not project user preferences: {e}")
                traceback.print_exc()

        # Fallback: small uniform seed so the page isn't empty
        if not seed_adjustments or max(seed_adjustments.values()) == 0:
            seed_adjustments = {str(f['id']): 0.1 for f in features}

        # Store the initial projected preferences so the UI can show
        # them as the initial slider positions.
        session["initial_seed_adjustments"] = seed_adjustments
        # Also seed the cumulative adjustments so the first "Get Recommendations"
        # click (even without any slider changes) uses the projected profile.
        session["cumulative_adjustments"] = dict(seed_adjustments)
        session["feature_adjustments"] = dict(seed_adjustments)

        if is_sequential:
            initial_recs = generate_steered_recommendations_for_model(
                loader=loader, selected_movies=selected_movies,
                feature_adjustments=seed_adjustments,
                model_config=active_model, k=num_recommendations)
        elif enable_comparison and len(models) >= 2:
            initial_recs_a = generate_steered_recommendations_for_model(
                loader=loader, selected_movies=selected_movies,
                feature_adjustments=seed_adjustments,
                model_config=models[0], k=num_recommendations)
            initial_recs_b = generate_steered_recommendations_for_model(
                loader=loader, selected_movies=selected_movies,
                feature_adjustments=seed_adjustments,
                model_config=models[1], k=num_recommendations)
        else:
            if models:
                initial_recs = generate_steered_recommendations_for_model(
                    loader=loader, selected_movies=selected_movies,
                    feature_adjustments=seed_adjustments,
                    model_config=models[0], k=num_recommendations)
            else:
                initial_recs = generate_steered_recommendations(
                    loader=loader, selected_movies=selected_movies,
                    feature_adjustments=seed_adjustments, k=num_recommendations)
    except Exception as e:
        print(f"[steering_interface] Could not generate initial recs: {e}")
        traceback.print_exc()

    # Determine steering_mode: in sequential mode each model can have its own
    if is_sequential:
        steering_mode = active_model.get("steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE))
    else:
        steering_mode = active_model_cfg.get("steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE))

    title = active_model_cfg.get("name", tr("sae_steering_title"))

    # Compute initial cluster values from the user's MOVIE GENRES,
    # not from SAE neuron weights.  Each cluster has a characteristic
    # genre profile (from its neurons' activated movies).  We compute
    # the user's genre vector from their selected movies and project
    # it onto each cluster's genre centroid (cosine similarity).
    # Then center and normalize to [-1, +1].
    initial_cluster_values = {}
    if cluster_profile and selected_movies:
        try:
            user_genre_vec = _compute_user_genre_vector(
                selected_movies, active_sae_model_id
            )
            cluster_centroids = _compute_cluster_genre_centroids(
                cluster_profile, active_sae_model_id
            )
            if user_genre_vec is not None and cluster_centroids:
                sims = {}
                for cid, centroid in cluster_centroids.items():
                    dot = sum(user_genre_vec.get(g, 0) * centroid.get(g, 0) for g in set(user_genre_vec) | set(centroid))
                    norm_u = max(sum(v**2 for v in user_genre_vec.values())**0.5, 1e-8)
                    norm_c = max(sum(v**2 for v in centroid.values())**0.5, 1e-8)
                    sims[cid] = dot / (norm_u * norm_c)
                g_mean = sum(sims.values()) / max(len(sims), 1)
                devs = {cid: s - g_mean for cid, s in sims.items()}
                max_dev = max(abs(d) for d in devs.values()) or 0.001
                initial_cluster_values = {cid: round(d / max_dev, 4) for cid, d in devs.items()}
        except Exception as e:
            print(f"[steering_interface] Could not compute cluster values: {e}")
            traceback.print_exc()

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
        "seed_adjustments": session.get("initial_seed_adjustments", {}),
        "cluster_profile": cluster_profile,
        "initial_cluster_values": initial_cluster_values,
        "show_general_features": conf.get("show_general_features", True),
        "feature_selection_algorithm": active_model_cfg.get("feature_selection_algorithm", conf.get("feature_selection_algorithm")),
        "preferences_approved": bool(session.get("iteration_preferences_approved", False)),
        "iteration_locked_final": bool(session.get("iteration_locked_final", False)),
        "num_recommendations": num_recommendations,
        "header_subtitle": _get_steering_subtitle(steering_mode),
        "header_guidance": _get_steering_guidance(steering_mode),
    }

    return render_template("steering_interface.html", **params)


def get_neurons_by_ids(neuron_ids, model_id: str = None):
    """
    Get feature info for specific neuron IDs (admin-selected or personalized).

    Uses LLM labels as the primary source (via label_neurons_by_ids_llm),
    falls back to dynamic TF-IDF labels, then static labels, then generic names.
    """
    features = []

    # --- Try LLM labels first ---
    all_labels: dict = {}  # neuron_id (int) -> label info dict
    cached_labels = _load_cached_model_labels(model_id=model_id)
    if cached_labels:
        all_labels.update({
            int(nid): cached_labels[int(nid)]
            for nid in neuron_ids
            if int(nid) in cached_labels
        })

    try:
        from .sae_recommender import get_sae_recommender
        from .llm_labeling import label_neurons_by_ids_llm

        recommender = get_sae_recommender(model_id=model_id)
        recommender.load()

        if recommender.item_features is not None and recommender.item_ids is not None:
            effective_model_id = model_id or recommender.model_id or DEFAULT_TOPK_SAE_MODEL_ID
            all_labels = label_neurons_by_ids_llm(
                neuron_ids=[int(n) for n in neuron_ids],
                model_id=effective_model_id,
                item_features=recommender.item_features,
                item_ids=recommender.item_ids,
            )
    except Exception as e:
        print(f"[get_neurons_by_ids] LLM labels unavailable, trying dynamic: {e}")
        # Fall back to dynamic TF-IDF labels
        try:
            from .sae_recommender import get_sae_recommender
            from .dynamic_labeling import label_neurons_by_ids

            recommender = get_sae_recommender(model_id=model_id)
            recommender.load()

            if recommender.item_features is not None and recommender.item_ids is not None:
                effective_model_id = model_id or recommender.model_id or DEFAULT_TOPK_SAE_MODEL_ID
                all_labels = label_neurons_by_ids(
                    neuron_ids=[int(n) for n in neuron_ids],
                    model_id=effective_model_id,
                    item_features=recommender.item_features,
                    item_ids=recommender.item_ids,
                )
        except Exception as e2:
            print(f"[get_neurons_by_ids] Dynamic labels also unavailable: {e2}")
            traceback.print_exc()

    # --- Static labels as secondary fallback ---
    static_labels = {}
    static_analysis = {}
    try:
        from pathlib import Path
        data_dir = Path(__file__).parent / "data"
        labels_path = data_dir / "neuron_labels.json"
        analysis_path = data_dir / "neuron_analysis.json"
        if labels_path.exists():
            with open(labels_path) as f:
                static_labels = json.load(f)
        if analysis_path.exists():
            with open(analysis_path) as f:
                static_analysis = json.load(f)
    except Exception:
        pass

    try:
        for neuron_id in neuron_ids:
            nid = int(neuron_id)
            nid_str = str(neuron_id)
            label = None
            description = None
            movie_count = 0

            # LLM / dynamic label (from cache or freshly computed)
            if nid in all_labels:
                dl = all_labels[nid]
                label = dl.get('label', '')
                description = dl.get('description', '')
                movie_count = dl.get('activation_count', 0)

            # Static fallback — only use for movie_count, NOT for the label.
            # Old static labels like "Comedy · Suicide Attempt" are confusing;
            # prefer showing "Neuron N{id}" until the LLM labels it properly.
            if not label:
                info = static_analysis.get(nid_str, {})
                movie_count = movie_count or info.get('activation_count', 0)

            if not label:
                label = f"Neuron N{neuron_id}"

            if not description:
                description = f"Latent concept (N{neuron_id}): {label}"

            features.append({
                'id': nid,
                'label': label,
                'category': 'latent',
                'description': description,
                'activation': 0.5,
                'movie_count': movie_count,
            })

    except Exception as e:
        print(f"[get_neurons_by_ids] Error: {e}")
        traceback.print_exc()
        features = [
            {'id': int(nid), 'label': f'Feature {nid}', 'category': 'other', 'activation': 0.5, 'movie_count': 0}
            for nid in neuron_ids
        ]

    return features


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
        client_disliked = [m for m in data.get("disliked_movies", []) if m is not None]
        client_liked = [m for m in data.get("liked_movies", []) if m is not None]
        suppressed_genres = data.get("suppressed_features", data.get("suppressed_genres", []))
        search_context = data.get("search_context", {})
        preferences_approved = bool(
            data.get("preferences_approved", session.get("iteration_preferences_approved", False))
        )

        # Expand cluster-level adjustments and grouped sliders into neuron deltas
        cluster_profile = session.get("cluster_profile", [])
        cluster_map = {c["id"]: c["neuron_ids"] for c in cluster_profile}
        current_features = session.get("current_features", [])
        feature_adjustments = _expand_feature_adjustments(
            raw_adjustments=raw_adjustments,
            current_features=current_features,
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
        models = conf.get("models", [])

        # Sequential mode overrides enable_comparison
        comparison_mode_cfg = conf.get("comparison_mode", "side_by_side")
        is_sequential_cfg = comparison_mode_cfg == "sequential" and len(models) >= 2
        if is_sequential_cfg:
            enable_comparison = False

        num_recommendations = max(1, int(conf.get("num_recommendations", 10)))

        # Determine active SAE model_id for this request
        active_model_cfg = _get_active_model_config(conf)
        active_sae_id = active_model_cfg.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
        steering_mode_for_iteration = active_model_cfg.get(
            "steering_mode", conf.get("steering_mode", DEFAULT_STEERING_MODE)
        )

        if not preferences_approved:
            return jsonify({
                "status": "error",
                "message": "Please confirm your likes/dislikes before continuing.",
                "recommendations": [],
                "recommendations_a": [],
                "recommendations_b": [],
            }), 200

        # ---- Cumulative preference accumulation ----
        # Track TWO things separately:
        #   1. cumulative_adjustments: full weight vector for the model
        #      (seed profile + all slider changes)
        #   2. user_touched_features: IDs of features the user explicitly
        #      moved sliders for (NOT seed neurons) — used by
        #      _compute_updated_sliders to decide which sliders to rotate out
        #
        # Slider amplification: the frontend sends deltas in [-1, +1] but
        # the SAE scoring needs larger magnitudes to visibly shift the
        # top-20 over a seed profile that sums to ~10.  Scale factor 3
        # means a full-range slider move (+1) adds +3 to the neuron weight,
        # enough to noticeably re-rank recommendations.
        SLIDER_AMP = 1.0
        previous_adjustments = session.get("cumulative_adjustments", {})
        user_touched = set(session.get("user_touched_features", []))
        for key, val in feature_adjustments.items():
            skey = str(key)
            prev = float(previous_adjustments.get(skey, 0))
            raw_delta = float(val)
            new = raw_delta * SLIDER_AMP
            if abs(raw_delta) > 0.001:
                previous_adjustments[skey] = round(prev + new, 4)
                user_touched.add(skey)
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

        # ---- Persistent liked / disliked (session-wide) ----
        # Both sets are EXCLUDED from future recommendations so that
        # each iteration surfaces fresh movies to evaluate.
        persistent_liked = set(session.get("persistent_liked", []))
        persistent_disliked = set(session.get("persistent_disliked", []))
        for mid in client_liked:
            if mid is not None:
                mid = int(mid)
                persistent_liked.add(mid)
                persistent_disliked.discard(mid)
        for mid in client_disliked:
            if mid is not None:
                mid = int(mid)
                persistent_disliked.add(mid)
                persistent_liked.discard(mid)
        session["persistent_liked"] = list(persistent_liked)
        session["persistent_disliked"] = list(persistent_disliked)

        # ---- Liked-movie neuron boost (equivalent weight to slider interaction) ----
        if client_liked:
            like_boost = _boost_from_liked_movies(client_liked, strength=0.30, model_id=active_sae_id)
            for nid, val in like_boost.items():
                skey = str(nid)
                # Add to model adjustments (positive only)
                if val > 0:
                    feature_adjustments[skey] = round(
                        float(feature_adjustments.get(skey, 0)) + val, 4
                    )
            # Also update cumulative so UI reflects the change
            session["cumulative_adjustments"] = {
                **session.get("cumulative_adjustments", {}),
                **{k: v for k, v in feature_adjustments.items()}
            }

        current_iteration = session.get("iteration", 1)
        participation_id = session.get("participation_id")
        if participation_id:
            log_interaction(
                participation_id,
                "feature-adjustment",
                iteration=current_iteration,
                phase=session.get("current_phase", 0),
                model_id=active_sae_id,
                steering_mode=steering_mode_for_iteration,
                adjustments=feature_adjustments,
                interaction_mode=interaction_mode,
                enable_comparison=enable_comparison,
                excluded_movies=excluded_movies_from_text,
                liked_movies=client_liked,
                disliked_movies=list(persistent_disliked),
                search_context=search_context,
                negative_adjustment_ids=[int(k) for k, v in feature_adjustments.items() if float(v) < 0],
                negative_adjustment_count=sum(1 for v in feature_adjustments.values() if float(v) < 0),
            )

        selected_movies = session.get("elicitation_selected_movies", [])
        # ORIGINAL: exclude liked/disliked from future iterations
        # excluded_movie_ids = list(set(
        #     (excluded_movies_from_text or []) +
        #     list(persistent_disliked) +
        #     list(persistent_liked)
        # ))
        # EXPERIMENT: allow re-showing items across iterations
        excluded_movie_ids = list(set(
            (excluded_movies_from_text or [])
        ))
        if excluded_movie_ids:
            session["excluded_movies_from_text"] = excluded_movie_ids

        from plugins.utils.data_loading import load_ml_dataset
        loader = load_ml_dataset()

        _seed = session.get("initial_seed_adjustments")
        
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

        if is_sequential:
            active_model = session.get("active_model_config", models[current_phase])
            all_excluded_movies = list(set(selected_movies + excluded_movie_ids))
            recommendations = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=all_excluded_movies,
                feature_adjustments=feature_adjustments,
                model_config=active_model,
                k=num_recommendations,
                suppressed_genres=suppressed_genres,
                seed_adjustments=_seed,
            )
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

        elif enable_comparison and len(models) >= 2:
            # A/B Comparison Mode - generate recommendations for both models
            model_a_config = models[0]
            model_b_config = models[1]
            
            print(f"[A/B Comparison] Model A config: {model_a_config}")
            print(f"[A/B Comparison] Model B config: {model_b_config}")
            
            # Combine selected movies with excluded movies from text
            all_excluded_movies = list(set(selected_movies + excluded_movie_ids))
            
            recommendations_a = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=all_excluded_movies,
                feature_adjustments=feature_adjustments,
                model_config=model_a_config,
                k=num_recommendations,
                suppressed_genres=suppressed_genres,
                seed_adjustments=_seed,
            )
            
            recommendations_b = generate_steered_recommendations_for_model(
                loader=loader,
                selected_movies=all_excluded_movies,
                feature_adjustments=feature_adjustments,
                model_config=model_b_config,
                k=num_recommendations,
                suppressed_genres=suppressed_genres,
                seed_adjustments=_seed,
            )
            
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
        else:
            # Single model mode
            # Combine selected movies with excluded movies from text
            all_excluded_movies = list(set(selected_movies + excluded_movie_ids))
            if models:
                recommendations = generate_steered_recommendations_for_model(
                    loader=loader,
                    selected_movies=all_excluded_movies,
                    feature_adjustments=feature_adjustments,
                    model_config=models[0],
                    k=num_recommendations,
                    suppressed_genres=suppressed_genres,
                    seed_adjustments=_seed,
                )
            else:
                recommendations = generate_steered_recommendations(
                    loader=loader,
                    selected_movies=all_excluded_movies,
                    feature_adjustments=feature_adjustments,
                    k=num_recommendations
                )
            response_data["recommendations"] = recommendations
        
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
            # Pass only USER-TOUCHED adjustments to decide which sliders
            # rotate out.  Seed-profile neurons must NOT trigger rotation.
            slider_only_adjustments = {
                k: v for k, v in feature_adjustments.items()
                if k in user_touched
            }
            updated_features = _compute_updated_sliders(
                current_features=current_features,
                cumulative_adjustments=slider_only_adjustments,
                liked_movie_ids=list(persistent_liked),
                disliked_movie_ids=list(persistent_disliked),
                model_id=active_sae_id,
                num_sliders=NUM_SLIDERS,
            )
            if updated_features and updated_features != current_features:
                session["current_features"] = updated_features
                response_data["updated_features"] = updated_features
                old_ids = {f['id'] for f in current_features}
                new_count = len([f for f in updated_features if f['id'] not in old_ids])
                print(f"[adjust_features] Sliders refreshed: {new_count} new features")

        # Cluster values: sum cumulative neuron weights per cluster,
        # then use mean-per-neuron with deviation from global mean.
        # Simpler and consistent with the fixed genre-centroid approach.
        if cluster_profile:
            cumul = session.get("cumulative_adjustments", {})
            # Which clusters are the user's cumulative adjustments pulling toward?
            # Weight each cluster by its share of the total adjustment weight.
            raw_cv = {}
            for c in cluster_profile:
                nids = c["neuron_ids"]
                if nids:
                    vals = [float(cumul.get(str(nid), 0)) for nid in nids]
                    raw_cv[c["id"]] = sum(abs(v) for v in vals) / len(nids)
                else:
                    raw_cv[c["id"]] = 0.0
            # Also factor in sign: positive cumulative = boost
            signed_cv = {}
            for c in cluster_profile:
                nids = c["neuron_ids"]
                if nids:
                    vals = [float(cumul.get(str(nid), 0)) for nid in nids]
                    signed_cv[c["id"]] = sum(vals) / len(nids)
                else:
                    signed_cv[c["id"]] = 0.0
            g_mean = sum(signed_cv.values()) / max(len(signed_cv), 1)
            devs = {cid: v - g_mean for cid, v in signed_cv.items()}
            max_dev = max(abs(d) for d in devs.values()) or 0.001
            cluster_values = {cid: round(d / max_dev, 4) for cid, d in devs.items()}
            response_data["cluster_values"] = cluster_values

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


def generate_steered_recommendations_for_model(loader, selected_movies, feature_adjustments, model_config, k=20, suppressed_genres=None, seed_adjustments=None):
    """
    Generate recommendations for a specific model configuration (for A/B testing).
    
    Args:
        loader: Movie data loader
        selected_movies: User's selected movies from elicitation (and excluded from text)
        feature_adjustments: Neuron adjustments from steering
        model_config: Dict with 'base', 'sae' keys specifying model
        k: Number of recommendations
        suppressed_genres: List of genre names to filter out completely
    
    Returns:
        List of recommendation dicts
    """
    suppressed_genres = suppressed_genres or []
    sae_model_id = model_config.get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    base_model_id = model_config.get("base", "elsa")
    
    try:
        from .sae_recommender import get_sae_recommender
        
        # Get recommender for the specific SAE model (supports A/B comparison)
        recommender = get_sae_recommender(model_id=sae_model_id)
        recommender.load()

        if recommender.item_features is None or recommender.item_ids is None:
            print(
                "[generate_steered_recommendations_for_model] SAE runtime activations "
                "missing; falling back to metadata-based recommendations"
            )
            return _fallback_genre_recommendations(loader, selected_movies, feature_adjustments, k)
        
        print(f"[generate_steered_recommendations_for_model] Using SAE model: {sae_model_id}")
        print(f"[generate_steered_recommendations_for_model] Feature adjustments: {feature_adjustments}")
        
        # Convert adjustments
        neuron_adjustments = {
            int(key): float(value) for key, value in feature_adjustments.items()
        }
        
        print(f"[generate_steered_recommendations_for_model] Neuron adjustments: {neuron_adjustments}")
        
        # All movie references (from elicitation, likes, dislikes) are movieIds.
        exclude_movie_ids = []
        for movie_ref in selected_movies:
            try:
                exclude_movie_ids.append(int(movie_ref))
            except (ValueError, TypeError):
                continue
        
        allowed_ids = set(loader.movies_df_indexed.index.tolist())

        # NOTE: Image filter disabled — it was too restrictive and reduced
        # recommendation quality.  Movies without posters will show a
        # placeholder icon in the UI.
        # if hasattr(loader, 'movie_index_to_url') and loader.movie_index_to_url:
        #     ids_with_images = set()
        #     for movie_idx in loader.movie_index_to_url:
        #         try:
        #             ids_with_images.add(loader.movie_index_to_id[movie_idx])
        #         except (KeyError, IndexError):
        #             continue
        #     if ids_with_images:
        #         allowed_ids = allowed_ids & ids_with_images
        #         print(f"[...] Restricted to {len(allowed_ids)} movies with images")

        # Request many more than needed so that filtering (unknown IDs,
        # missing metadata, suppressed genres) still leaves at least k items.
        seed_neuron_adjustments = None
        if seed_adjustments:
            seed_neuron_adjustments = {
                int(key): float(value) for key, value in seed_adjustments.items()
            }

        raw_recommendations = recommender.get_recommendations(
            feature_adjustments=neuron_adjustments,
            n_items=max(k * 15, 300),
            exclude_items=exclude_movie_ids,
            allowed_ids=allowed_ids,
            seed_adjustments=seed_neuron_adjustments,
        )
        print(f"[generate_steered_recommendations_for_model] Raw recommendations: {len(raw_recommendations)}")
        
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
        if skipped_missing_meta:
            sample_ids = skipped_missing_meta[:10]
            print(f"[generate_steered_recommendations_for_model] Skipped {len(skipped_missing_meta)} items with missing metadata, sample: {sample_ids}")
        if len(results) < k:
            print(f"[generate_steered_recommendations_for_model] WARNING: only {len(results)} results after filtering (target {k})")
        print(f"[generate_steered_recommendations_for_model] Returning {len(results)} recommendations (target {k})")
        return results[:k]
        
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
    return generate_steered_recommendations_for_model(
        loader=loader,
        selected_movies=selected_movies,
        feature_adjustments=feature_adjustments,
        model_config=default_config,
        k=k,
    )


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


@bp.route("/get-recommendations", methods=["GET"])
def get_recommendations():
    """Get recommendations based on current feature settings"""
    # TODO: Generate recommendations with steering applied
    recommendations = []
    
    return jsonify(recommendations)


def _compute_updated_sliders(
    current_features: list,
    cumulative_adjustments: dict,
    liked_movie_ids: list,
    disliked_movie_ids: list,
    model_id: str = None,
    num_sliders: int = 21,
) -> list:
    """Recompute the dynamic slider feature list after an iteration.

    User-adjusted sliders rotate OUT (their cumulative effect is baked
    into the model).  Freed slots are filled by engagement-driven
    discovery neurons (liked - disliked activation signal), then by
    globally popular features as a fallback.  Every new slider starts
    at 0 in the UI — it represents a DELTA from the current state.
    """
    import torch as _torch

    # Identify features the user explicitly touched this round
    touched_ids = set()
    for fid_str, val in cumulative_adjustments.items():
        if abs(float(val)) > 0.001:
            touched_ids.add(int(fid_str))

    try:
        from .sae_recommender import get_sae_recommender
        recommender = get_sae_recommender(model_id=model_id)
        recommender.load()

        if recommender.item_features is None or recommender.item_ids is None:
            return current_features

        MIN_NEURON_MOVIES = 20

        id_to_idx = {int(mid): i for i, mid in enumerate(recommender.item_ids)}
        decoder_vecs = _build_decoder_vecs(recommender)

        features_np = recommender.item_features
        if isinstance(features_np, _torch.Tensor):
            features_np = features_np.cpu().numpy()
        neuron_act_counts = np.sum(features_np > 0, axis=0)

        # Build engagement signal from liked / disliked movies
        liked_acts = []
        for mid in (liked_movie_ids or []):
            idx = id_to_idx.get(int(mid))
            if idx is not None:
                a = features_np[idx]
                liked_acts.append(a)
        preference_signal = np.mean(liked_acts, axis=0) if liked_acts else None

        if disliked_movie_ids and preference_signal is not None:
            disliked_acts = []
            for mid in disliked_movie_ids:
                idx = id_to_idx.get(int(mid))
                if idx is not None:
                    disliked_acts.append(features_np[idx])
            if disliked_acts:
                preference_signal = preference_signal - 0.5 * np.mean(disliked_acts, axis=0)

        neuron_scores = sorted(
            enumerate(preference_signal), key=lambda x: x[1], reverse=True
        ) if preference_signal is not None else []

        # Keep UNTOUCHED features (user didn't adjust them)
        kept = []
        used_ids = set()
        seen_labels = set()
        for f in current_features:
            if f['id'] not in touched_ids:
                f = _ensure_feature_group_metadata(f)
                kept.append(f)
                used_ids.update(f.get('member_ids', [f['id']]))
                seen_labels.add(_normalize_label(f.get('label', '')))
        used_ids.update(touched_ids)

        slots = num_sliders - len(kept)
        selected_ids = [f['id'] for f in kept]

        # Fill freed slots — engagement-based discovery neurons
        # Only consider neurons with ≥MIN_NEURON_MOVIES activations and a real label
        new_neuron_ids = []
        for nid, score in neuron_scores:
            if len(new_neuron_ids) >= slots * 4:
                break
            if nid not in used_ids and score > 0 and int(neuron_act_counts[nid]) >= MIN_NEURON_MOVIES:
                new_neuron_ids.append(int(nid))

        if new_neuron_ids:
            new_features = get_neurons_by_ids(new_neuron_ids, model_id=model_id)
            for nf in new_features:
                if len(kept) >= num_sliders:
                    break
                nf = _ensure_feature_group_metadata(nf)
                mc = nf.get('movie_count', 0)
                if mc < MIN_NEURON_MOVIES:
                    continue
                lbl = nf.get('label', '')
                if lbl.lower().startswith('feature ') or lbl.lower().startswith('neuron n') or lbl.lower().startswith('latent concept'):
                    continue
                if _is_near_duplicate_label(lbl, seen_labels):
                    continue
                if _is_cosine_duplicate(nf['id'], selected_ids, decoder_vecs):
                    continue
                nf["role"] = "user_activated"
                kept.append(nf)
                used_ids.update(nf.get('member_ids', [nf['id']]))
                seen_labels.add(_normalize_label(lbl))
                selected_ids.append(nf['id'])

        # Fallback: global popular features (same quality filters)
        if len(kept) < num_sliders:
            global_features = get_sae_features(top_k=num_sliders * 4, model_id=model_id)
            for gf in global_features:
                if len(kept) >= num_sliders:
                    break
                gf = _ensure_feature_group_metadata(gf)
                if gf['id'] in used_ids:
                    continue
                mc = gf.get('movie_count', 0)
                if mc < MIN_NEURON_MOVIES:
                    continue
                lbl = gf.get('label', '')
                if lbl.lower().startswith('feature ') or lbl.lower().startswith('neuron n') or lbl.lower().startswith('latent concept'):
                    continue
                if _is_near_duplicate_label(lbl, seen_labels):
                    continue
                if _is_cosine_duplicate(gf['id'], selected_ids, decoder_vecs):
                    continue
                gf['role'] = 'user_activated'
                kept.append(gf)
                used_ids.update(gf.get('member_ids', [gf['id']]))
                seen_labels.add(_normalize_label(lbl))
                selected_ids.append(gf['id'])

        n_rotated = sum(1 for f in current_features if f['id'] in touched_ids)
        n_new = len(kept) - (len(current_features) - n_rotated)
        print(f"[_compute_updated_sliders] {n_rotated} rotated out, {n_new} new discovery features")
        return kept[:num_sliders]

    except Exception as e:
        print(f"[_compute_updated_sliders] Error: {e}")
        traceback.print_exc()
        return current_features


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


def _build_top_features(raw_adjustments: dict, top_n: int = 7, model_id: str = None) -> list:
    """Return the top-N most impacted neurons with human-readable labels.

    Prefers LLM labels (with full description), falls back to dynamic/static.
    """
    from pathlib import Path

    data_dir = Path(__file__).parent / "data"

    label_lookup: dict = {}  # nid -> {label, description, activation_count}

    # LLM labels (best quality)
    try:
        llm_path = None
        if model_id:
            llm_path = data_dir / f"llm_labels_{model_id}_llm.json"
        if not llm_path or not llm_path.exists():
            for p in data_dir.glob("llm_labels_*_llm.json"):
                llm_path = p
                break
        if llm_path and llm_path.exists():
            with open(llm_path) as f:
                llm_raw = json.load(f)
            for nid_str, info in llm_raw.items():
                lbl = info.get('label', '')
                if lbl and not lbl.lower().startswith('feature '):
                    label_lookup[int(nid_str)] = {
                        'label': lbl,
                        'description': info.get('description', ''),
                        'movie_count': info.get('activation_count', 0),
                    }
    except Exception:
        pass

    # Dynamic labels as fallback
    if not label_lookup:
        try:
            import glob as _glob
            dyn_files = _glob.glob(str(data_dir / "dynamic_labels_*_v*.json"))
            if model_id:
                specific = data_dir / f"dynamic_labels_{model_id}_v2.json"
                if specific.exists():
                    dyn_files = [str(specific)]
            if dyn_files:
                with open(dyn_files[0]) as f:
                    dyn = json.load(f)
                for nid_str, info in dyn.items():
                    lbl = info.get('label', '')
                    if lbl:
                        label_lookup[int(nid_str)] = {
                            'label': lbl,
                            'description': info.get('description', ''),
                            'movie_count': info.get('activation_count', 0),
                        }
        except Exception:
            pass

    sorted_neurons = sorted(raw_adjustments.items(),
                            key=lambda x: abs(x[1]), reverse=True)

    results = []
    seen_labels: set = set()

    for neuron_id, weight in sorted_neurons:
        neuron_id = int(neuron_id)
        info = label_lookup.get(neuron_id)
        label = info['label'] if info else f"Feature {neuron_id}"
        if label in seen_labels:
            continue
        seen_labels.add(label)

        direction = "boost" if weight > 0 else "suppress"
        results.append({
            "neuron_id": neuron_id,
            "label": label,
            "description": info.get('description', '') if info else '',
            "movie_count": info.get('movie_count', 0) if info else 0,
            "weight": round(float(weight), 3),
            "direction": direction,
        })
        if len(results) >= top_n:
            break

    return results


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
    """Search all labeled SAE features by name (substring match).

    Returns JSON list of {id, label, description, movie_count}.
    Uses LLM labels (cached) as primary, dynamic labels as fallback.
    """
    query = request.args.get("q", "").strip().lower()
    if len(query) < 2:
        return jsonify([])

    conf = _normalize_study_config(load_user_study_config(session.get("user_study_id")))
    active_sae_id = _get_active_sae_model_id(conf)

    labels = {}

    try:
        from .sae_recommender import get_sae_recommender
        rec = get_sae_recommender(model_id=active_sae_id)
        rec.load()
        if rec.item_features is None or rec.item_ids is None:
            return jsonify([])

        # LLM labels (cached, never triggers runtime LLM)
        try:
            from .llm_labeling import get_llm_labels
            labels = dict(get_llm_labels(
                model_id=active_sae_id,
                item_features=rec.item_features,
                item_ids=rec.item_ids,
            ))
        except Exception:
            pass

        # Dynamic labels fill gaps
        try:
            from .dynamic_labeling import get_dynamic_labels
            dyn = get_dynamic_labels(
                model_id=active_sae_id,
                item_features=rec.item_features,
                item_ids=rec.item_ids,
            )
            for nid, info in dyn.items():
                nid_int = int(nid)
                if nid_int not in labels:
                    labels[nid_int] = info
        except Exception:
            pass
    except Exception as e:
        print(f"[search-features] Error: {e}")
        return jsonify([])

    current_feature_ids = {f['id'] for f in session.get("current_features", [])}

    results = []
    for nid, info in labels.items():
        nid = int(nid)
        label = info.get('label', '')
        desc = info.get('description', '')
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
                'id': nid,
                'label': label,
                'description': desc,
                'movie_count': info.get('activation_count', 0),
                'already_shown': nid in current_feature_ids,
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
    if participation_id:
        log_interaction(
            participation_id,
            "preferences-approved",
            iteration=session.get("iteration", 1),
            phase=session.get("current_phase", 0),
            model_id=active_model.get("sae", DEFAULT_TOPK_SAE_MODEL_ID),
            steering_mode=active_model.get("steering_mode", DEFAULT_STEERING_MODE),
            liked_movies=data.get("liked_movies", []),
            disliked_movies=data.get("disliked_movies", []),
        )

    return jsonify({"status": "ok", "approved": True})


# ============================================================================
# Movie Feedback (like / dislike)
# ============================================================================

@bp.route("/log-movie-feedback", methods=["POST"])
def log_movie_feedback():
    """Log a like / dislike / neutral action on a recommended movie."""
    try:
        data = request.get_json(force=True)
        movie_id = data.get("movie_id")
        action = data.get("action", "neutral")
        iteration = data.get("iteration", session.get("iteration", 1))

        participation_id = session.get("participation_id")
        if participation_id:
            log_interaction(
                participation_id,
                "movie-feedback",
                movie_id=movie_id,
                action=action,
                iteration=iteration
            )
        return jsonify({"status": "ok"})
    except Exception as e:
        print(f"[log_movie_feedback] Error: {e}")
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
            continuation_url=url_for("utils.finish")
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
        fetch_results_url=url_for(f"{__plugin_name__}.fetch_results", guid=guid)
    )


@bp.route("/fetch-results/<guid>")
def fetch_results(guid):
    """Fetch results data for analysis"""
    user_study = UserStudy.query.filter(UserStudy.guid == guid).first()
    if not user_study:
        return jsonify({"error": "Study not found"}), 404
    
    participants = Participation.query.filter(
        (Participation.time_finished != None) & 
        (Participation.user_study_id == user_study.id)
    ).all()
    
    results = {
        "study_guid": guid,
        "total_participants": len(participants),
        "participants": []
    }
    
    for p in participants:
        # Get all interactions for this participant
        interactions = Interaction.query.filter(
            Interaction.participation == p.id
        ).order_by(Interaction.time.asc()).all()
        
        participant_data = {
            "participant_id": p.id,
            "time_joined": p.time_joined.isoformat() if p.time_joined else None,
            "time_finished": p.time_finished.isoformat() if p.time_finished else None,
            "language": p.language,
            "demographics": {
                "age_group": p.age_group,
                "gender": p.gender,
                "education": p.education,
                "ml_familiar": p.ml_familiar
            },
            "feature_adjustments": [],
            "elicitation_selections": [],
            "total_iterations": 0
        }
        
        for interaction in interactions:
            data = json.loads(interaction.data) if interaction.data else {}
            
            if interaction.interaction_type == "feature-adjustment":
                participant_data["feature_adjustments"].append({
                    "iteration": data.get("iteration"),
                    "adjustments": data.get("adjustments", {}),
                    "time": interaction.time.isoformat() if interaction.time else None
                })
                participant_data["total_iterations"] = max(
                    participant_data["total_iterations"], 
                    data.get("iteration", 0)
                )
            
            elif interaction.interaction_type == "elicitation-completed":
                participant_data["elicitation_selections"] = data.get("selected_movies", [])
        
        results["participants"].append(participant_data)
    
    # Aggregate steering patterns across all participants
    all_adjustments = {}
    for p in results["participants"]:
        for adj in p["feature_adjustments"]:
            for feature, value in adj.get("adjustments", {}).items():
                if feature not in all_adjustments:
                    all_adjustments[feature] = []
                all_adjustments[feature].append(value)
    
    # Calculate average adjustments per feature
    results["aggregate_stats"] = {
        "avg_adjustments": {
            feature: sum(values) / len(values) if values else 0
            for feature, values in all_adjustments.items()
        },
        "adjustment_counts": {
            feature: len(values)
            for feature, values in all_adjustments.items()
        }
    }
    
    return jsonify(results)


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
