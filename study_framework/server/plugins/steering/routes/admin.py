"""Study setup and administration endpoints."""

import traceback

from flask import jsonify, redirect, render_template, request
from flask_login import login_required

from server.platform.shared.common import get_tr

from ..constants import DEFAULT_TOPK_SAE_MODEL_ID, Modalities
from ..plugin import bp, get_lang, languages
from ..recommendation.semantic_registry import load_semantic_clusters
from ..service.initialization import long_initialization


@bp.route("/create")
@login_required
def create():
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
    return jsonify(
        [
            {
                "name": "MovieLens 32M Filtered",
                "id": "ml-32m-filtered",
                "description": "Curated MovieLens dataset (8328 movies)",
            }
        ]
    )


@bp.route("/available-sae-models")
def available_sae_models():
    from ..recommendation.sae_recommender import get_available_models

    return jsonify(get_available_models())


@bp.route("/available-steering-modes")
def available_steering_modes():
    return jsonify(
        [
            {
                "name": "No Steering (movie selection only)",
                "id": Modalities.NONE,
                "description": "No explicit steering controls, only movie feedback",
            },
            {
                "name": "Feature Sliders",
                "id": Modalities.SLIDERS,
                "description": "Continuous adjustment of feature strengths",
            },
            {
                "name": "Feature Toggles",
                "id": Modalities.TOGGLES,
                "description": "Binary on/off for features",
            },
            {
                "name": "Natural Language (Text)",
                "id": Modalities.TEXT,
                "description": (
                    "Describe preferences in natural language and map them to "
                    "feature adjustments"
                ),
            },
            {
                "name": "Hybrid (Sliders + Text)",
                "id": Modalities.HYBRID,
                "description": "Combine direct feature controls with natural-language steering",
            },
        ]
    )


@bp.route("/available-feature-selection-algorithms")
def available_feature_selection_algorithms():
    return jsonify(
        [
            {
                "id": "personalized_grouped_topk",
                "name": "Personalized grouped Top-K",
                "description": (
                    "Personalize sliders from elicitation picks and deduplicate "
                    "similar concepts."
                ),
            },
            {
                "id": "global_label_topk",
                "name": "Global label-diverse Top-K",
                "description": (
                    "Show globally strong, label-diverse features independent of "
                    "elicitation."
                ),
            },
        ]
    )


@bp.route("/available-neurons")
def available_neurons():
    model_id = request.args.get("model_id") or DEFAULT_TOPK_SAE_MODEL_ID
    if not model_id or str(model_id).strip().lower() == "none":
        model_id = DEFAULT_TOPK_SAE_MODEL_ID
    try:
        semantic_clusters = load_semantic_clusters(model_id)
        clusters = []
        for cluster in semantic_clusters["clusters"]:
            clusters.append(
                {
                    "id": cluster["cluster_id"],
                    "label": cluster["label"],
                    "category": "latent",
                    "description": cluster.get("description", ""),
                    "score": cluster["support"],
                }
            )
        clusters.sort(key=lambda row: (-row["score"], row["label"].lower()))
        return jsonify(clusters)
    except Exception as exc:
        print(f"[available_neurons] Error: {exc}")
        traceback.print_exc()
        return jsonify([])


@bp.route("/initialize", methods=["GET"])
@login_required
def initialize():
    guid = request.args.get("guid")
    long_initialization(guid)
    return redirect(request.args.get("continuation_url"))
