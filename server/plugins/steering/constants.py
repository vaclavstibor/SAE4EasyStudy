"""Constants and defaults for the SAE steering plugin."""

try:
    from .recommendation.model_store import DEFAULT_TOPK_SAE_MODEL_ID
except ImportError:
    from model_store import DEFAULT_TOPK_SAE_MODEL_ID


PLUGIN_NAME = "sae_steering"
PLUGIN_VERSION = "0.1.0"
PLUGIN_DESCRIPTION = "SAE-based interpretable and steerable neural recommendations"

DEFAULT_STEERING_MODE = "sliders"
DEFAULT_FEATURE_SELECTION_ALGORITHM = "personalized_grouped_topk"
DEFAULT_BASE_MODEL_ID = "elsa"
DEFAULT_DATASET_VARIANT = "ml-32m-filtered"
DEFAULT_SELECTION_SIGNAL_WEIGHT = 0.5
DEFAULT_SELECTION_SIGNAL_WEIGHT_WITH_FEATURE_CONTROLS = 0.25


class Modalities:
    SLIDERS = "sliders"
    TOGGLES = "toggles"
    TEXT = "text"
    EXAMPLES = "examples"
    RESET = "reset"
    NONE = "none"
    HYBRID = "both"


SUPPORTED_DATASET_VARIANTS = {"ml-32m-filtered"}
SUPPORTED_STEERING_MODES = {
    Modalities.SLIDERS,
    Modalities.TOGGLES,
    Modalities.TEXT,
    Modalities.HYBRID,
    Modalities.NONE,
}
SUPPORTED_FEATURE_SELECTION_ALGORITHMS = {
    "personalized_grouped_topk",
    "global_label_topk",
}

FUZZY_LABEL_JACCARD_THRESHOLD = 0.65
PROLIFIC_BASE_URL = "https://app.prolific.com/submissions/complete"

TEXT_STEERING_MAX_QUERY_CHARS = 200

DEFAULT_TEXT_COMPOSITION_MODE = "replace"
SUPPORTED_TEXT_COMPOSITION_MODES = {"replace", "add", "intersect"}

DEFAULT_RERANKING_STRATEGY = "feature-conditioned"
SUPPORTED_RERANKING_STRATEGIES = {
    # Adaptive additive blend (the current default). Adds a clamped
    # γ · sae_scores term on top of CF + genre. See sae_recommender.py
    # and docs/equations.md §10.
    "feature-conditioned",
    # Latent-space perturbation. Builds a SAE-derived direction in
    # ELSA's item-embedding space, adds α · direction to the user
    # seed, then ranks with pure CF (no additive SAE term). See
    # docs/equations.md §10.2.
    "latent-perturbation",
    # Hard-constraint filter. Keeps only candidates whose SAE score
    # is at least τ × max(positive sae score), then ranks the
    # surviving subset by base CF + genre. See docs/equations.md §10.3.
    "constrained-subset",
}

#: Default α used by the ``latent-perturbation`` strategy. The seed
#: gain ``α · direction`` is small on purpose — strong perturbations
#: easily push the seed off the CF-trained manifold and produce
#: nonsense. Researchers can override via ``conf['latent_perturbation_alpha']``.
DEFAULT_LATENT_PERTURBATION_ALPHA = 0.30

#: Default τ used by the ``constrained-subset`` strategy. Items whose
#: ``sae_score >= τ · max_positive_sae_score`` survive the filter and
#: are ranked by base CF + genre. Setting τ=0 effectively disables the
#: hard constraint (any item with non-negative SAE score passes).
DEFAULT_CONSTRAINED_SUBSET_TAU = 0.25


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
            "base": DEFAULT_BASE_MODEL_ID,
            "sae": DEFAULT_TOPK_SAE_MODEL_ID,
            "steering_mode": "none",
            "feature_selection_algorithm": DEFAULT_FEATURE_SELECTION_ALGORITHM,
        },
    ]
