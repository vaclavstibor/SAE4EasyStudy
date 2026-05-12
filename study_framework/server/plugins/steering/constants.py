"""Constants and defaults for the SAE steering plugin."""

try:
    from .recommendation.model_store import DEFAULT_BOOTSTRAP_COMMAND, DEFAULT_TOPK_SAE_MODEL_ID
except ImportError:
    from model_store import DEFAULT_BOOTSTRAP_COMMAND, DEFAULT_TOPK_SAE_MODEL_ID


PLUGIN_NAME = "sae_steering"
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "Research Team"
PLUGIN_AUTHOR_CONTACT = "research@example.com"
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
    "feature-conditioned",
}


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
