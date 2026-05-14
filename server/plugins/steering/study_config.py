"""Study config normalization and active-model resolution."""

from flask import session

from .constants import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_DATASET_VARIANT,
    DEFAULT_FEATURE_SELECTION_ALGORITHM,
    DEFAULT_RERANKING_STRATEGY,
    DEFAULT_SELECTION_SIGNAL_WEIGHT,
    DEFAULT_SELECTION_SIGNAL_WEIGHT_WITH_FEATURE_CONTROLS,
    DEFAULT_STEERING_MODE,
    DEFAULT_TEXT_COMPOSITION_MODE,
    DEFAULT_TOPK_SAE_MODEL_ID,
    SUPPORTED_DATASET_VARIANTS,
    SUPPORTED_FEATURE_SELECTION_ALGORITHMS,
    SUPPORTED_RERANKING_STRATEGIES,
    SUPPORTED_STEERING_MODES,
    SUPPORTED_TEXT_COMPOSITION_MODES,
    Modalities,
    get_default_models,
)
from .modalities.examples import DEFAULT_EXAMPLE_TOP_K, DEFAULT_EXAMPLE_WEIGHT
from .modalities.text import DEFAULT_TEXT_TOP_K, DEFAULT_TEXT_WEIGHT
from .modalities.toggles import DEFAULT_TOGGLE_WEIGHT


def approach_label(idx: int) -> str:
    if 0 <= idx < 26:
        return f"Approach {chr(65 + idx)}"
    return f"Approach {idx + 1}"


def normalize_steering_mode(mode: str) -> str:
    mode = (mode or DEFAULT_STEERING_MODE).strip().lower()
    if mode in SUPPORTED_STEERING_MODES:
        return mode
    return DEFAULT_STEERING_MODE


def derive_steering_mode_from_modalities(
    raw_modalities=None, fallback: str = DEFAULT_STEERING_MODE
) -> str:
    if isinstance(raw_modalities, str):
        raw_modalities = [raw_modalities]
    modalities = {
        str(item or "").strip().lower() for item in raw_modalities or [] if str(item or "").strip()
    }
    has_sliders = Modalities.SLIDERS in modalities
    has_toggles = Modalities.TOGGLES in modalities
    has_text = Modalities.TEXT in modalities
    if has_toggles:
        return Modalities.TOGGLES
    if has_sliders and has_text:
        return Modalities.HYBRID
    if has_sliders:
        return Modalities.SLIDERS
    if has_text:
        return Modalities.TEXT
    if modalities <= {Modalities.RESET, Modalities.EXAMPLES} or not modalities:
        return Modalities.NONE if raw_modalities is not None else normalize_steering_mode(fallback)
    return normalize_steering_mode(fallback)


def derive_enabled_modalities(mode: str, raw_modalities=None) -> list:
    mode = normalize_steering_mode(mode)
    if isinstance(raw_modalities, str):
        raw_modalities = [raw_modalities]
    normalized = []
    for item in raw_modalities or []:
        value = str(item or "").strip().lower()
        if value:
            normalized.append(value)
    if not normalized:
        if mode == Modalities.NONE:
            normalized = []
        elif mode == Modalities.TEXT:
            normalized = [Modalities.TEXT, Modalities.RESET]
        elif mode == Modalities.TOGGLES:
            normalized = [Modalities.TOGGLES, Modalities.RESET]
        elif mode == Modalities.HYBRID:
            normalized = [Modalities.SLIDERS, Modalities.TEXT, Modalities.RESET]
        else:
            normalized = [Modalities.SLIDERS, Modalities.RESET]
    deduped = []
    for item in normalized:
        if item == Modalities.TOGGLES and Modalities.SLIDERS in deduped:
            deduped = [existing for existing in deduped if existing != Modalities.SLIDERS]
        if item == Modalities.SLIDERS and Modalities.TOGGLES in deduped:
            deduped = [existing for existing in deduped if existing != Modalities.TOGGLES]
        if item not in deduped:
            deduped.append(item)
    return deduped


def normalize_feature_selection_algorithm(algorithm: str) -> str:
    algorithm = (algorithm or DEFAULT_FEATURE_SELECTION_ALGORITHM).strip().lower()
    if algorithm in SUPPORTED_FEATURE_SELECTION_ALGORITHMS:
        return algorithm
    return DEFAULT_FEATURE_SELECTION_ALGORITHM


def normalize_text_composition_mode(mode) -> str:
    mode = str(mode or DEFAULT_TEXT_COMPOSITION_MODE).strip().lower()
    if mode in SUPPORTED_TEXT_COMPOSITION_MODES:
        return mode
    return DEFAULT_TEXT_COMPOSITION_MODE


def normalize_reranking_strategy(strategy) -> str:
    strategy = str(strategy or DEFAULT_RERANKING_STRATEGY).strip().lower()
    if strategy in SUPPORTED_RERANKING_STRATEGIES:
        return strategy
    return DEFAULT_RERANKING_STRATEGY


def normalize_dataset_variant(dataset_id: str) -> str:
    dataset_id = (dataset_id or DEFAULT_DATASET_VARIANT).strip().lower()
    if dataset_id in SUPPORTED_DATASET_VARIANTS:
        return dataset_id
    return DEFAULT_DATASET_VARIANT


def default_selection_signal_weight(steering_mode: str) -> float:
    mode = normalize_steering_mode(steering_mode)
    if mode in {Modalities.SLIDERS, Modalities.HYBRID, Modalities.TOGGLES}:
        return DEFAULT_SELECTION_SIGNAL_WEIGHT_WITH_FEATURE_CONTROLS
    return DEFAULT_SELECTION_SIGNAL_WEIGHT


def get_study_dataset_variant(conf: dict) -> str:
    return normalize_dataset_variant((conf or {}).get("dataset"))


def normalize_study_config(conf):
    conf = dict(conf or {})
    conf["skip_participation_details"] = conf.get("skip_participation_details", True)
    conf["disable_demographics"] = conf.get("disable_demographics", True)
    conf["show_general_features"] = conf.get("show_general_features", False)
    conf["dataset"] = normalize_dataset_variant(conf.get("dataset"))
    conf["randomize_approach_order"] = bool(conf.get("randomize_approach_order", True))
    conf["text_steering_top_k"] = max(1, int(conf.get("text_steering_top_k", DEFAULT_TEXT_TOP_K)))
    conf["example_selection_top_k"] = max(
        1, int(conf.get("example_selection_top_k", DEFAULT_EXAMPLE_TOP_K))
    )
    conf["use_selected_movies_as_examples"] = bool(
        conf.get("use_selected_movies_as_examples", False)
    )
    conf["toggle_default_weight"] = float(conf.get("toggle_default_weight", DEFAULT_TOGGLE_WEIGHT))
    conf["text_steering_weight"] = float(conf.get("text_steering_weight", DEFAULT_TEXT_WEIGHT))
    conf["example_selection_weight"] = float(
        conf.get("example_selection_weight", DEFAULT_EXAMPLE_WEIGHT)
    )
    conf["selection_signal_weight"] = float(
        conf.get(
            "selection_signal_weight",
            default_selection_signal_weight(conf.get("steering_mode", DEFAULT_STEERING_MODE)),
        )
    )
    conf["feature_selection_algorithm"] = normalize_feature_selection_algorithm(
        conf.get("feature_selection_algorithm")
    )
    raw_text_cfg = conf.get("text_steering") if isinstance(conf.get("text_steering"), dict) else {}
    conf["text_steering"] = {
        "composition_mode": normalize_text_composition_mode(
            (raw_text_cfg or {}).get("composition_mode")
        ),
        "max_query_chars": int((raw_text_cfg or {}).get("max_query_chars") or 200),
    }
    conf["reranking_strategy"] = normalize_reranking_strategy(conf.get("reranking_strategy"))

    legacy_mode = normalize_steering_mode(conf.get("steering_mode", DEFAULT_STEERING_MODE))
    conf["enabled_modalities"] = derive_enabled_modalities(
        legacy_mode,
        conf.get("enabled_modalities"),
    )
    raw_models = conf.get("models") or []
    if not raw_models:
        raw_models = get_default_models()
        if not conf.get("enable_comparison", False):
            raw_models = raw_models[:1]

    models = []
    for idx, raw_model in enumerate(raw_models):
        model = dict(raw_model or {})
        model["id"] = model.get("id") or f"approach_{idx + 1}"
        model["name"] = model.get("name") or approach_label(idx)
        model["base"] = model.get("base") or DEFAULT_BASE_MODEL_ID
        model["sae"] = model.get("sae") or DEFAULT_TOPK_SAE_MODEL_ID
        raw_enabled_modalities = model.get("enabled_modalities", conf["enabled_modalities"])
        model["steering_mode"] = derive_steering_mode_from_modalities(
            raw_enabled_modalities,
            fallback=model.get("steering_mode", legacy_mode),
        )
        model["enabled_modalities"] = derive_enabled_modalities(
            model["steering_mode"],
            raw_enabled_modalities,
        )
        model["feature_selection_algorithm"] = normalize_feature_selection_algorithm(
            model.get("feature_selection_algorithm", conf["feature_selection_algorithm"])
        )
        model["text_steering_top_k"] = max(
            1,
            int(model.get("text_steering_top_k", conf["text_steering_top_k"])),
        )
        model["example_selection_top_k"] = max(
            1,
            int(model.get("example_selection_top_k", conf["example_selection_top_k"])),
        )
        model["use_selected_movies_as_examples"] = bool(
            model.get("use_selected_movies_as_examples", conf["use_selected_movies_as_examples"])
        )
        model["toggle_default_weight"] = float(
            model.get("toggle_default_weight", conf["toggle_default_weight"])
        )
        model["text_steering_weight"] = float(
            model.get("text_steering_weight", conf["text_steering_weight"])
        )
        model["example_selection_weight"] = float(
            model.get("example_selection_weight", conf["example_selection_weight"])
        )
        model["selection_signal_weight"] = float(
            model.get(
                "selection_signal_weight", default_selection_signal_weight(model["steering_mode"])
            )
        )
        model["text_composition_mode"] = normalize_text_composition_mode(
            model.get("text_composition_mode", conf["text_steering"]["composition_mode"])
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


def get_active_model_config(conf, phase_idx=None):
    from .service.participation import get_effective_models

    conf = normalize_study_config(conf)
    models = get_effective_models(conf)
    if not models:
        return {
            "id": "single",
            "name": "Approach A",
            "base": DEFAULT_BASE_MODEL_ID,
            "sae": DEFAULT_TOPK_SAE_MODEL_ID,
            "steering_mode": DEFAULT_STEERING_MODE,
            "enabled_modalities": derive_enabled_modalities(DEFAULT_STEERING_MODE),
            "feature_selection_algorithm": DEFAULT_FEATURE_SELECTION_ALGORITHM,
        }
    if phase_idx is None:
        phase_idx = session.get("current_phase", 0)
    return models[min(max(int(phase_idx), 0), len(models) - 1)]


def get_active_sae_model_id(conf, phase_idx=None):
    model_id = get_active_model_config(conf, phase_idx).get("sae", DEFAULT_TOPK_SAE_MODEL_ID)
    if not model_id or str(model_id).strip().lower() == "none":
        return DEFAULT_TOPK_SAE_MODEL_ID
    return model_id


def get_phase_questionnaire_filename(conf, phase_idx=None):
    conf = normalize_study_config(conf)
    model = get_active_model_config(conf, phase_idx)
    return model.get("phase_questionnaire_file") or conf.get("phase_questionnaire_file")


def get_steering_subtitle(steering_mode: str) -> str:
    steering_mode = normalize_steering_mode(steering_mode)
    if steering_mode == Modalities.TEXT:
        return "Describe what you want to steer your recommendations."
    if steering_mode == Modalities.HYBRID:
        return "Write text or adjust features to steer your recommendations."
    if steering_mode == Modalities.NONE:
        return "Review recommendations and select movies you would watch."
    return "Adjust features to steer your recommendations."


def get_steering_guidance(steering_mode: str) -> str:
    steering_mode = normalize_steering_mode(steering_mode)
    if steering_mode == Modalities.TEXT:
        return (
            "Start by reviewing the current recommendations, then describe the kind of "
            "change you want in your own words before getting updated recommendations."
        )
    if steering_mode == Modalities.HYBRID:
        return (
            "Start by reviewing the current recommendations, then either write what you "
            "want or adjust the discovered concepts before getting updated recommendations."
        )
    if steering_mode == Modalities.NONE:
        return (
            "Start by reviewing the current recommendations, select what you would watch, "
            "and then continue to the next recommendation update."
        )
    return (
        "Start by reviewing the current recommendations, adjust the discovered concepts "
        "below, and then get updated recommendations."
    )
