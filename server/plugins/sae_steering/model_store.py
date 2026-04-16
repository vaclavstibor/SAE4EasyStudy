from pathlib import Path


DEFAULT_TOPK_SAE_MODEL_ID = "TopKSAE-1024"
DEFAULT_LOCAL_MODEL_FILENAME = f"{DEFAULT_TOPK_SAE_MODEL_ID}.ckpt"
DEFAULT_RUNTIME_FEATURES_FILENAME = f"item_sae_features_{DEFAULT_TOPK_SAE_MODEL_ID}.pt"
DEFAULT_BOOTSTRAP_COMMAND = "cd server && python plugins/sae_steering/bootstrap_model.py"

PLUGIN_DIR = Path(__file__).resolve().parent
MODELS_DIR = PLUGIN_DIR / "models"
DATA_DIR = PLUGIN_DIR / "data"

REMOTE_MODEL_ASSET_CANDIDATES = (
    f"{DEFAULT_TOPK_SAE_MODEL_ID}.ckpt",
    f"{DEFAULT_TOPK_SAE_MODEL_ID}.pt",
    "model.ckpt",
    "model.pt",
)

REMOTE_RUNTIME_ASSET_CANDIDATES = (
    DEFAULT_RUNTIME_FEATURES_FILENAME,
    f"{DEFAULT_RUNTIME_FEATURES_FILENAME}.xz",
    f"{DEFAULT_RUNTIME_FEATURES_FILENAME}.gz",
    f"{DEFAULT_RUNTIME_FEATURES_FILENAME}.zip",
    "item_embeddings.pt",
    "item_embeddings.pt.xz",
    "item_embeddings.pt.gz",
    "item_sae_features.pt",
    "item_sae_features.pt.xz",
    "item_sae_features.pt.gz",
)


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def iter_local_model_paths(model_id: str = DEFAULT_TOPK_SAE_MODEL_ID):
    return (
        MODELS_DIR / f"{model_id}.ckpt",
        MODELS_DIR / f"{model_id}.pt",
    )


def find_local_model_path(model_id: str = DEFAULT_TOPK_SAE_MODEL_ID):
    for path in iter_local_model_paths(model_id):
        if path.exists():
            return path
    return None


def format_missing_model_message(model_id: str = DEFAULT_TOPK_SAE_MODEL_ID) -> str:
    expected_paths = ", ".join(str(path) for path in iter_local_model_paths(model_id))
    return (
        f"Missing SAE model '{model_id}'. Expected one of: {expected_paths}. "
        f"Download it with `{DEFAULT_BOOTSTRAP_COMMAND}`."
    )
