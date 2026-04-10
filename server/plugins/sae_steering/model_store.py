from pathlib import Path


DEFAULT_TOPK_SAE_MODEL_ID = "www_TopKSAE_8192"
DEFAULT_LOCAL_MODEL_FILENAME = f"{DEFAULT_TOPK_SAE_MODEL_ID}.ckpt"
DEFAULT_BOOTSTRAP_COMMAND = "cd server && python plugins/sae_steering/bootstrap_model.py"

PLUGIN_DIR = Path(__file__).resolve().parent
MODELS_DIR = PLUGIN_DIR / "models"

REMOTE_ASSET_CANDIDATES = (
    f"{DEFAULT_TOPK_SAE_MODEL_ID}.ckpt",
    f"{DEFAULT_TOPK_SAE_MODEL_ID}.pt",
    "model.ckpt",
    "model.pt",
    "model.pkl",
)


def ensure_models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


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
