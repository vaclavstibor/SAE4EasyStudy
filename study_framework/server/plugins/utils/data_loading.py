import functools
import os
import pickle
import re
import time
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from ml_data_loader import MLDataLoader

from server.platform.shared.common import get_abs_project_root_path
from pathlib import Path


_ML_VARIANT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _resolve_safe_cache_path(ml_variant: str) -> Path:
    """Resolve the pickle cache path inside the allow-list root (server/cache/utils/<ml_variant>).

    Reject any traversal attempt by enforcing a strict variant name pattern and re-resolving
    the final path under the canonical cache root.
    """
    if not isinstance(ml_variant, str) or not _ML_VARIANT_RE.match(ml_variant):
        raise ValueError(f"Invalid dataset variant: {ml_variant!r}")
    cache_root = Path(get_abs_project_root_path()).resolve() / "cache" / "utils"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = (cache_root / ml_variant).resolve()
    if cache_root not in cache_dir.parents and cache_dir != cache_root:
        raise ValueError(f"Variant directory escapes cache root: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (cache_dir / "data_cache.pckl").resolve()
    if cache_root not in cache_path.parents:
        raise ValueError(f"Cache path escapes cache root: {cache_path}")
    return cache_path


# Loads the movielens dataset
@functools.lru_cache(maxsize=None)
def load_ml_dataset(ml_variant="ml-32m-filtered"):
    basedir = os.path.join(get_abs_project_root_path(), 'static', 'datasets')
    cache_path_obj = _resolve_safe_cache_path(ml_variant)
    cache_base_dir = str(cache_path_obj.parent)
    cache_path = str(cache_path_obj)
    if cache_path_obj.exists():
        print(f"Trying to load data cache from: {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            print(f"Cache corrupt ({exc}), rebuilding...")
            os.remove(cache_path)

    if True:
        print("Cache not available, loading everything again")
        
        ratings_path = os.path.join(basedir, f"{ml_variant}/ratings.csv")
        movies_path = os.path.join(basedir, f"{ml_variant}/movies.csv")
        tags_path = os.path.join(basedir, f"{ml_variant}/tags.csv")
        links_path = os.path.join(basedir, f"{ml_variant}/links.csv")
        plots_csv_path = os.path.join(basedir, f"{ml_variant}/plots.csv")
        if not os.path.exists(plots_csv_path):
            plots_csv_path = None
        img_dir_path = os.path.join(basedir, ml_variant, "img")
        # Ensure img dir path exists
        Path(img_dir_path).mkdir(parents=True, exist_ok=True)

        start_time = time.perf_counter()
        loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
            filters=[],
            img_dir_path=img_dir_path, plots_csv_path=plots_csv_path,
            skip_matrices=True,
        )
        loader = loader.load()
        print(f"## Loading took: {time.perf_counter() - start_time}")

        print(f"Caching the data to {cache_path}")
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(loader, f)
        os.replace(tmp_path, cache_path)

        return loader