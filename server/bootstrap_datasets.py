#!/usr/bin/env python3
"""
Download and extract the MovieLens dataset assets required by EasyStudy.

Expected release assets by default:
  - ml-32m-filtered.zip containing the filtered MovieLens CSVs and the img directory
"""

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


SERVER_DIR = Path(__file__).resolve().parent
DATASET_DIR = SERVER_DIR / "static" / "datasets" / "ml-32m-filtered"
IMG_DIR = DATASET_DIR / "img"
DEFAULT_GITHUB_REPO = os.environ.get(
    "DATASET_GITHUB_REPO",
    os.environ.get("SAE_MODEL_GITHUB_REPO", "vaclavstibor/SAE4EasyStudy"),
)
DEFAULT_RELEASE_TAG = os.environ.get(
    "DATASET_RELEASE_TAG",
    os.environ.get("SAE_MODEL_RELEASE_TAG", "latest"),
)
DEFAULT_DATASET_ASSET = os.environ.get("ML_DATASET_ASSET", "ml-32m-filtered.zip")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("DATASET_DOWNLOAD_TIMEOUT", "120"))
REQUIRED_DATASET_FILES = (
    "ratings.csv",
    "movies.csv",
    "tags.csv",
    "links.csv",
    "plots.csv",
)


def _github_headers(token: str = "") -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "EasyStudy-dataset-bootstrap",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_json(url: str, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _release_api_url(repo: str, tag: str) -> str:
    repo = repo.strip("/")
    if tag == "latest":
        return f"https://api.github.com/repos/{repo}/releases/latest"
    safe_tag = urllib.parse.quote(tag, safe="")
    return f"https://api.github.com/repos/{repo}/releases/tags/{safe_tag}"


def _select_asset(release: dict, asset_name: str) -> dict:
    assets = release.get("assets") or []
    if not assets:
        raise RuntimeError("The selected release has no uploaded assets.")
    for asset in assets:
        if asset.get("name") == asset_name:
            return asset
    available = ", ".join(sorted(asset.get("name", "") for asset in assets))
    raise RuntimeError(f"Asset '{asset_name}' was not found. Available assets: {available}")


def _download_asset(asset: dict, destination: Path, headers: dict, timeout: int) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(asset["browser_download_url"], headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(destination)


def _extract_zip(zip_path: Path, destination: Path, strip_prefix: str = "") -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            parts = list(PurePosixPath(info.filename).parts)
            if strip_prefix and parts and parts[0] == strip_prefix:
                parts = parts[1:]
            if not parts:
                continue

            target_path = destination.joinpath(*parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            resolved = target_path.resolve()
            if destination.resolve() not in (resolved, *resolved.parents):
                raise RuntimeError(f"Refusing to extract outside destination: {info.filename}")

            with archive.open(info) as src, target_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _dataset_ready() -> bool:
    return all((DATASET_DIR / name).exists() for name in REQUIRED_DATASET_FILES)


def _images_ready() -> bool:
    return IMG_DIR.exists() and any(IMG_DIR.glob("*.jpg"))


def _ensure_zip_asset(local_zip: Path, release: dict, asset_name: str, headers: dict, timeout: int) -> Path:
    if local_zip.exists() and zipfile.is_zipfile(local_zip):
        return local_zip
    asset = _select_asset(release, asset_name)
    local_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset asset: {asset_name}")
    _download_asset(asset, local_zip, headers=headers, timeout=timeout)
    if not zipfile.is_zipfile(local_zip):
        raise RuntimeError(f"Downloaded asset is not a valid zip file: {local_zip}")
    return local_zip


def _ensure_dataset_files(release: dict, headers: dict, timeout: int, dataset_asset: str) -> None:
    if _dataset_ready() and _images_ready():
        print(f"MovieLens dataset and poster images already present in {DATASET_DIR}")
        return

    local_zip = DATASET_DIR / Path(dataset_asset).name
    zip_path = _ensure_zip_asset(local_zip, release, dataset_asset, headers, timeout)
    print(f"Extracting {zip_path.name} into {DATASET_DIR}")
    _extract_zip(zip_path, DATASET_DIR, strip_prefix="ml-32m-filtered")

    if not _dataset_ready():
        missing = [name for name in REQUIRED_DATASET_FILES if not (DATASET_DIR / name).exists()]
        raise RuntimeError(f"Dataset extraction finished but required files are still missing: {missing}")
    if not _images_ready():
        raise RuntimeError(
            "Dataset extraction finished but no JPG files were found in "
            f"{IMG_DIR}. Make sure {dataset_asset} contains the img directory."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Download and extract MovieLens dataset assets.")
    parser.add_argument("--repo", default=DEFAULT_GITHUB_REPO, help="GitHub repository slug, e.g. owner/repo")
    parser.add_argument("--tag", default=DEFAULT_RELEASE_TAG, help="Release tag to fetch, or 'latest'")
    parser.add_argument(
        "--dataset-asset",
        default=DEFAULT_DATASET_ASSET,
        help="Zip asset containing ml-32m-filtered CSVs and the img directory",
    )
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""), help="Optional GitHub token")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = _github_headers(args.token)
    timeout = DEFAULT_TIMEOUT_SECONDS

    try:
        release = _fetch_json(_release_api_url(args.repo, args.tag), headers=headers, timeout=timeout)
        _ensure_dataset_files(release, headers, timeout, args.dataset_asset)
        print(f"Dataset assets ready from release: {release.get('tag_name', args.tag)}")
        return 0
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("Release or dataset asset not found. Check repo/tag/asset names.", file=sys.stderr)
        else:
            print(f"GitHub request failed with HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Dataset bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
