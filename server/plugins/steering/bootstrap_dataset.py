#!/usr/bin/env python3
"""Download the filtered MovieLens dataset from a GitHub Release and extract it.

Usage (module, from repo root so ``server`` package is importable):

    python -m server.plugins.steering.bootstrap_dataset
    python -m server.plugins.steering.bootstrap_dataset --tag v2.0
    python -m server.plugins.steering.bootstrap_dataset --asset ml-32m-filtered.zip

Env vars (all optional with sane defaults):

    DATASET_GITHUB_REPO       e.g. vaclavstibor/SAE4EasyStudy
    DATASET_RELEASE_TAG       e.g. v2.0  or  latest  (default: latest)
    ML_LATEST_DATASET_ASSET   asset filename in the release (default: ml-32m-filtered.zip)
    DATASET_DOWNLOAD_TIMEOUT  HTTP timeout in seconds (default: 300)
    GITHUB_TOKEN              bearer token for private releases

The zip is extracted under ``/app/server/static/datasets/`` so that
``ml-32m-filtered/ratings.csv`` (and siblings) appear at the path the
entrypoint validates.

When a persistent volume is mounted the parent directory may already be a
symlink to the volume — extraction works transparently in that case.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_GITHUB_REPO = os.environ.get("DATASET_GITHUB_REPO", "")
DEFAULT_RELEASE_TAG = os.environ.get("DATASET_RELEASE_TAG", "latest")
DEFAULT_ASSET_NAME = os.environ.get("ML_LATEST_DATASET_ASSET", "ml-32m-filtered.zip")
DEFAULT_TIMEOUT = int(os.environ.get("DATASET_DOWNLOAD_TIMEOUT", "300"))
DEST_DIR = Path("/app/server/static/datasets")


def _headers(token: str = "") -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "EasyStudy-dataset-bootstrap"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _fetch_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as r:
        return json.load(r)


def _release_url(repo: str, tag: str) -> str:
    repo = repo.strip("/")
    if tag == "latest":
        return f"https://api.github.com/repos/{repo}/releases/latest"
    return f"https://api.github.com/repos/{repo}/releases/tags/{urllib.parse.quote(tag, safe='')}"


def _download(url: str, dest: Path, token: str) -> None:
    req = urllib.request.Request(url, headers={**_headers(token), "Accept": "application/octet-stream"})
    print(f"  Downloading {dest.name} …", flush=True)
    total = 0
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"  {total // (1 << 20)} MB downloaded\r", end="", flush=True)
    print(f"\n  Saved {total // (1 << 20)} MB to {dest}")


def bootstrap(repo: str, tag: str, asset_name: str, token: str) -> None:
    if not repo:
        print(
            "ERROR: DATASET_GITHUB_REPO is not set.\n"
            "Set it to e.g. vaclavstibor/SAE4EasyStudy and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching release metadata: repo={repo!r} tag={tag!r}")
    release = _fetch_json(_release_url(repo, tag), token)
    resolved_tag = release.get("tag_name", tag)
    print(f"Release: {release.get('name', resolved_tag)}")

    assets_by_name = {a["name"]: a for a in (release.get("assets") or []) if a.get("name")}
    if asset_name not in assets_by_name:
        available = ", ".join(sorted(assets_by_name))
        print(
            f"ERROR: asset '{asset_name}' not found in release {resolved_tag}.\n"
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    asset = assets_by_name[asset_name]
    download_url = asset["browser_download_url"]

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DEST_DIR / asset_name
    _download(download_url, zip_path, token)

    print(f"Extracting {zip_path} → {DEST_DIR} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEST_DIR)
    zip_path.unlink()
    print("Dataset bootstrap complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download dataset from GitHub Releases")
    parser.add_argument("--repo", default=DEFAULT_GITHUB_REPO)
    parser.add_argument("--tag", default=DEFAULT_RELEASE_TAG)
    parser.add_argument("--asset", default=DEFAULT_ASSET_NAME)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = parser.parse_args()
    bootstrap(args.repo, args.tag, args.asset, args.token)
