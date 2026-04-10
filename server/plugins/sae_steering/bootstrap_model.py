#!/usr/bin/env python3
"""
Download the SAE steering checkpoint from GitHub Releases.

Examples:
    cd server
    python plugins/sae_steering/bootstrap_model.py
    python plugins/sae_steering/bootstrap_model.py --tag v1.0.0
    python plugins/sae_steering/bootstrap_model.py --asset-name model.pkl
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from .model_store import (
        DEFAULT_LOCAL_MODEL_FILENAME,
        DEFAULT_TOPK_SAE_MODEL_ID,
        REMOTE_ASSET_CANDIDATES,
        ensure_models_dir,
    )
except ImportError:
    from model_store import (
        DEFAULT_LOCAL_MODEL_FILENAME,
        DEFAULT_TOPK_SAE_MODEL_ID,
        REMOTE_ASSET_CANDIDATES,
        ensure_models_dir,
    )


DEFAULT_GITHUB_REPO = os.environ.get("SAE_MODEL_GITHUB_REPO", "vaclavstibor/EasyStudy")
DEFAULT_RELEASE_TAG = os.environ.get("SAE_MODEL_RELEASE_TAG", "latest")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("SAE_MODEL_DOWNLOAD_TIMEOUT", "60"))


def _github_headers(token: str = "") -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "EasyStudy-SAE-bootstrap",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_json(url: str, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _fetch_text(url: str, headers: dict, timeout: int) -> str:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _release_api_url(repo: str, tag: str) -> str:
    repo = repo.strip("/")
    if tag == "latest":
        return f"https://api.github.com/repos/{repo}/releases/latest"
    safe_tag = urllib.parse.quote(tag, safe="")
    return f"https://api.github.com/repos/{repo}/releases/tags/{safe_tag}"


def _select_asset(release: dict, asset_name: str = "") -> dict:
    assets = release.get("assets") or []
    if not assets:
        raise RuntimeError("The selected release has no uploaded assets.")

    by_name = {asset.get("name"): asset for asset in assets if asset.get("name")}
    if asset_name:
        asset = by_name.get(asset_name)
        if asset:
            return asset
        raise RuntimeError(
            f"Asset '{asset_name}' was not found. Available assets: {', '.join(sorted(by_name))}"
        )

    for candidate in REMOTE_ASSET_CANDIDATES:
        asset = by_name.get(candidate)
        if asset:
            return asset

    non_checksum_assets = [asset for asset in assets if not asset.get("name", "").endswith(".sha256")]
    if len(non_checksum_assets) == 1:
        return non_checksum_assets[0]

    raise RuntimeError(
        "Could not choose a model asset automatically. "
        f"Available assets: {', '.join(sorted(by_name))}"
    )


def _extract_sha256(text: str) -> str:
    match = re.search(r"\b([a-fA-F0-9]{64})\b", text)
    return match.group(1).lower() if match else ""


def _expected_sha256(asset: dict, assets: list, headers: dict, timeout: int) -> str:
    digest = asset.get("digest")
    if isinstance(digest, str) and digest.lower().startswith("sha256:"):
        return digest.split(":", 1)[1].strip().lower()

    asset_name = asset.get("name", "")
    companion_names = []
    if asset_name:
        companion_names.append(f"{asset_name}.sha256")
        companion_names.append(f"{Path(asset_name).stem}.sha256")
    companion_names.append(f"{DEFAULT_TOPK_SAE_MODEL_ID}.sha256")
    companion_names.append(f"{DEFAULT_LOCAL_MODEL_FILENAME}.sha256")

    seen = set()
    for companion_name in companion_names:
        if companion_name in seen:
            continue
        seen.add(companion_name)
        match = next((item for item in assets if item.get("name") == companion_name), None)
        if not match:
            continue
        checksum_text = _fetch_text(match["browser_download_url"], headers=headers, timeout=timeout)
        checksum = _extract_sha256(checksum_text)
        if checksum:
            return checksum

    return ""


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_asset(asset: dict, destination: Path, headers: dict, timeout: int) -> str:
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    hasher = hashlib.sha256()
    request = urllib.request.Request(asset["browser_download_url"], headers=headers)

    with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            hasher.update(chunk)

    tmp_path.replace(destination)
    return hasher.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Download the SAE steering model from GitHub Releases.")
    parser.add_argument("--repo", default=DEFAULT_GITHUB_REPO, help="GitHub repository slug, e.g. owner/repo")
    parser.add_argument("--tag", default=DEFAULT_RELEASE_TAG, help="Release tag to fetch, or 'latest'")
    parser.add_argument(
        "--asset-name",
        default=os.environ.get("SAE_MODEL_ASSET_NAME", ""),
        help="Exact release asset name to download; defaults to auto-detection",
    )
    parser.add_argument(
        "--output",
        default="",
        help=f"Destination path; defaults to plugins/sae_steering/models/{DEFAULT_LOCAL_MODEL_FILENAME}",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the local file already exists")
    parser.add_argument(
        "--require-checksum",
        action="store_true",
        help="Fail if no SHA256 checksum is available in the release metadata/assets",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="Optional GitHub token for API access",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = _github_headers(args.token)
    timeout = DEFAULT_TIMEOUT_SECONDS

    try:
        release = _fetch_json(_release_api_url(args.repo, args.tag), headers=headers, timeout=timeout)
        asset = _select_asset(release, asset_name=args.asset_name)
        assets = release.get("assets") or []
        expected_sha = _expected_sha256(asset, assets, headers=headers, timeout=timeout)

        if args.require_checksum and not expected_sha:
            raise RuntimeError("No SHA256 checksum was found for the selected release asset.")

        output_path = Path(args.output).expanduser().resolve() if args.output else ensure_models_dir() / DEFAULT_LOCAL_MODEL_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.force:
            if expected_sha:
                actual_sha = _sha256_file(output_path)
                if actual_sha == expected_sha:
                    print(f"Model already present and verified: {output_path}")
                    print(f"Release: {release.get('tag_name', args.tag)} | Asset: {asset.get('name')}")
                    return 0
                print("Local model exists but checksum differs; downloading a fresh copy...")
            else:
                print(f"Model already present: {output_path}")
                print("Checksum unavailable, so the existing file was kept. Use --force to replace it.")
                return 0

        actual_sha = _download_asset(asset, output_path, headers=headers, timeout=timeout)
        if expected_sha and actual_sha != expected_sha:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {asset.get('name')}: expected {expected_sha}, got {actual_sha}"
            )

        print(f"Downloaded model to: {output_path}")
        print(f"Release: {release.get('tag_name', args.tag)} | Asset: {asset.get('name')}")
        print(f"SHA256: {actual_sha}")
        return 0
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("Release or asset not found. Check --repo, --tag, or --asset-name.", file=sys.stderr)
        else:
            print(f"GitHub request failed with HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Model bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
