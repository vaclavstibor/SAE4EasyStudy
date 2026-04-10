#!/usr/bin/env python3
"""
Download the SAE steering checkpoint and runtime features from GitHub Releases.

Examples:
    cd server
    python plugins/sae_steering/bootstrap_model.py
    python plugins/sae_steering/bootstrap_model.py --tag v1.0.0
    python plugins/sae_steering/bootstrap_model.py --model-asset-name model.pkl
"""

import argparse
import gzip
import hashlib
import json
import lzma
import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

try:
    from .model_store import (
        DEFAULT_LOCAL_MODEL_FILENAME,
        DEFAULT_RUNTIME_FEATURES_FILENAME,
        DEFAULT_TOPK_SAE_MODEL_ID,
        REMOTE_MODEL_ASSET_CANDIDATES,
        REMOTE_RUNTIME_ASSET_CANDIDATES,
        ensure_data_dir,
        ensure_models_dir,
    )
except ImportError:
    from model_store import (
        DEFAULT_LOCAL_MODEL_FILENAME,
        DEFAULT_RUNTIME_FEATURES_FILENAME,
        DEFAULT_TOPK_SAE_MODEL_ID,
        REMOTE_MODEL_ASSET_CANDIDATES,
        REMOTE_RUNTIME_ASSET_CANDIDATES,
        ensure_data_dir,
        ensure_models_dir,
    )


DEFAULT_GITHUB_REPO = os.environ.get("SAE_MODEL_GITHUB_REPO", "vaclavstibor/SAE4EasyStudy")
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


def _select_asset(release: dict, asset_name: str, candidates: tuple[str, ...], purpose: str) -> dict:
    assets = release.get("assets") or []
    if not assets:
        raise RuntimeError("The selected release has no uploaded assets.")

    by_name = {asset.get("name"): asset for asset in assets if asset.get("name")}
    if asset_name:
        asset = by_name.get(asset_name)
        if asset:
            return asset
        raise RuntimeError(
            f"{purpose}: asset '{asset_name}' was not found. Available assets: {', '.join(sorted(by_name))}"
        )

    for candidate in candidates:
        asset = by_name.get(candidate)
        if asset:
            return asset

    raise RuntimeError(
        f"{purpose}: could not choose an asset automatically. "
        f"Tried: {', '.join(candidates)}. Available assets: {', '.join(sorted(by_name))}"
    )


def _extract_sha256(text: str) -> str:
    match = re.search(r"\b([a-fA-F0-9]{64})\b", text)
    return match.group(1).lower() if match else ""


def _expected_sha256(asset: dict, assets: list, headers: dict, timeout: int, aliases: tuple[str, ...] = ()) -> str:
    digest = asset.get("digest")
    if isinstance(digest, str) and digest.lower().startswith("sha256:"):
        return digest.split(":", 1)[1].strip().lower()

    asset_name = asset.get("name", "")
    companion_names = []
    if asset_name:
        companion_names.append(f"{asset_name}.sha256")
        companion_names.append(f"{Path(asset_name).stem}.sha256")
    companion_names.extend(name for name in aliases if name)

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


def _is_compressed_asset(name: str) -> bool:
    return name.endswith((".xz", ".gz", ".zip"))


def _extract_downloaded_asset(download_path: Path, output_path: Path, asset_name: str) -> None:
    if asset_name.endswith(".xz"):
        with lzma.open(download_path, "rb") as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return
    if asset_name.endswith(".gz"):
        with gzip.open(download_path, "rb") as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return
    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(download_path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if len(members) != 1:
                raise RuntimeError(
                    f"Runtime SAE features zip must contain exactly one file, found {len(members)}."
                )
            with archive.open(members[0]) as src, output_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return
    shutil.move(str(download_path), str(output_path))


def _ensure_asset(
    *,
    release: dict,
    assets: list,
    headers: dict,
    timeout: int,
    purpose: str,
    asset_name: str,
    candidates: tuple[str, ...],
    output_path: Path,
    force: bool,
    require_checksum: bool,
    checksum_aliases: tuple[str, ...] = (),
) -> None:
    asset = _select_asset(release, asset_name=asset_name, candidates=candidates, purpose=purpose)
    expected_sha = _expected_sha256(
        asset,
        assets,
        headers=headers,
        timeout=timeout,
        aliases=checksum_aliases,
    )

    if require_checksum and not expected_sha:
        raise RuntimeError(f"{purpose}: no SHA256 checksum was found for asset '{asset.get('name')}'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        if expected_sha and not _is_compressed_asset(asset.get("name", "")):
            actual_sha = _sha256_file(output_path)
            if actual_sha == expected_sha:
                print(f"{purpose}: already present and verified at {output_path}")
                return
            print(f"{purpose}: local file exists but checksum differs; downloading a fresh copy...")
        elif expected_sha and _is_compressed_asset(asset.get("name", "")):
            print(f"{purpose}: already present at {output_path}")
            print("Compressed release asset cannot be re-verified after extraction. Use --force to replace it.")
            return
        else:
            print(f"{purpose}: already present at {output_path}")
            print("Checksum unavailable, so the existing file was kept. Use --force to replace it.")
            return

    download_path = output_path
    if _is_compressed_asset(asset.get("name", "")):
        download_path = output_path.with_name(f"{output_path.name}{Path(asset['name']).suffix}.download")

    actual_sha = _download_asset(asset, download_path, headers=headers, timeout=timeout)
    if expected_sha and actual_sha != expected_sha:
        download_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{purpose}: SHA256 mismatch for {asset.get('name')}: expected {expected_sha}, got {actual_sha}"
        )

    if download_path != output_path:
        output_path.unlink(missing_ok=True)
        _extract_downloaded_asset(download_path, output_path, asset.get("name", ""))
        download_path.unlink(missing_ok=True)

    print(f"{purpose}: downloaded {asset.get('name')} -> {output_path}")
    print(f"{purpose}: SHA256 {actual_sha}")


def _optional_asset_download(
    *,
    release: dict,
    assets: list,
    headers: dict,
    timeout: int,
    purpose: str,
    asset_name: str,
    candidates: tuple[str, ...],
    output_path: Path,
    force: bool,
    require_checksum: bool,
    checksum_aliases: tuple[str, ...] = (),
    optional: bool = False,
) -> None:
    try:
        _ensure_asset(
            release=release,
            assets=assets,
            headers=headers,
            timeout=timeout,
            purpose=purpose,
            asset_name=asset_name,
            candidates=candidates,
            output_path=output_path,
            force=force,
            require_checksum=require_checksum,
            checksum_aliases=checksum_aliases,
        )
    except RuntimeError as exc:
        if optional:
            print(f"[bootstrap_model] Optional asset missing: {purpose}: {exc}")
            return
        raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the SAE steering model checkpoint and runtime features from GitHub Releases."
    )
    parser.add_argument("--repo", default=DEFAULT_GITHUB_REPO, help="GitHub repository slug, e.g. owner/repo")
    parser.add_argument("--tag", default=DEFAULT_RELEASE_TAG, help="Release tag to fetch, or 'latest'")
    parser.add_argument(
        "--model-asset-name",
        "--asset-name",
        dest="model_asset_name",
        default=os.environ.get("SAE_MODEL_ASSET_NAME", ""),
        help="Exact model checkpoint asset name; defaults to auto-detection",
    )
    parser.add_argument(
        "--model-output",
        "--output",
        dest="model_output",
        default="",
        help=f"Checkpoint destination; defaults to plugins/sae_steering/models/{DEFAULT_LOCAL_MODEL_FILENAME}",
    )
    parser.add_argument(
        "--runtime-asset-name",
        dest="runtime_asset_name",
        default=os.environ.get("SAE_RUNTIME_ASSET_NAME", DEFAULT_RUNTIME_FEATURES_FILENAME),
        help="Exact runtime features asset name; defaults to item_sae_features_<model>.pt",
    )
    parser.add_argument(
        "--runtime-output",
        dest="runtime_output",
        default="",
        help=f"Runtime features destination; defaults to plugins/sae_steering/data/{DEFAULT_RUNTIME_FEATURES_FILENAME}",
    )
    parser.add_argument(
        "--label-asset-name",
        dest="label_asset_name",
        default=os.environ.get("SAE_LABEL_ASSET_NAME", ""),
        help="Exact LLM label cache asset name; defaults to auto-detection",
    )
    parser.add_argument(
        "--label-output",
        dest="label_output",
        default="",
        help="Label cache destination; defaults to plugins/sae_steering/data/llm_labels_<model>_llm.json",
    )
    parser.add_argument(
        "--label-optional",
        action="store_true",
        help="Do not fail if the label cache asset is missing",
    )
    parser.add_argument(
        "--skip-runtime-features",
        action="store_true",
        help="Download only the checkpoint and skip the precomputed item_sae_features asset",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if local files already exist")
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
        assets = release.get("assets") or []

        model_output = (
            Path(args.model_output).expanduser().resolve()
            if args.model_output
            else ensure_models_dir() / DEFAULT_LOCAL_MODEL_FILENAME
        )
        _ensure_asset(
            release=release,
            assets=assets,
            headers=headers,
            timeout=timeout,
            purpose="Model checkpoint",
            asset_name=args.model_asset_name,
            candidates=REMOTE_MODEL_ASSET_CANDIDATES,
            output_path=model_output,
            force=args.force,
            require_checksum=args.require_checksum,
            checksum_aliases=(f"{DEFAULT_TOPK_SAE_MODEL_ID}.sha256", f"{DEFAULT_LOCAL_MODEL_FILENAME}.sha256"),
        )

        if not args.skip_runtime_features:
            runtime_output = (
                Path(args.runtime_output).expanduser().resolve()
                if args.runtime_output
                else ensure_data_dir() / DEFAULT_RUNTIME_FEATURES_FILENAME
            )
            _ensure_asset(
                release=release,
                assets=assets,
                headers=headers,
                timeout=timeout,
                purpose="Runtime SAE features",
                asset_name=args.runtime_asset_name,
                candidates=REMOTE_RUNTIME_ASSET_CANDIDATES,
                output_path=runtime_output,
                force=args.force,
                require_checksum=args.require_checksum,
                checksum_aliases=(
                    f"{DEFAULT_RUNTIME_FEATURES_FILENAME}.sha256",
                    f"{Path(DEFAULT_RUNTIME_FEATURES_FILENAME).stem}.sha256",
                ),
            )

        label_output = (
            Path(args.label_output).expanduser().resolve()
            if args.label_output
            else ensure_data_dir() / f"llm_labels_{DEFAULT_TOPK_SAE_MODEL_ID}_llm.json"
        )
        label_candidates = (
            f"llm_labels_{DEFAULT_TOPK_SAE_MODEL_ID}_llm.json",
            "llm_labels_llm.json",
        )
        _optional_asset_download(
            release=release,
            assets=assets,
            headers=headers,
            timeout=timeout,
            purpose="LLM label cache",
            asset_name=args.label_asset_name,
            candidates=label_candidates,
            output_path=label_output,
            force=args.force,
            require_checksum=args.require_checksum,
            optional=args.label_optional,
        )

        print(f"Release assets ready from release: {release.get('tag_name', args.tag)}")
        return 0
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print("Release or asset not found. Check repo, tag, or asset names.", file=sys.stderr)
        else:
            print(f"GitHub request failed with HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Model bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
