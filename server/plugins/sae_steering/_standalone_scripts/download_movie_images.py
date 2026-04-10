#!/usr/bin/env python3
"""
Download MovieLens movie poster images from TMDB API.

Usage:
    python download_movie_images.py [--output-dir PATH] [--api-key KEY] [--limit N]

This script:
  1. Reads the MovieLens ml-latest links.csv to get TMDB IDs
  2. Fetches poster URLs from the TMDB API
  3. Downloads and saves poster images

Requirements:
    pip install requests pandas tqdm

You need a TMDB API key: https://www.themoviedb.org/settings/api
Set it via --api-key or env var TMDB_API_KEY.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"  # 342px wide posters
TMDB_API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"


def get_tmdb_poster_path(tmdb_id: int, api_key: str) -> str | None:
    """Fetch poster_path from TMDB API for a given TMDB movie ID."""
    try:
        resp = requests.get(
            TMDB_API_URL.format(tmdb_id=tmdb_id),
            params={"api_key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("poster_path")
        elif resp.status_code == 429:
            # Rate limited — wait and retry
            retry_after = int(resp.headers.get("Retry-After", 2))
            time.sleep(retry_after)
            return get_tmdb_poster_path(tmdb_id, api_key)
        else:
            return None
    except Exception as e:
        print(f"  [warn] TMDB API error for {tmdb_id}: {e}")
        return None


def download_image(url: str, dest: Path) -> bool:
    """Download an image from a URL and save it to dest."""
    try:
        resp = requests.get(url, timeout=15, stream=True)
        if resp.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            return True
        return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MovieLens poster images from TMDB")
    parser.add_argument("--links-csv", type=str, default=None,
                        help="Path to MovieLens links.csv (auto-detected if omitted)")
    parser.add_argument("--output-dir", type=str, default="images",
                        help="Directory to save downloaded images (default: images)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="TMDB API key (or set TMDB_API_KEY env var)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Download at most N images (0 = all)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip images that already exist on disk (default: True)")
    parser.add_argument("--manifest", type=str, default="image_manifest.json",
                        help="JSON manifest mapping movieId -> image filename")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("ERROR: TMDB API key required. Pass --api-key or set TMDB_API_KEY env var.")
        print("Get one at: https://www.themoviedb.org/settings/api")
        sys.exit(1)

    # Find links.csv
    if args.links_csv:
        links_path = Path(args.links_csv)
    else:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent / "data" / "ml-latest" / "links.csv",
            Path("data/ml-latest/links.csv"),
            Path("server/static/datasets/ml-latest/links.csv"),
            Path("../../static/datasets/ml-latest/links.csv"),
        ]
        links_path = None
        for c in candidates:
            if c.exists():
                links_path = c
                break
        if links_path is None:
            # Try via cache/utils
            cache_links = Path(__file__).parent.parent.parent / "cache" / "utils" / "ml-latest" / "links.csv"
            if cache_links.exists():
                links_path = cache_links

    if links_path is None or not links_path.exists():
        print(f"ERROR: Could not find links.csv. Pass --links-csv explicitly.")
        sys.exit(1)

    print(f"Reading {links_path}")
    links_df = pd.read_csv(links_path)
    # links.csv has: movieId, imdbId, tmdbId
    links_df = links_df.dropna(subset=["tmdbId"])
    links_df["tmdbId"] = links_df["tmdbId"].astype(int)
    print(f"Found {len(links_df)} movies with TMDB IDs")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    manifest_path = output_dir / args.manifest
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"Loaded existing manifest with {len(manifest)} entries")

    rows = links_df.to_dict("records")
    if args.limit > 0:
        rows = rows[:args.limit]

    downloaded = 0
    skipped = 0
    failed = 0

    for row in tqdm(rows, desc="Downloading posters"):
        movie_id = int(row["movieId"])
        tmdb_id = int(row["tmdbId"])
        filename = f"{movie_id}.jpg"
        dest = output_dir / filename

        if args.skip_existing and dest.exists():
            if str(movie_id) not in manifest:
                manifest[str(movie_id)] = filename
            skipped += 1
            continue

        poster_path = get_tmdb_poster_path(tmdb_id, api_key)
        if not poster_path:
            failed += 1
            continue

        url = f"{TMDB_IMAGE_BASE}{poster_path}"
        if download_image(url, dest):
            manifest[str(movie_id)] = filename
            downloaded += 1
        else:
            failed += 1

        # Respect TMDB rate limit (~40 req/10s)
        time.sleep(0.25)

        # Save manifest periodically
        if downloaded % 100 == 0:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f)

    # Final manifest save
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
