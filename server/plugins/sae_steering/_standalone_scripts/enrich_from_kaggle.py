"""
Enrich MovieLens data with TMDB metadata from Kaggle dataset.

Uses Polars for fast CSV processing.

Required files in static/datasets/kaggle-tmdb/:
- credits.csv (cast & crew)
- keywords.csv (plot keywords)
- movies_metadata.csv (optional - for extra metadata)

Usage:
    python enrich_from_kaggle.py
"""

import ast
import json
import pickle
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = Path(__file__).parent.parent.parent / "static" / "datasets"
ML_LATEST_DIR = DATASET_DIR / "ml-latest"
KAGGLE_DIR = DATASET_DIR / "kaggle-tmdb"

# Output files
ENRICHED_TAGS_FILE = DATA_DIR / "enriched_tags.pkl"
TMDB_DATA_FILE = ML_LATEST_DIR / "tmdb_data.json"


def safe_literal_eval(val):
    """Safely parse stringified JSON/Python literals."""
    if val is None or val == "":
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def load_movielens_data() -> Tuple[pl.DataFrame, Dict]:
    """Load MovieLens movies and links."""
    print("1. Loading MovieLens data...")
    
    # Movies
    print("   Loading movies.csv...", flush=True)
    movies_df = pl.read_csv(ML_LATEST_DIR / "movies.csv")
    print(f"   ✓ {len(movies_df):,} movies")
    
    # Links (contains tmdbId)
    print("   Loading links.csv...", flush=True)
    links_df = pl.read_csv(ML_LATEST_DIR / "links.csv")
    print(f"   ✓ {len(links_df):,} links")
    
    # Merge
    print("   Merging...", flush=True)
    movies_df = movies_df.join(
        links_df.select(["movieId", "tmdbId"]),
        on="movieId",
        how="left"
    )
    print(f"   ✓ Merged")
    
    # Load item2index
    item2index_path = DATA_DIR / "item2index.pkl"
    if not item2index_path.exists():
        raise FileNotFoundError(f"item2index.pkl not found at {item2index_path}")
    
    print("   Loading item2index.pkl...", flush=True)
    with open(item2index_path, "rb") as f:
        item2index = pickle.load(f)
    print(f"   ✓ {len(item2index):,} items in SAE index")
    
    return movies_df, item2index


def load_kaggle_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load Kaggle TMDB dataset."""
    print("\n2. Loading Kaggle TMDB data...")
    
    if not KAGGLE_DIR.exists():
        print(f"\n   ❌ ERROR: Kaggle dataset not found!")
        print(f"   Expected location: {KAGGLE_DIR}")
        print("\n   Download from: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
        raise FileNotFoundError(f"Kaggle dataset not found at {KAGGLE_DIR}")
    
    credits_df = None
    keywords_df = None
    metadata_df = None
    
    # Credits (cast & crew)
    credits_path = KAGGLE_DIR / "credits.csv"
    if credits_path.exists():
        print("   Loading credits.csv...", flush=True)
        credits_df = pl.read_csv(credits_path)
        print(f"   ✓ {len(credits_df):,} movies with credits")
    else:
        print("   ⚠ credits.csv not found")
    
    # Keywords
    keywords_path = KAGGLE_DIR / "keywords.csv"
    if keywords_path.exists():
        print("   Loading keywords.csv...", flush=True)
        keywords_df = pl.read_csv(keywords_path)
        print(f"   ✓ {len(keywords_df):,} movies with keywords")
    else:
        print("   ⚠ keywords.csv not found")
    
    # Metadata (optional)
    metadata_path = KAGGLE_DIR / "movies_metadata.csv"
    if metadata_path.exists():
        print("   Loading movies_metadata.csv...", flush=True)
        # Read with infer_schema_length to handle mixed types
        metadata_df = pl.read_csv(
            metadata_path,
            infer_schema_length=10000,
            ignore_errors=True
        )
        # Filter to valid numeric IDs - handle both string and int types
        id_col = metadata_df["id"]
        if id_col.dtype == pl.Utf8:
            metadata_df = metadata_df.filter(
                pl.col("id").str.contains(r"^\d+$")
            ).with_columns(
                pl.col("id").cast(pl.Int64)
            )
        else:
            # Already numeric, just filter out nulls
            metadata_df = metadata_df.filter(pl.col("id").is_not_null())
        print(f"   ✓ {len(metadata_df):,} movies with metadata")
    else:
        print("   ⚠ movies_metadata.csv not found")
    
    return credits_df, keywords_df, metadata_df


def load_existing_descriptions() -> Dict[str, str]:
    """Load plot descriptions from descriptions.json."""
    desc_file = ML_LATEST_DIR / "descriptions.json"
    if desc_file.exists():
        print("\n3. Loading existing descriptions...")
        with open(desc_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        plots = {mid: info.get("plot", "") for mid, info in data.items() if info.get("plot")}
        print(f"   ✓ {len(plots):,} plot descriptions")
        return plots
    return {}


def build_tmdb_cache(
    movies_df: pl.DataFrame,
    credits_df: pl.DataFrame,
    keywords_df: pl.DataFrame,
    metadata_df: pl.DataFrame,
    plots: Dict[str, str]
) -> Dict[str, Dict]:
    """Build TMDB cache from Kaggle data."""
    print("\n4. Building TMDB data cache...")
    
    # Convert to dictionaries for fast lookup
    credits_by_id = {}
    if credits_df is not None:
        print("   Indexing credits...", flush=True)
        for row in credits_df.iter_rows(named=True):
            credits_by_id[row['id']] = row
        print(f"   ✓ {len(credits_by_id):,} indexed")
    
    keywords_by_id = {}
    if keywords_df is not None:
        print("   Indexing keywords...", flush=True)
        for row in keywords_df.iter_rows(named=True):
            keywords_by_id[row['id']] = row
        print(f"   ✓ {len(keywords_by_id):,} indexed")
    
    metadata_by_id = {}
    if metadata_df is not None:
        print("   Indexing metadata...", flush=True)
        for row in metadata_df.iter_rows(named=True):
            metadata_by_id[row['id']] = row
        print(f"   ✓ {len(metadata_by_id):,} indexed")
    
    # Process movies
    tmdb_cache = {}
    print("   Processing movies...", flush=True)
    
    for row in tqdm(movies_df.iter_rows(named=True), total=len(movies_df), desc="   Building cache"):
        movie_id = row['movieId']
        tmdb_id = row.get('tmdbId')
        
        if tmdb_id is None:
            continue
        
        tmdb_id = int(tmdb_id)
        cache_key = str(tmdb_id)
        
        movie_data = {
            "tmdb_id": tmdb_id,
            "movieId": movie_id,
            "title": row.get('title', ''),
            "genres_ml": row.get('genres', ''),
            "cast": [],
            "directors": [],
            "writers": [],
            "keywords": [],
            "genres": [],
            "overview": plots.get(str(movie_id), ""),
        }
        
        # Add credits
        if tmdb_id in credits_by_id:
            credit_row = credits_by_id[tmdb_id]
            
            # Parse cast
            cast_data = safe_literal_eval(credit_row.get('cast', '[]'))
            for person in cast_data[:15]:
                movie_data["cast"].append({
                    "name": person.get("name", ""),
                    "character": person.get("character", ""),
                    "order": person.get("order", 99),
                    "gender": person.get("gender", 0),
                })
            
            # Parse crew
            crew_data = safe_literal_eval(credit_row.get('crew', '[]'))
            for person in crew_data:
                job = person.get("job", "")
                name = person.get("name", "")
                if job == "Director":
                    movie_data["directors"].append(name)
                elif job in ["Writer", "Screenplay", "Story"]:
                    if name not in movie_data["writers"]:
                        movie_data["writers"].append(name)
        
        # Add keywords
        if tmdb_id in keywords_by_id:
            kw_row = keywords_by_id[tmdb_id]
            kw_data = safe_literal_eval(kw_row.get('keywords', '[]'))
            movie_data["keywords"] = [k.get("name", "") for k in kw_data]
        
        # Add metadata
        if tmdb_id in metadata_by_id:
            meta_row = metadata_by_id[tmdb_id]
            
            # Safe conversion for numeric fields
            def safe_num(val, default=0):
                if val is None:
                    return default
                try:
                    return float(val) if val else default
                except (ValueError, TypeError):
                    return default
            
            movie_data["budget"] = safe_num(meta_row.get("budget"))
            movie_data["revenue"] = safe_num(meta_row.get("revenue"))
            movie_data["runtime"] = safe_num(meta_row.get("runtime"))
            movie_data["vote_average"] = safe_num(meta_row.get("vote_average"))
            movie_data["vote_count"] = safe_num(meta_row.get("vote_count"))
            movie_data["popularity"] = safe_num(meta_row.get("popularity"))
            movie_data["release_date"] = meta_row.get("release_date", "") or ""
            movie_data["original_language"] = meta_row.get("original_language", "") or ""
            
            # Parse genres
            genres_data = safe_literal_eval(meta_row.get('genres', '[]'))
            movie_data["genres"] = [g.get("name", "") for g in genres_data if isinstance(g, dict)]
            
            # Parse production companies
            companies_data = safe_literal_eval(meta_row.get('production_companies', '[]'))
            movie_data["production_companies"] = [
                c.get("name", "") for c in companies_data[:5] if isinstance(c, dict)
            ]
            
            # Parse production countries
            countries_data = safe_literal_eval(meta_row.get('production_countries', '[]'))
            movie_data["production_countries"] = [
                c.get("name", "") for c in countries_data if isinstance(c, dict)
            ]
            
            # Collection
            collection_str = meta_row.get('belongs_to_collection', '')
            if collection_str:
                collection_data = safe_literal_eval(collection_str)
                if isinstance(collection_data, dict):
                    movie_data["belongs_to_collection"] = collection_data.get("name", "")
        
        tmdb_cache[cache_key] = movie_data
    
    print(f"   ✓ Built cache for {len(tmdb_cache):,} movies")
    return tmdb_cache


def create_enriched_tags(movies_df: pl.DataFrame, item2index: Dict, tmdb_cache: Dict) -> Dict[str, List[int]]:
    """Create enriched tag-to-items mapping."""
    print("\n5. Creating enriched tags...")
    
    enriched_tags = defaultdict(list)
    
    for row in tqdm(movies_df.iter_rows(named=True), total=len(movies_df), desc="   Processing"):
        movie_id = row['movieId']
        tmdb_id = row.get('tmdbId')
        
        if movie_id not in item2index:
            continue
        
        item_idx = item2index[movie_id]
        
        # Genre tags from MovieLens
        genres = row.get('genres', '')
        if genres:
            for genre in genres.split('|'):
                genre = genre.strip().lower()
                if genre and genre != "(no genres listed)":
                    enriched_tags[f"genre:{genre}"].append(item_idx)
        
        # TMDB enriched data
        if tmdb_id is not None:
            tmdb_data = tmdb_cache.get(str(int(tmdb_id)))
            
            if tmdb_data:
                # Actor tags
                for actor in tmdb_data.get("cast", []):
                    name = actor["name"].lower()
                    order = actor.get("order", 99)
                    
                    if order < 3:
                        enriched_tags[f"actor:{name}"].append(item_idx)
                        enriched_tags[f"lead:{name}"].append(item_idx)
                    elif order < 15:
                        enriched_tags[f"actor:{name}"].append(item_idx)
                
                # Director tags
                for director in tmdb_data.get("directors", []):
                    enriched_tags[f"director:{director.lower()}"].append(item_idx)
                
                # Writer tags
                for writer in tmdb_data.get("writers", [])[:3]:
                    enriched_tags[f"writer:{writer.lower()}"].append(item_idx)
                
                # Keyword tags
                for keyword in tmdb_data.get("keywords", []):
                    if keyword:
                        enriched_tags[f"keyword:{keyword.lower()}"].append(item_idx)
                
                # Studio tags
                for company in tmdb_data.get("production_companies", [])[:3]:
                    if company:
                        enriched_tags[f"studio:{company.lower()}"].append(item_idx)
                
                # Collection tags
                collection = tmdb_data.get("belongs_to_collection", "")
                if collection:
                    enriched_tags[f"collection:{collection.lower()}"].append(item_idx)
                
                # Country tags
                for country in tmdb_data.get("production_countries", []):
                    if country:
                        enriched_tags[f"country:{country.lower()}"].append(item_idx)
                
                # Decade tags
                release_date = tmdb_data.get("release_date", "")
                if release_date and len(release_date) >= 4:
                    year = release_date[:4]
                    if year.isdigit():
                        decade = (int(year) // 10) * 10
                        enriched_tags[f"decade:{decade}s"].append(item_idx)
                        if int(year) >= 2000:
                            enriched_tags[f"year:{year}"].append(item_idx)
                
                # Rating tags
                rating = tmdb_data.get("vote_average", 0) or 0
                vote_count = tmdb_data.get("vote_count", 0) or 0
                if rating >= 8.0 and vote_count > 1000:
                    enriched_tags["rating:highly rated"].append(item_idx)
                elif rating >= 7.0 and vote_count > 500:
                    enriched_tags["rating:well received"].append(item_idx)
                
                # Runtime tags
                runtime = tmdb_data.get("runtime", 0) or 0
                if runtime > 180:
                    enriched_tags["length:epic"].append(item_idx)
                elif runtime > 150:
                    enriched_tags["length:long"].append(item_idx)
                elif runtime > 90:
                    enriched_tags["length:standard"].append(item_idx)
                elif runtime > 0:
                    enriched_tags["length:short"].append(item_idx)
                
                # Budget tags
                budget = tmdb_data.get("budget", 0) or 0
                if budget > 100_000_000:
                    enriched_tags["budget:blockbuster"].append(item_idx)
                elif budget > 30_000_000:
                    enriched_tags["budget:high budget"].append(item_idx)
                elif budget > 0:
                    enriched_tags["budget:indie"].append(item_idx)
    
    # Filter minimum occurrences
    min_occurrences = 5
    filtered_tags = {
        tag: items for tag, items in enriched_tags.items()
        if len(items) >= min_occurrences
    }
    
    print(f"\n   Enriched tags created:")
    print(f"   Total unique tags: {len(filtered_tags):,}")
    
    # Count by category
    categories = defaultdict(int)
    for tag in filtered_tags.keys():
        category = tag.split(":")[0]
        categories[category] += 1
    
    for cat, count in sorted(categories.items()):
        print(f"   {cat}: {count:,} tags")
    
    return filtered_tags


def main():
    """Main enrichment pipeline."""
    print("=" * 60)
    print("MOVIE DATA ENRICHMENT (Kaggle + Polars)")
    print("=" * 60)
    
    # Load MovieLens
    movies_df, item2index = load_movielens_data()
    
    # Load Kaggle data
    credits_df, keywords_df, metadata_df = load_kaggle_data()
    
    # Load existing descriptions
    plots = load_existing_descriptions()
    
    # Build TMDB cache
    tmdb_cache = build_tmdb_cache(movies_df, credits_df, keywords_df, metadata_df, plots)
    
    # Save TMDB cache
    print(f"\n   Saving TMDB cache to {TMDB_DATA_FILE.name}...")
    with open(TMDB_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(tmdb_cache, f, ensure_ascii=False)
    print(f"   ✓ Saved")
    
    # Create enriched tags
    enriched_tags = create_enriched_tags(movies_df, item2index, tmdb_cache)
    
    # Save enriched tags
    print("\n6. Saving enriched tags...")
    DATA_DIR.mkdir(exist_ok=True)
    with open(ENRICHED_TAGS_FILE, 'wb') as f:
        pickle.dump(enriched_tags, f)
    print(f"   ✓ Saved to: {ENRICHED_TAGS_FILE}")
    
    # Show samples
    print("\n" + "=" * 60)
    print("SAMPLE TAGS")
    print("=" * 60)
    
    for category in ["actor", "lead", "director", "genre", "keyword", "studio", "collection", "decade", "country", "rating", "budget"]:
        sample_tags = [t for t in enriched_tags.keys() if t.startswith(f"{category}:")]
        if sample_tags:
            print(f"\n{category.upper()} ({len(sample_tags):,} tags):")
            for tag in sorted(sample_tags, key=lambda t: -len(enriched_tags[t]))[:5]:
                print(f"  {tag}: {len(enriched_tags[tag]):,} movies")
    
    print("\n" + "=" * 60)
    print("DONE! Next: python build_text_index.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
