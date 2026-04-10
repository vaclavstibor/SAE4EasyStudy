import datetime
import time
from imdb import Cinemagoer

import pandas as pd
import numpy as np
from collections import defaultdict
import os

import json

from scipy.spatial.distance import squareform, pdist

from PIL import Image
from io import BytesIO
import requests

# from app import pm
from flask import url_for, has_app_context

from plugins.utils.helpers import cos_sim_np

# Movie-lens data loader

class RatingUserFilter:
    def __init__(self, min_ratings_per_user):
        self.min_ratings_per_user = min_ratings_per_user

    def __call__(self, loader):
        # First filter out users who gave <= 1 ratings
        loader.ratings_df = loader.ratings_df[loader.ratings_df['userId'].map(loader.ratings_df['userId'].value_counts()) >= self.min_ratings_per_user]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
        print(f"Ratings shape after user filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")
        
# Filters out all low ratings
class RatingLowFilter:
    def __init__(self, min_rating):
        self.min_rating = min_rating
    def __call__(self, loader):
        loader.ratings_df = loader.ratings_df[loader.ratings_df.rating >= self.min_rating]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)

class RatingMovieFilter:
    def __init__(self, min_ratings_per_movie):
        self.min_ratings_per_movie = min_ratings_per_movie
    def __call__(self, loader):
        # Filter out users that were rated <= 1 times
        loader.ratings_df = loader.ratings_df[loader.ratings_df['movieId'].map(loader.ratings_df['movieId'].value_counts()) >= self.min_ratings_per_movie]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
        print(f"Ratings shape after item filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")

class RatingTagFilter:
    def __init__(self, min_tags_per_movie):
        self.min_tags_per_movie = min_tags_per_movie
    def __call__(self, loader):
        # Filter out movies that do not have enough tags (we do not want those movies to end up in the dense pool for group based elicitation)
        # Even if they have many ratings
        tags_per_movie = loader.tags_df.groupby("movieId")["movieId"].count()
        tags_per_movie = tags_per_movie[tags_per_movie > self.min_tags_per_movie]
        print(f"Ratings shape before tag filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")
        loader.ratings_df = loader.ratings_df[loader.ratings_df.movieId.isin(tags_per_movie)]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
        print(f"Ratings shape after tag filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")

class RatedMovieFilter:
    def __call__(self, loader):
        # We are only interested in movies for which we hav
        loader.movies_df = loader.movies_df[loader.movies_df.movieId.isin(loader.ratings_df.movieId.unique())]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

# Filters out all ratings of movies that do not have enough ratings per year
class RatingsPerYearFilter:
    def __init__(self, min_ratings_per_year):
        self.min_ratings_per_year = min_ratings_per_year

    def __call__(self, loader):
        movies_df_indexed = loader.movies_df.set_index("movieId")

        # Add column with age of each movie
        movies_df_indexed.loc[:, "age"] = movies_df_indexed.year.max() - movies_df_indexed.year
        
        # Calculate number of ratings per year for each of the movies
        loader.ratings_df.loc[:, "ratings_per_year"] = loader.ratings_df['movieId'].map(loader.ratings_df['movieId'].value_counts()) / loader.ratings_df['movieId'].map(movies_df_indexed["age"])
        
        # Filter out movies that do not have enough yearly ratings
        loader.ratings_df = loader.ratings_df[loader.ratings_df.ratings_per_year >= self.min_ratings_per_year]

class MovieFilterByYear:
    def __init__(self, min_year):
        self.min_year = min_year
        
    def _parse_year(self, x):
        x = x.split("(")
        if len(x) <= 1:
            return 0
        try:
            return int(x[-1].split(")")[0])
        except:
            return 0

    def __call__(self, loader):
        # Filter out unrated movies and old movies
        # Add year column      
        loader.movies_df.loc[:, "year"] = loader.movies_df.title.apply(self._parse_year)
        loader.movies_df = loader.movies_df[loader.movies_df.year >= self.min_year]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

class RatingFilterOld:
    def __init__(self, oldest_rating_year):
        self.oldest_rating_year = oldest_rating_year
    def __call__(self, loader):
        # Marker for oldest rating
        oldest_rating = datetime.datetime(year=self.oldest_rating_year, month=1, day=1, tzinfo=datetime.timezone.utc).timestamp()
        # Filter ratings that are too old
        loader.ratings_df = loader.ratings_df[loader.ratings_df.timestamp > oldest_rating]
        #loader.ratings_df = loader.ratings_df.reset_index(drop=True)

# Just filters out tags that are not present on any of the rated movies
class TagsRatedMoviesFilter:
    def __call__(self, loader):
        print(f"TagsRatedMoviesFilter before: {loader.tags_df.shape}")
        loader.tags_df = loader.tags_df[loader.tags_df.movieId.isin(loader.ratings_df.movieId.unique())]
        loader.tags_df = loader.tags_df.reset_index(drop=True)
        print(f"TagsRatedMoviesFilter after: {loader.tags_df.shape}")

class TagsFilter:
    def __init__(self, most_rated_items_subset_ids, min_num_tag_occurrences):
        self.min_num_tag_occurrences = min_num_tag_occurrences
        self.most_rated_items_subset_ids = most_rated_items_subset_ids
    def __call__(self, loader):
        # For the purpose of group-based preference elicitation we are only interested in tags that occurr in dense subset of items
        # over which the groups are defined
        loader.tags_df = loader.tags_df[loader.tags_df.movieId.isin(self.most_rated_items_subset_ids)]
        print(f"Tags_df shape: {loader.tags_df.shape}")
        # We also only consider tags that have enough occurrences, otherwise we will not be able to find enough representants (movies) for each tag
        loader.tags_df = loader.tags_df[loader.tags_df['tag'].map(loader.tags_df['tag'].value_counts()) >= self.min_num_tag_occurrences]
        print(f"Tags_df shape: {loader.tags_df.shape}")
        loader.tags_df = loader.tags_df.reset_index(drop=True)

class LinkFilter:
    def __call__(self, loader):
        loader.links_df = loader.links_df[loader.links_df.index.isin((loader.movies_df.movieId))]

class MLDataLoader:
    def __init__(
        self,
        ratings_path,
        movies_path,
        tags_path,
        links_path,
        filters=None,
        rating_matrix_path=None,
        img_dir_path=None,
        descriptions_path=None,
        skip_matrices: bool = False,
    ):

        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.tags_path = tags_path
        self.filters = filters
        self.links_path = links_path
        self.rating_matrix_path = rating_matrix_path

        self.ratings_df = None
        self.movies_df = None
        self.movies_df_indexed = None
        self.tags_df = None
        self.links_df = None
        self.rating_matrix = None
        self.movie_index_to_id = None
        self.movie_id_to_index = None
        self.num_movies = None
        self.num_users = None
        self.user_to_user_index = None
        self.movie_index_to_description = None
        self.tag_counts_per_movie = None

        self.access = Cinemagoer()
        self.movie_index_to_url = dict()
        self.similarity_matrix = None

        self.img_dir_path = img_dir_path
        self.descriptions_path = descriptions_path

        self.trailer_urls = None
        self.plots = None

        # Optional: skip building rating/similarity matrices (for large datasets)
        self.skip_matrices = skip_matrices

    def _get_image(self, imdbId):
        try:
            return self.access.get_movie(imdbId)["full-size cover url"]
        except Exception as e:
            print(f"@@ Exception e={e}")
            return ""

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle imdb
        del state["access"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.access = Cinemagoer()
        # Reload image mappings after unpickling from cache
        # This ensures movie_index_to_url is up to date even if cache is old
        if self.img_dir_path and os.path.exists(self.img_dir_path):
            print(f"[DEBUG __setstate__] Reloading image mappings from {self.img_dir_path}")
            self.movie_index_to_url = dict()
            already_downloaded = os.listdir(self.img_dir_path)
            jpg_files = [f for f in already_downloaded if f.endswith('.jpg') and not f.startswith('.')]
            print(f"[DEBUG __setstate__] Found {len(jpg_files)} JPG files on disk")
            
            # Check if movie_id_to_index exists
            if not hasattr(self, 'movie_id_to_index'):
                print(f"[DEBUG __setstate__] WARNING: movie_id_to_index not found! Cannot map images.")
                return
            
            print(f"[DEBUG __setstate__] movie_id_to_index has {len(self.movie_id_to_index)} entries")
            
            mapped_count = 0
            not_in_index_count = 0
            sample_not_mapped = []
            
            for img_name in jpg_files:
                try:
                    movie_id = int(img_name.split(".jpg")[0])
                    if movie_id in self.movie_id_to_index:
                        movie_idx = self.movie_id_to_index[movie_id]
                        self.movie_index_to_url[movie_idx] = os.path.join(self.img_dir_path, img_name)
                        mapped_count += 1
                    else:
                        not_in_index_count += 1
                        if len(sample_not_mapped) < 5:
                            sample_not_mapped.append(movie_id)
                except (ValueError, IndexError):
                    continue
            
            print(f"[DEBUG __setstate__] Mapped {mapped_count} images to movie_index_to_url (out of {len(jpg_files)} total)")
            if not_in_index_count > 0:
                print(f"[DEBUG __setstate__] {not_in_index_count} images NOT mapped (movie_id not in movie_id_to_index)")
                print(f"[DEBUG __setstate__] Sample unmapped movie_ids: {sample_not_mapped}")

    def get_trailer_url(self, movie_idx):
        movie_id = str(self.movie_index_to_id[movie_idx])
        if self.trailer_urls is None or movie_id not in self.trailer_urls:
            return ""
        return self.trailer_urls[movie_id]
    
    def get_plot(self, movie_idx):
        movie_id = str(self.movie_index_to_id[movie_idx])
        if self.plots is None or movie_id not in self.plots:
            return ""
        return self.plots[str(movie_id)]

    # Download all the images
    def download_images(self):
        i = 0
        start_time = time.perf_counter()
        for movie_idx, url in self.movie_index_to_url.items():
            
            if i % 100 == 0:
                print(f"{i}/{len(self.movie_index_to_url)} images were downloaded, took: {time.perf_counter() - start_time}")
                start_time = time.perf_counter()

            movie_id = self.movie_index_to_id[movie_idx]
            
            err = False
            try:
                resp = requests.get(url, stream=True)
            except:
                err = True

            if err or resp.status_code != 200:
                print(f"Error downloading image from url={url} for movie_id={movie_id}")
                # Try once more retrieving fresh url for the given movie
                imdbId = self.links_df.loc[movie_id].imdbId
                new_url = self._get_image(imdbId)

                err = False
                try:
                    resp = requests.get(new_url, stream=True)
                except:
                    err = True

                if err or resp.status_code != 200:
                    print(f"Even new URL={new_url} failed")
                    continue # Ignore the movie
                else:
                    print("New URL succeeded")

            img = Image.open(BytesIO(resp.content))
            width, height = img.size
            TARGET_WIDTH = 200
            coef = TARGET_WIDTH / width
            new_height = int(height * coef)
            img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS).convert('RGB')
            img.save(os.path.join(self.img_dir_path, f'{movie_id}.jpg'), quality=90)

            i += 1

    def get_image(self, movie_idx):
        if self.img_dir_path and has_app_context():
            movie_id = self.movie_index_to_id[movie_idx]
            img_path = os.path.join(self.img_dir_path, f'{movie_id}.jpg')
            
            # If image exists on disk, use it immediately
            if os.path.exists(img_path):
                item_id = self.movie_index_to_id[movie_idx]
                return url_for('static', filename=f'datasets/ml-latest/img/{item_id}.jpg')
            
            # Image doesn't exist — return None immediately.
            # Never block the request thread with an IMDb download.
            return None

        if movie_idx in self.movie_index_to_url:
            return self.movie_index_to_url[movie_idx]
        return None

    def apply_tag_filter(self, tag_filter, *args, **kwargs):
        self.tags_df = tag_filter(self.tags_df, *args, **kwargs)

        # Set of all unique tags
        self.tags = set(self.tags_df.tag.unique())
        # Maps movie index to tag counts per movie
        self.tag_counts_per_movie = { movie_index : defaultdict(int) for movie_index in self.movie_index_to_id.keys() }
        for group_name, group_df in self.tags_df.groupby("movieId"):
            for _, row in group_df.iterrows():
                self.tag_counts_per_movie[self.movie_id_to_index[group_name]][row.tag] += 1

    # Passing local_movie_images as parameter to prevent cache changes
    # If local_movie_images==True, we will use local files instead of image urls inside img uris
    def load(self):
        #### Data Loading ####

        # Load ratings
        self.ratings_df = pd.read_csv(self.ratings_path)
        print(f"Ratings shape: {self.ratings_df.shape}, n_users = {self.ratings_df.userId.unique().size}, n_items = {self.ratings_df.movieId.unique().size}")
        
        # Load tags and convert them to lower case
        self.tags_df = pd.read_csv(self.tags_path)
        self.tags_df.tag = self.tags_df.tag.str.casefold()

        # Load movies
        self.movies_df = pd.read_csv(self.movies_path)

        # Load links
        self.links_df = pd.read_csv(self.links_path, index_col=0)
        
        #### Filtering ####

        # # Filter rating dataframe
        # if self.ratings_df_filter:
        #     self.ratings_df = self.ratings_df_filter(self.ratings_df, self)

        # # Filter movies dataframe
        # if self.movies_df_filter:
        #     self.movies_df = self.movies_df_filter(self.movies_df, self.ratings_df)

        for f in self.filters:
            f(self)

        self.movie_index_to_id = pd.Series(self.movies_df.movieId.values,index=self.movies_df.index).to_dict()
        self.movie_id_to_index = pd.Series(self.movies_df.index,index=self.movies_df.movieId.values).to_dict()
        num_movies = len(self.movie_id_to_index)

        # # Filter tags dataframe
        # if self.tags_df_filter:
        #     self.apply_tag_filter(self.tags_df_filter, self.ratings_df)


        unique_users = self.ratings_df.userId.unique()
        num_users = unique_users.size
        
        self.user_to_user_index = dict(zip(unique_users, range(num_users)))
        
        if not self.skip_matrices:
            ratings_df_i = self.ratings_df.copy()
            ratings_df_i.userId = ratings_df_i.userId.map(self.user_to_user_index)
            ratings_df_i.movieId = ratings_df_i.movieId.map(self.movie_id_to_index)
            self.rating_matrix = self.ratings_df.pivot(index='userId', columns='movieId', values="rating").fillna(0).values
            # Faster, tensorflow-based implementation
            self.similarity_matrix = cos_sim_np(self.rating_matrix.T)
            self.distance_matrix = 1.0 - self.similarity_matrix
        else:
            self.rating_matrix = None
            self.similarity_matrix = None
            self.distance_matrix = None
        
        # Maps movie index to text description
        self.movies_df["description"] = self.movies_df.title + ' ' + self.movies_df.genres
        self.movie_index_to_description = dict(zip(self.movies_df.index, self.movies_df.description))
        

        self.movies_df_indexed = self.movies_df.set_index("movieId")


        if self.descriptions_path:
            with open(self.descriptions_path, "r", encoding="UTF-8") as f:
                data = json.load(f)
                self.trailer_urls = {movie_id: records["videos"][0] if len(records["videos"]) > 0 else "" for movie_id, records in data.items()}
                self.plots = {movie_id: records["plot"] for movie_id, records in data.items()}

        # First check which images are downloaded so far
        print(f"[DEBUG load] Checking images in img_dir_path: {self.img_dir_path}")
        print(f"[DEBUG load] img_dir_path exists: {os.path.exists(self.img_dir_path)}")
        
        already_downloaded = [] if not os.path.exists(self.img_dir_path) else os.listdir(self.img_dir_path)
        print(f"[DEBUG load] Found {len(already_downloaded)} files in img_dir_path")
        
        # Skip system files
        jpg_files = [f for f in already_downloaded if f.endswith('.jpg') and not f.startswith('.')]
        print(f"[DEBUG load] Found {len(jpg_files)} JPG files (after filtering)")
        
        self.movie_index_to_url = dict()
        mapped_count = 0
        for img_name in jpg_files:
            try:
                movie_id = int(img_name.split(".jpg")[0])
                if movie_id in self.movie_id_to_index: # It could be possible we have more images than movies (due to filtering)
                    movie_idx = self.movie_id_to_index[movie_id]
                    self.movie_index_to_url[movie_idx] = os.path.join(self.img_dir_path, img_name)
                    mapped_count += 1
            except (ValueError, IndexError) as e:
                print(f"[DEBUG load] Could not parse movie_id from {img_name}: {e}")
                continue
        
        print(f"[DEBUG load] Already downloaded images for: {len(self.movie_index_to_url)} movies (mapped {mapped_count} files)")
        if len(self.movie_index_to_url) > 0:
            sample_keys = list(self.movie_index_to_url.keys())[:3]
            print(f"[DEBUG load] Sample movie_index_to_url keys: {sample_keys}")
        return self