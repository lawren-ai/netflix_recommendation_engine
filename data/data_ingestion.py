import requests
import pandas as pd
import time
import json
from typing import List, Dict, Optional
from config.settings import settings

class TMDBDataFetcher:
    """Handles fetching movie data from TMDB API"""

    def __init__(self):
        self.base_url = settings.TMDB_BASE_URL
        self.api_key = settings.TMDB_API_KEY
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a request to TMDB API with error handling and rate limiting
        """
        if params is None:
            params = {}
        
        params['api_key'] = self.api_key
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            # Rate limiting
            time.sleep(0.25) # 4 requests per second 

            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
        
    def get_popular_movies(self, pages: int = 5) -> List[Dict]:
        """
        Fetch popular movies from TMDB

         Why start with popular movies?
        - Good baseline data
        - Less likely to have missing information
        - Represents what users actually watch
        """
        movies = []

        print(f"🎬 Fetching popular movies ({pages} pages)...")
        
        for page in range(1, pages + 1):
            print(f" 📄 Fetching page {page}/{pages}")
            data = self._make_request("movie/popular", {"page": page})

            if data and "results" in data:
                movies.extend(data['results'])
            
            else:
                print(f"⚠️  Failed to fetch page {page}")
            
        print(f"✅ Fetched {len(movies)} movies")
        return movies
    
    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get detailed information for a specific movie
        """
        return self._make_request(f"movie/{movie_id}")
    
    def get_movie_credits(self, movie_id: int) -> Optional[Dict]:
        """Get cast and crew information for a movie"""
        return self._make_request(f"movie/{movie_id}/credits")
    
class DataProcessor:
    """Processes raw TMDB data into features we can use for recommendations"""

    def clean_movie_data(self, raw_movies: List[Dict]) -> pd.DataFrame:
        """Convert raw TMDB data into a clean pandas DataFrame"""

        processed_movies = []

        for movie in raw_movies:
            try:
                # Extract inportant features
                processed_movie = {
                    'movie_id': movie.get('id'),
                    'title': movie.get('title', 'Unknown'),
                    'overview': movie.get('overview', ''),
                    'genres': self._extract_genres(movie.get('genre_ids', [])),
                    'release_date': movie.get('release_date'),
                    'vote_average': movie.get('vote_average', 0),
                    'vote_count': movie.get('vote_count', 0),
                    'popularity': movie.get('popularity', 0),
                    'original_language': movie.get('original_language', 'en'),
                    'adult': movie.get('adult', False)
                }

                # Only include movies with basic required info
                if processed_movie['movie_id'] and processed_movie['title']:
                    processed_movies.append(processed_movie)

            except Exception as e:
                print(f"⚠️  Error processing movie {movie.get('id', 'unknown')}: {e}")
                continue

        df = pd.DataFrame(processed_movies)

        print(f"📊 Data Quality Report: ")
        print(f"  • Total movies: {len(df)}")
        print(f"  • Missing overviews: {df['overview'].isna().sum()}")
        print(f"  • Average rating: {df['vote_average'].mean():.2f}")
        
        return df
    
    def _extract_genres(self, genre_ids: List[int]) -> List[str]:
        """
        Convert TMDB genre IDs to genre names
        """
        # TMDB genre ID mapping (subset)
        genre_map = {
            28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
            80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
            14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
            9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
            10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
        }            
        
        return [genre_map.get(gid, f"Unknown_{gid}") for gid in genre_ids]
    

def fetch_initial_dataset():
    """Fetch our initial dataset - what to run once to bootstrap"""
    fetcher = TMDBDataFetcher()
    processor = DataProcessor()

    # Fetch popular movies
    raw_movies = fetcher.get_popular_movies(pages=3)  # starting small

    # process into clean DataFrane
    movies_df = processor.clean_movie_data(raw_movies)

    # Save into data directory
    movies_df.to_csv("data/raw/movies_popular.csv", index =  False)

    print("🎉 Initial dataset created! ")
    return movies_df

if __name__ == "__main__":
    # Test our data pipeline 
    df = fetch_initial_dataset()
    print("f\n📋 Sample of data")
    print(df.head(3))
    