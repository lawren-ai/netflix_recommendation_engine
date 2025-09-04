import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

class PopularityRecommender:
    """
    Popularity-based recommendation system for cold start problem
    use cases:
    - New users with no viewing history
    - Trending/featured content sections
    - Falback when personalized recommendations fail
    """

    def __init__(self):
        self.movies_df = None
        self.genre_popularity = {}
        self.overall_trending = None

    def load_data(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Load movie data"""
        self.movies_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.movies_df)} movies for popularity analysis")
        return True
    
    def calculate_popularity_scores(self):
        """
        Calculate different types of popularity scores

        - Most watchd(vote_count)
        - Highest rated(vote_average)
        - Currently trending (popularitu score from TMDB)
        - Balanced score (combination)
        """

        print(" Calculating popularity scores...")

        self.movies_df['norm_vote_count'] = (
            self.movies_df['vote_count'] / self.movies_df['vote_count'].max()
        )
        self.movies_df['norm_vote_average'] = (
            self.movies_df['vote_average'] / 10.0 
        )
        self.movies_df['norm_popularity'] = (
            self.movies_df['popularity'] / self.movies_df['popularity'].max()
        )

        # Weighted popularity score 
        # Weight: 40% current trending, 30% rating, 30% vote count
        self.movies_df['weighted_popularity'] = (
            0.4 * self.movies_df['norm_popularity'] +
            0.3 * self.movies_df['norm_vote_average'] +
            0.3 * self.movies_df['norm_vote_count']
        )

        print("Popularity scores calculated")
    
    def analyze_genre_trends(self):
        """
        Calculate popularity by genre
        Use: "Trending in Action", "Top Comedy", e.t.c
        """
        genre_stats = defaultdict(list)

        for _, movie in self.movies_df.iterrows():
            try:
                # Handle genre data
                genres = movie['genres']
                if isinstance(genres, str):
                    genres= eval(genres)
                for genre in genres:
                    genre_stats[genre].append({
                        'title': movie['title'],
                        'popularity_score': movie['weighted_popularuty'],
                        'vote_average': movie['vote_average'],
                        'vote_count': movie['vote_count']
                    })
            except:
                continue

        
        # calculate acerage popularity by genre
        self.genre_popularity = {}
        for genre, movies in genre_stats.items():
            if len(movies) >= 2:
                avg_popularity = np.mean([m['popularity_score'] for m in movies])
                self.genre_popularity[genre] = {
                    'avg_popularity': avg_popularity,
                    'movie_count': len(movies),
                    'top_movies': sorted(movies,
                                         key=lambda x: x['popularity_score'],
                                         reverse=True)[:3]
                }
        print(f" ANalyzed {len(self.genre_popularity)} genres")

    def get_trending_overall(self, n_movies: int = 10)-> List[Dict]:
        """Get overall trending movies for homepage"""
        trending = self.movies_df.nlargest(n_movies, 'weighted_popularity')

        recommendations = []
        for _, movie in trending.iterrows():
            recommendations.append({
                'title': movie['title'],
                'popularity_score': movie['weighted_popularity'],
                'vote_average': movie['vote_average'],
                'overview': movie['overview'][:150] + "..." if len(str(movie['overview'])) > 150 else movie['overview'],
                'genres': movie['genres'],
                'recommendation_reason': 'Trending Now'
            })

        return recommendations
    
    def get_trending_by_genre(self, genre: str, n_movies: int=5) -> List[Dict]:
        """Get trending movies in a specific genre"""
        if genre not in self.genre_popularity:
            print(f"Genre '{genre}' not found")
            print(f"AVailable genres: {list(self.genre_popularity.keys())}")
            return []
        
        # Filter movies by genre
        genre_movies = []
        for _, movie in self.movies_df.iterrows():
            try:
                movie_genres = movie['genres']
                if isinstance(movie_genres, str):
                    movie_genres = eval(movie_genres)
                
                if genre in movie_genres:
                    genre_movies.append(movie)
            except:
                continue

        # sort by popularity and take top 4
        genre_df = pd.DataFrame(genre_movies)
        if genre_df.empty:
            return []
        
        trending = genre_df.nlargest(n_movies, 'weighted_popularity')

        recommendations = []
        for _, movie in trending.iterrowa():
            recommendations.append({
                'title': movie['title'],
                'popularity_score': movie['weighted_popularity'],
                'vote_average': movie['vote_average'],
                'overview': movie['overview'][:150] + "..." if len(str(movie['overview'])) >150 else movie['overview'],
                'genres': movie['genres'],
                'recommendation_readon': f"Trending in {genre}"
            })
        
        return recommendations
    
    def get_genre_insights(self):
        """Get insights about what is trending across genres"""
        print("\n GENRE POPULAITY INSIGHTS")
        print("="*50)

        # Sprt genres by popularity
        sorted_genres = sorted(
            self.genre_popularity.items(),
            key=lambda x: x[1]['avg_popularity'],
            reverse=True
        )

        for genre, stats in sorted_genres:
            print(f"{genre}")
            print(f"    Average popularity: {stats['avg_popularity']:.3f}")
            print(f"    Movies in dataset: {stats['movie_count']}")
            print(f"   Top movie: {stats['top_movies'][0]['title']}")

    def train(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Train the popularity recommender"""
        print("training popularity based recommender...")

        self.load_data(csv_path)
        self.calculate_popularity_scores()
        self.analyze_genre_trends()

        print(" Populariity recommender ready!")


def demo_popularity_recommendations():
    """Test popularity recommender"""
    recommender = PopularityRecommender()
    recommender.train()

    print("\n" + "="*60)
    print(" OVERALL TRENDING MOVIES")
    print("="*60)

    trending = recommender.get_trending_overall(5)
    for i, movie in enumerate(trending, 1):
        print(f"\n{i}, {movie['title']} (Score: {movie['popularity_score']:.3f})")
        print(f"    Rating: {movie['vote_average']}/10")
        print(f"     {movie['overview']}")

    # Show genre insights
    recommender.get_genre_insights()

    # Test genre-specific recommendations
    print("\n" + "="*60)
    print("TRENDING IN ACTION")
    print("="*60)

    action_movies = recommender.get_trending_by_genre('Action', 3)
    for i, movie in enumerate(action_movies, 1):
        print(f"\n{i}, movie['title]")
        print(f"    Rating: {movie['vote_average']}/10")

    
if __name__ == "__main__":
    demo_popularity_recommendations()

