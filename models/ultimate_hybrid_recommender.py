from typing import List, Dict, Optional
import random
import pandas as pd
from models.improved_content_recommender import ImprovedContentRecommender
from models.popularity_recommender import PopularityRecommender
from models.collaborative_filtering import CollaborativeFilteringRecommender

class UltimateHybridRecommender:
    """
    Netflix-level hybrid system combining ALL recommendation strategies:
    
    1. Collaborative Filtering (primary for users with data)
    2. Content-Based (secondary, for diversity)
    3. Popularity-Based (fallback for cold start)
    
    This is the production architecture used by Netflix, Spotify, Amazon!
    """
    
    def __init__(self):
        self.content_recommender = ImprovedContentRecommender(
            min_similarity_threshold=0.3,
            min_rating_threshold=6.0
        )
        self.popularity_recommender = PopularityRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender(n_factors=30)
        self.is_trained = False
        self.user_ratings_db = None  # Simulated user ratings database
        
    def train(self, csv_path: str = 'data/raw/movies_popular.csv', user_ratings: pd.DataFrame = None):
        """Train all recommendation engines"""
        print("🚀 Training ULTIMATE hybrid recommendation system...")
        
        # Train content-based and popularity engines
        print("\n1️⃣ Training content-based engine...")
        self.content_recommender.train(csv_path)
        
        print("\n2️⃣ Training popularity-based engine...")
        self.popularity_recommender.train(csv_path)
        
        # Train collaborative filtering engine
        print("\n3️⃣ Training collaborative filtering engine...")
        self.collaborative_recommender.train(csv_path, user_ratings)
        
        # Store user ratings for decision making
        if user_ratings is not None:
            self.user_ratings_db = user_ratings
        else:
            self.user_ratings_db = self.collaborative_recommender.ratings_df
        
        self.is_trained = True
        print("\n🎉 ULTIMATE hybrid system ready!")
        print("   ✅ Content-based recommendations")
        print("   ✅ Popularity-based recommendations") 
        print("   ✅ Collaborative filtering recommendations")
        return True
    
    def get_smart_recommendations(self, user_profile: Dict) -> Dict:
        """
        Netflix-style intelligent recommendation routing
        
        Decision tree:
        - New users (no data) → Popularity + Genre preferences
        - Users with some data → Hybrid (Collaborative + Content + Popular)
        - Power users (lots of data) → Collaborative primary + Content for diversity
        """
        user_id = user_profile.get('user_id', 'anonymous')
        liked_movies = user_profile.get('liked_movies', [])
        preferred_genres = user_profile.get('preferred_genres', [])
        
        # Count user's interaction history
        user_interaction_count = 0
        if self.user_ratings_db is not None and isinstance(user_id, int):
            user_ratings = self.user_ratings_db[self.user_ratings_db['user_id'] == user_id]
            user_interaction_count = len(user_ratings)
        
        print(f"🤖 Ultimate recommendations for user {user_id}")
        print(f"   Liked movies: {len(liked_movies)}")
        print(f"   Rating history: {user_interaction_count} ratings")
        
        # Routing logic (Netflix's secret sauce!)
        if user_interaction_count >= 10:
            return self._get_collaborative_primary_recommendations(user_id, liked_movies)
        elif user_interaction_count >= 3:
            return self._get_hybrid_recommendations(user_id, liked_movies, preferred_genres)
        else:
            return self._get_cold_start_recommendations(liked_movies, preferred_genres)
    
    def _get_collaborative_primary_recommendations(self, user_id: int, liked_movies: List[str]) -> Dict:
        """
        Primary collaborative filtering with content-based diversity
        For power users with lots of interaction data
        """
        recommendations = {
            'strategy': 'collaborative_primary',
            'sections': []
        }
        
        # Main recommendations from collaborative filtering
        collab_recs = self.collaborative_recommender.get_matrix_factorization_recommendations(
            user_id, n_recommendations=8
        )
        
        if collab_recs:
            recommendations['sections'].append({
                'title': '🎯 Because Users Like You Also Loved',
                'movies': collab_recs,
                'reason': 'Based on similar user preferences (collaborative filtering)'
            })
        
        # Add some content-based for diversity
        if liked_movies:
            content_recs = []
            for movie in liked_movies[-2:]:  # Use recent likes
                movie_recs = self.content_recommender.get_improved_recommendations(movie, 2)
                content_recs.extend(movie_recs)
            
            if content_recs:
                # Remove duplicates
                seen_titles = {movie['title'] for section in recommendations['sections'] for movie in section['movies']}
                diverse_recs = [rec for rec in content_recs if rec['title'] not in seen_titles][:3]
                
                if diverse_recs:
                    recommendations['sections'].append({
                        'title': '🎭 More Like What You Watched',
                        'movies': diverse_recs,
                        'reason': 'Content-based diversity recommendations'
                    })
        
        return recommendations
    
    def _get_hybrid_recommendations(self, user_id: int, liked_movies: List[str], 
                                  preferred_genres: List[str]) -> Dict:
        """
        Balanced hybrid approach for users with moderate interaction history
        """
        recommendations = {
            'strategy': 'balanced_hybrid',
            'sections': []
        }
        
        # Try collaborative filtering first
        if isinstance(user_id, int):
            collab_recs = self.collaborative_recommender.get_user_based_recommendations(
                user_id, n_recommendations=5
            )
            
            if collab_recs:
                recommendations['sections'].append({
                    'title': '👥 Users Like You Also Enjoyed',
                    'movies': collab_recs,
                    'reason': 'Based on similar users\' preferences'
                })
        
        # Add content-based recommendations
        if liked_movies:
            content_recs = []
            for movie in liked_movies:
                movie_recs = self.content_recommender.get_improved_recommendations(movie, 2)
                content_recs.extend(movie_recs)
            
            if content_recs:
                # Remove duplicates
                seen_titles = {movie['title'] for section in recommendations['sections'] for movie in section['movies']}
                unique_content_recs = []
                for rec in content_recs:
                    if rec['title'] not in seen_titles:
                        unique_content_recs.append(rec)
                        seen_titles.add(rec['title'])
                
                if unique_content_recs:
                    recommendations['sections'].append({
                        'title': '🎬 Similar to Your Favorites',
                        'movies': unique_content_recs[:4],
                        'reason': 'Content-based on your liked movies'
                    })
        
        # Add trending as backup
        trending = self.popularity_recommender.get_trending_overall(3)
        if trending:
            recommendations['sections'].append({
                'title': '🔥 Also Trending',
                'movies': trending,
                'reason': 'Popular with all users'
            })
        
        return recommendations
    
    def _get_cold_start_recommendations(self, liked_movies: List[str], 
                                      preferred_genres: List[str]) -> Dict:
        """
        Cold start strategy for new users
        """
        recommendations = {
            'strategy': 'cold_start_popularity',
            'sections': []
        }
        
        # Overall trending
        trending = self.popularity_recommender.get_trending_overall(5)
        recommendations['sections'].append({
            'title': '🔥 Trending Now',
            'movies': trending,
            'reason': 'Popular with all users'
        })
        
        # Genre-specific if preferences provided
        if preferred_genres:
            for genre in preferred_genres[:2]:
                genre_movies = self.popularity_recommender.get_trending_by_genre(genre, 3)
                if genre_movies:
                    recommendations['sections'].append({
                        'title': f'🎭 Best in {genre}',
                        'movies': genre_movies,
                        'reason': f'Top-rated {genre} movies'
                    })
        
        return recommendations

# Factory function for evaluation
def create_ultimate_hybrid_for_evaluation():
    """Create the ultimate hybrid system for evaluation"""
    return UltimateHybridRecommender()

# Demo function
def demo_ultimate_system():
    """Demo the complete Netflix-style system"""
    print("🎬 ULTIMATE NETFLIX-STYLE RECOMMENDATION SYSTEM")
    print("="*70)
    
    system = UltimateHybridRecommender()
    system.train()
    
    # Test different user types
    test_users = [
        {
            'user_id': 1,  # Power user (lots of ratings)
            'liked_movies': ['Superman', 'Mission: Impossible - The Final Reckoning'],
            'preferred_genres': ['Action']
        },
        {
            'user_id': 150,  # Moderate user 
            'liked_movies': ['Superman'],
            'preferred_genres': ['Action', 'Adventure']
        },
        {
            'user_id': 'new_user',  # Brand new user
            'liked_movies': [],
            'preferred_genres': ['Comedy', 'Drama']
        }
    ]
    
    for user_profile in test_users:
        print(f"\n" + "="*70)
        print(f"📱 USER SCENARIO: {user_profile['user_id']}")
        print("="*70)
        
        recommendations = system.get_smart_recommendations(user_profile)
        print(f"Strategy: {recommendations['strategy']}")
        
        for section in recommendations['sections']:
            print(f"\n{section['title']}")
            print(f"Reason: {section['reason']}")
            for i, movie in enumerate(section['movies'][:2], 1):
                print(f"  {i}. {movie['title']}")

if __name__ == "__main__":
    demo_ultimate_system()