from typing import List, Dict, Optional
import random
from models.improved_content_recommender import ImprovedContentRecommender
from models.popularity_recommender import PopularityRecommender

class ImprovedHybridRecommender:
    """
    Enhanced hybrid system with improved content-based engine
    """
    
    def __init__(self, similarity_threshold=0.4, rating_threshold=6.5):
        self.content_recommender = ImprovedContentRecommender(
            min_similarity_threshold=similarity_threshold,
            min_rating_threshold=rating_threshold
        )
        self.popularity_recommender = PopularityRecommender()
        self.is_trained = False
        
    def train(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Train both improved engines"""
        print("🚀 Training improved hybrid system...")
        
        content_success = self.content_recommender.train(csv_path)
        self.popularity_recommender.train(csv_path)
        
        self.is_trained = True
        print("🎉 Improved hybrid system ready!")
        return content_success
    
    def get_smart_recommendations(self, user_profile: Dict) -> Dict:
        """Enhanced smart recommendations with quality filtering"""
        user_id = user_profile.get('user_id', 'anonymous')
        liked_movies = user_profile.get('liked_movies', [])
        preferred_genres = user_profile.get('preferred_genres', [])
        
        print(f"🤖 Enhanced recommendations for user {user_id}")
        
        if len(liked_movies) == 0:
            return self._get_cold_start_recommendations(preferred_genres)
        else:
            return self._get_personalized_recommendations(liked_movies, preferred_genres)
    
    def _get_personalized_recommendations(self, liked_movies: List[str], 
                                        preferred_genres: List[str]) -> Dict:
        """Get high-quality personalized recommendations"""
        recommendations = {
            'strategy': 'improved_personalized',
            'sections': []
        }
        
        # Get content-based recommendations
        all_content_recs = []
        for movie in liked_movies:
            movie_recs = self.content_recommender.get_improved_recommendations(
                movie, n_recommendations=5
            )
            if movie_recs:
                all_content_recs.extend(movie_recs)
        
        if all_content_recs:
            # Remove duplicates, sort by similarity
            seen_titles = set()
            unique_recs = []
            for rec in sorted(all_content_recs, 
                            key=lambda x: x['similarity_score'], reverse=True):
                if rec['title'] not in seen_titles:
                    unique_recs.append(rec)
                    seen_titles.add(rec['title'])
            
            if unique_recs:
                recommendations['sections'].append({
                    'title': '🎯 High-Quality Matches',
                    'movies': unique_recs[:6],
                    'reason': 'Similar to your liked movies with great ratings'
                })
        
        # Add trending as backup
        trending = self.popularity_recommender.get_trending_overall(3)
        if trending:
            recommendations['sections'].append({
                'title': '🔥 Also Trending',
                'movies': trending,
                'reason': 'Popular high-quality movies'
            })
        
        return recommendations
    
    def _get_cold_start_recommendations(self, preferred_genres: List[str]) -> Dict:
        """Cold start with genre preferences"""
        recommendations = {
            'strategy': 'improved_cold_start',
            'sections': []
        }
        
        # Overall trending
        trending = self.popularity_recommender.get_trending_overall(4)
        recommendations['sections'].append({
            'title': '🔥 Trending Now',
            'movies': trending,
            'reason': 'Popular with all users'
        })
        
        # Genre-specific
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

# Test function for evaluation
def create_improved_hybrid_for_evaluation():
    """Factory function to create improved hybrid for evaluation"""
    return ImprovedHybridRecommender(similarity_threshold=0.3, rating_threshold=6.0)