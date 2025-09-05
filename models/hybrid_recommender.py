from typing import List, Dict, Optional
import random
from models.content_based_recommender import ContentBasedRecommender
from models.popularity_recommender import PopularityRecommender

class HybridRecommender:
    """
    Netflix-style hybrid recommendation system
    
    Production strategy:
    - New users → Popularity-based recommendations
    - Users with viewing history → Content-based recommendations  
    - Fallback strategies when primary method fails
    - A/B testing capabilities for algorithm comparison
    """
    
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.popularity_recommender = PopularityRecommender()
        self.is_trained = False
        
    def train(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Train both recommendation engines"""
        print("🚀 Training hybrid recommendation system...")
        
        # Train both engines
        print("\n1️⃣ Training content-based engine...")
        content_success = self.content_recommender.train(csv_path)
        
        print("\n2️⃣ Training popularity-based engine...")
        self.popularity_recommender.train(csv_path)
        
        self.is_trained = True
        print("\n🎉 Hybrid system ready!")
        return content_success
    
    def get_recommendations_for_new_user(self, preferred_genres: List[str] = None, 
                                       n_recommendations: int = 10) -> Dict:
        """
        Cold start solution: Recommendations for users with no viewing history
        
        Strategy:
        - Mix of overall trending + genre-based trending
        - Include variety across different genres
        """
        if not self.is_trained:
            raise ValueError("System not trained! Call train() first.")
        
        print("🆕 Generating recommendations for new user...")
        
        recommendations = {
            'strategy': 'new_user_cold_start',
            'sections': []
        }
        
        # 1. Overall trending (always include)
        trending = self.popularity_recommender.get_trending_overall(5)
        recommendations['sections'].append({
            'title': '🔥 Trending Now',
            'movies': trending,
            'reason': 'Popular with all users'
        })
        
        # 2. Genre-specific recommendations if preferences provided
        if preferred_genres:
            for genre in preferred_genres[:2]:  # Max 2 genres to avoid overwhelming
                genre_movies = self.popularity_recommender.get_trending_by_genre(genre, 3)
                if genre_movies:
                    recommendations['sections'].append({
                        'title': f'🎭 Trending in {genre}',
                        'movies': genre_movies,
                        'reason': f'Popular {genre} movies'
                    })
        
        # 3. Diverse genre sampling (ensure variety)
        all_genres = list(self.popularity_recommender.genre_popularity.keys())
        if all_genres and not preferred_genres:
            # Pick 2 random genres for variety
            sample_genres = random.sample(all_genres, min(2, len(all_genres)))
            for genre in sample_genres:
                genre_movies = self.popularity_recommender.get_trending_by_genre(genre, 2)
                if genre_movies:
                    recommendations['sections'].append({
                        'title': f'🎬 {genre} Movies',
                        'movies': genre_movies,
                        'reason': f'Explore {genre} genre'
                    })
        
        return recommendations
    
    def get_recommendations_for_existing_user(self, liked_movies: List[str], 
                                            n_recommendations: int = 10) -> Dict:
        """
        Personalized recommendations for users with viewing history
        
        Strategy:
        - Primary: Content-based on user's liked movies
        - Secondary: Add some trending movies for discovery
        """
        if not self.is_trained:
            raise ValueError("System not trained! Call train() first.")
        
        print("👤 Generating personalized recommendations...")
        
        recommendations = {
            'strategy': 'personalized_content_based',
            'sections': []
        }
        
        # 1. Content-based recommendations for each liked movie
        all_content_recs = []
        for movie in liked_movies:
            movie_recs = self.content_recommender.get_recommendations(
                movie, n_recommendations=3
            )
            if movie_recs:
                all_content_recs.extend(movie_recs)
        
        if all_content_recs:
            # Remove duplicates and sort by similarity
            seen_titles = set()
            unique_recs = []
            for rec in sorted(all_content_recs, 
                            key=lambda x: x['similarity_score'], reverse=True):
                if rec['title'] not in seen_titles:
                    unique_recs.append(rec)
                    seen_titles.add(rec['title'])
            
            recommendations['sections'].append({
                'title': '🎯 Because You Watched',
                'movies': unique_recs[:n_recommendations//2],
                'reason': 'Based on your viewing history'
            })
        
        # 2. Add some trending movies for discovery
        trending = self.popularity_recommender.get_trending_overall(3)
        # Remove movies user already liked
        trending_filtered = [
            movie for movie in trending 
            if movie['title'] not in liked_movies
        ]
        
        if trending_filtered:
            recommendations['sections'].append({
                'title': '🔥 Trending Now',
                'movies': trending_filtered,
                'reason': 'Discover what\'s popular'
            })
        
        return recommendations
    
    def get_smart_recommendations(self, user_profile: Dict) -> Dict:
        """
        Smart router: Decide which strategy to use based on user profile
        
        This is the main production endpoint!
        """
        user_id = user_profile.get('user_id', 'anonymous')
        liked_movies = user_profile.get('liked_movies', [])
        preferred_genres = user_profile.get('preferred_genres', [])
        
        print(f"🤖 Smart recommendations for user {user_id}")
        print(f"   Viewing history: {len(liked_movies)} movies")
        
        # Decision logic (this is where the AI magic happens!)
        if len(liked_movies) == 0:
            # Cold start: New user
            print("   Strategy: Cold start (new user)")
            return self.get_recommendations_for_new_user(
                preferred_genres=preferred_genres
            )
        
        elif len(liked_movies) < 3:
            # Hybrid: Some history but still exploring
            print("   Strategy: Hybrid (exploring user)")
            # Combine both strategies
            personalized = self.get_recommendations_for_existing_user(liked_movies)
            trending = self.get_recommendations_for_new_user(preferred_genres, 5)
            
            # Merge strategies
            combined = {
                'strategy': 'hybrid_exploration',
                'sections': personalized['sections'] + trending['sections'][:1]
            }
            return combined
        
        else:
            # Full personalization
            print("   Strategy: Full personalization")
            return self.get_recommendations_for_existing_user(liked_movies)
    
    def run_ab_test(self, user_profile: Dict, test_group: str = 'A') -> Dict:
        """
        A/B testing framework for comparing recommendation strategies
        
        Production use: Test which algorithm performs better
        """
        if test_group == 'A':
            # Control group: Standard hybrid logic
            return self.get_smart_recommendations(user_profile)
        else:
            # Test group: Pure popularity-based
            return self.get_recommendations_for_new_user(
                user_profile.get('preferred_genres', [])
            )

# Demo function
def demo_hybrid_system():
    """Test our production-ready hybrid system"""
    system = HybridRecommender()
    system.train()
    
    print("\n" + "="*80)
    print("🎬 NETFLIX-STYLE HYBRID RECOMMENDATION DEMO")
    print("="*80)
    
    # Test Case 1: Brand new user
    print("\n📱 SCENARIO 1: Brand New User (No viewing history)")
    new_user = {
        'user_id': 'user_001', 
        'liked_movies': [], 
        'preferred_genres': ['Action', 'Thriller']
    }
    
    recommendations = system.get_smart_recommendations(new_user)
    print(f"Strategy used: {recommendations['strategy']}")
    
    for section in recommendations['sections']:
        print(f"\n{section['title']}")
        print(f"Reason: {section['reason']}")
        for i, movie in enumerate(section['movies'][:2], 1):  # Show top 2
            print(f"  {i}. {movie['title']} ⭐ {movie['vote_average']}")
    
    # Test Case 2: User with some viewing history
    print("\n" + "="*80)
    print("📱 SCENARIO 2: User with Viewing History")
    existing_user = {
        'user_id': 'user_002',
        'liked_movies': ['Superman', 'Mission: Impossible - The Final Reckoning'],
        'preferred_genres': ['Action']
    }
    
    recommendations = system.get_smart_recommendations(existing_user)
    print(f"Strategy used: {recommendations['strategy']}")
    
    for section in recommendations['sections']:
        print(f"\n{section['title']}")
        print(f"Reason: {section['reason']}")
        for i, movie in enumerate(section['movies'][:2], 1):
            print(f"  {i}. {movie['title']}")

if __name__ == "__main__":
    demo_hybrid_system()