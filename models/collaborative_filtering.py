import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFilteringRecommender:
    """
    Netflix-style collaborative filtering recommendation system
    
    Two approaches implemented:
    1. User-Based CF: Find similar users, recommend what they liked
    2. Matrix Factorization (SVD): Advanced technique using latent factors
    
    This is the algorithm that made Netflix famous!
    """
    
    def __init__(self, n_factors=50, min_interactions=3):
        self.movies_df = None
        self.ratings_matrix = None
        self.user_similarity_matrix = None
        self.movie_similarity_matrix = None
        self.svd_model = None
        self.user_factors = None
        self.movie_factors = None
        self.is_trained = False
        
        # Hyperparameters
        self.n_factors = n_factors  # Number of latent factors (Netflix uses 100-500)
        self.min_interactions = min_interactions
        
        # Mappings for sparse matrix
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        
    def load_data(self, movies_csv: str = 'data/raw/movies_popular.csv', 
                  ratings_data: pd.DataFrame = None):
        """
        Load movie and user rating data
        
        In production, ratings_data would come from your user interaction database
        """
        # Load movies
        self.movies_df = pd.read_csv(movies_csv)
        print(f"📚 Loaded {len(self.movies_df)} movies")
        
        # For demo, we'll create or use provided ratings data
        if ratings_data is not None:
            self.ratings_df = ratings_data
        else:
            # Generate realistic synthetic ratings
            self.ratings_df = self._generate_realistic_ratings()
            
        print(f"⭐ Loaded {len(self.ratings_df)} ratings from {self.ratings_df['user_id'].nunique()} users")
        
        return True
    
    def _generate_realistic_ratings(self, n_users=200) -> pd.DataFrame:
        """
        Generate realistic user ratings with patterns that collaborative filtering can learn
        
        Key insight: CF works because users have similar taste patterns
        """
        print("🧪 Generating realistic user rating patterns...")
        
        ratings_data = []
        movie_titles = self.movies_df['title'].tolist()
        
        # Create user archetypes (this is what makes CF work!)
        user_archetypes = {
            'action_lover': {'Action': 0.8, 'Adventure': 0.7, 'Thriller': 0.6, 'Comedy': 0.2},
            'drama_fan': {'Drama': 0.8, 'Romance': 0.7, 'Comedy': 0.5, 'Action': 0.2},
            'sci_fi_geek': {'Science Fiction': 0.9, 'Adventure': 0.6, 'Action': 0.5, 'Romance': 0.1},
            'family_viewer': {'Animation': 0.8, 'Comedy': 0.7, 'Adventure': 0.6, 'Horror': 0.0},
            'horror_enthusiast': {'Horror': 0.9, 'Thriller': 0.8, 'Mystery': 0.6, 'Comedy': 0.3}
        }
        
        for user_id in range(1, n_users + 1):
            # Each user is a mix of archetypes (realistic!)
            primary_archetype = np.random.choice(list(user_archetypes.keys()))
            secondary_archetype = np.random.choice(list(user_archetypes.keys()))
            
            primary_prefs = user_archetypes[primary_archetype]
            secondary_prefs = user_archetypes[secondary_archetype]
            
            # Blend preferences (70% primary, 30% secondary)
            user_prefs = {}
            all_genres = set(list(primary_prefs.keys()) + list(secondary_prefs.keys()))
            for genre in all_genres:
                user_prefs[genre] = (
                    0.7 * primary_prefs.get(genre, 0) + 
                    0.3 * secondary_prefs.get(genre, 0)
                )
            
            # Each user rates 10-30 movies
            n_ratings = np.random.randint(10, 31)
            rated_movies = np.random.choice(movie_titles, size=n_ratings, replace=False)
            
            for movie_title in rated_movies:
                # Get movie genres
                movie_row = self.movies_df[self.movies_df['title'] == movie_title]
                if movie_row.empty:
                    continue
                    
                movie_genres = movie_row.iloc[0]['genres']
                if isinstance(movie_genres, str):
                    movie_genres = eval(movie_genres)
                
                # Calculate preference score for this movie
                preference_score = 0
                for genre in movie_genres:
                    preference_score += user_prefs.get(genre, 0.1)
                
                preference_score = min(1.0, preference_score / len(movie_genres))
                
                # Convert preference to rating (1-5 scale)
                if preference_score >= 0.7:
                    rating = np.random.choice([4, 5], p=[0.3, 0.7])
                elif preference_score >= 0.5:
                    rating = np.random.choice([3, 4], p=[0.5, 0.5])
                elif preference_score >= 0.3:
                    rating = np.random.choice([2, 3], p=[0.6, 0.4])
                else:
                    rating = np.random.choice([1, 2], p=[0.7, 0.3])
                
                ratings_data.append({
                    'user_id': user_id,
                    'movie_title': movie_title,
                    'rating': rating,
                    'user_archetype': f"{primary_archetype}+{secondary_archetype}"
                })
        
        ratings_df = pd.DataFrame(ratings_data)
        print(f"✅ Generated {len(ratings_df)} ratings with realistic user patterns")
        
        # Show pattern analysis
        archetype_stats = ratings_df.groupby('user_archetype')['rating'].agg(['count', 'mean'])
        print(f"📊 User archetype patterns:")
        for archetype, stats in archetype_stats.head().iterrows():
            print(f"   {archetype}: {stats['count']} ratings, avg {stats['mean']:.2f}")
        
        return ratings_df
    
    def create_user_item_matrix(self):
        """
        Create the user-item rating matrix - the foundation of collaborative filtering
        
        Rows = Users, Columns = Movies, Values = Ratings
        This is typically a VERY sparse matrix (most users haven't seen most movies)
        """
        print("🔧 Creating user-item rating matrix...")
        
        # Create mappings
        unique_users = sorted(self.ratings_df['user_id'].unique())
        unique_movies = sorted(self.ratings_df['movie_title'].unique())
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}
        
        n_users = len(unique_users)
        n_movies = len(unique_movies)
        
        # Create sparse rating matrix
        ratings_matrix = np.zeros((n_users, n_movies))
        
        for _, row in self.ratings_df.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            movie_idx = self.movie_to_idx[row['movie_title']]
            ratings_matrix[user_idx, movie_idx] = row['rating']
        
        self.ratings_matrix = csr_matrix(ratings_matrix)
        
        # Calculate sparsity
        total_possible_ratings = n_users * n_movies
        actual_ratings = len(self.ratings_df)
        sparsity = (1 - actual_ratings / total_possible_ratings) * 100
        
        print(f"📊 Rating matrix created: {n_users} users × {n_movies} movies")
        print(f"   Sparsity: {sparsity:.1f}% (typical for real systems: 95-99%)")
        
    def calculate_user_similarity(self):
        """
        Calculate similarity between users based on their rating patterns
        
        Users who rate movies similarly are considered similar
        This is the core of user-based collaborative filtering
        """
        print("👥 Calculating user similarity patterns...")
        
        # Use cosine similarity on user vectors
        self.user_similarity_matrix = cosine_similarity(self.ratings_matrix)
        
        # Analyze similarity patterns
        n_users = self.ratings_matrix.shape[0]
        avg_similarity = np.mean(self.user_similarity_matrix[np.triu_indices(n_users, k=1)])
        
        print(f"📊 User similarity analysis:")
        print(f"   Average similarity: {avg_similarity:.3f}")
        print(f"   Most similar users: {np.max(self.user_similarity_matrix[np.triu_indices(n_users, k=1)]):.3f}")
        
    def train_matrix_factorization(self):
        """
        Train SVD (Singular Value Decomposition) model
        
        This is Netflix's secret sauce! It finds latent factors that explain user preferences:
        - Factor 1 might be "likes action movies" 
        - Factor 2 might be "prefers newer movies"
        - Factor 3 might be "enjoys complex plots"
        """
        print("🧠 Training matrix factorization (Netflix's secret sauce)...")
        
        self.svd_model = TruncatedSVD(n_components=self.n_factors, random_state=42)
        
        # Fit on the ratings matrix
        self.user_factors = self.svd_model.fit_transform(self.ratings_matrix)
        self.movie_factors = self.svd_model.components_.T
        
        # Calculate explained variance
        explained_variance_ratio = np.sum(self.svd_model.explained_variance_ratio_)
        
        print(f"✅ Matrix factorization complete!")
        print(f"   Latent factors: {self.n_factors}")
        print(f"   Explained variance: {explained_variance_ratio:.1%}")
        print(f"   User factors shape: {self.user_factors.shape}")
        print(f"   Movie factors shape: {self.movie_factors.shape}")
    
    def get_user_based_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        User-based collaborative filtering
        
        Algorithm:
        1. Find users most similar to target user
        2. Find movies those similar users liked
        3. Recommend movies target user hasn't seen
        """
        if user_id not in self.user_to_idx:
            print(f"❌ User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user similarities (excluding self)
        user_similarities = self.user_similarity_matrix[user_idx].copy()
        user_similarities[user_idx] = 0  # Exclude self
        
        # Find most similar users
        similar_user_indices = np.argsort(user_similarities)[::-1][:20]  # Top 20 similar users
        
        # Get movies the target user has already rated
        user_rated_movies = set()
        user_ratings = self.ratings_matrix[user_idx].toarray().flatten()
        for movie_idx, rating in enumerate(user_ratings):
            if rating > 0:
                user_rated_movies.add(self.idx_to_movie[movie_idx])
        
        # Calculate recommendation scores
        movie_scores = {}
        
        for similar_user_idx in similar_user_indices:
            similarity_score = user_similarities[similar_user_idx]
            if similarity_score <= 0:
                break
            
            # Get this similar user's ratings
            similar_user_ratings = self.ratings_matrix[similar_user_idx].toarray().flatten()
            
            for movie_idx, rating in enumerate(similar_user_ratings):
                if rating >= 4:  # Only consider highly rated movies
                    movie_title = self.idx_to_movie[movie_idx]
                    
                    # Skip movies user has already seen
                    if movie_title not in user_rated_movies:
                        if movie_title not in movie_scores:
                            movie_scores[movie_title] = 0
                        
                        # Weighted score by user similarity
                        movie_scores[movie_title] += similarity_score * rating
        
        # Sort and return top recommendations
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_title, score in sorted_movies[:n_recommendations]:
            # Get movie details
            movie_info = self.movies_df[self.movies_df['title'] == movie_title]
            if not movie_info.empty:
                movie_data = movie_info.iloc[0]
                recommendations.append({
                    'title': movie_title,
                    'collab_score': score,
                    'vote_average': movie_data['vote_average'],
                    'genres': movie_data['genres'],
                    'overview': movie_data['overview'][:150] + "..." if len(str(movie_data['overview'])) > 150 else movie_data['overview'],
                    'recommendation_reason': f'Loved by similar users (score: {score:.2f})'
                })
        
        print(f"🎯 Found {len(recommendations)} collaborative recommendations for user {user_id}")
        return recommendations
    
    def get_matrix_factorization_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        Matrix factorization recommendations using latent factors
        
        This is the advanced Netflix algorithm!
        """
        if user_id not in self.user_to_idx:
            print(f"❌ User {user_id} not found in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Get user's factor vector
        user_vector = self.user_factors[user_idx]
        
        # Calculate predicted ratings for all movies
        predicted_ratings = np.dot(user_vector, self.movie_factors.T)
        
        # Get movies user hasn't rated
        user_ratings = self.ratings_matrix[user_idx].toarray().flatten()
        unrated_movie_indices = np.where(user_ratings == 0)[0]
        
        # Get top predictions for unrated movies
        unrated_predictions = [(idx, predicted_ratings[idx]) for idx in unrated_movie_indices]
        unrated_predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_idx, predicted_rating in unrated_predictions[:n_recommendations]:
            movie_title = self.idx_to_movie[movie_idx]
            
            # Get movie details
            movie_info = self.movies_df[self.movies_df['title'] == movie_title]
            if not movie_info.empty:
                movie_data = movie_info.iloc[0]
                recommendations.append({
                    'title': movie_title,
                    'predicted_rating': predicted_rating,
                    'vote_average': movie_data['vote_average'],
                    'genres': movie_data['genres'],
                    'overview': movie_data['overview'][:150] + "..." if len(str(movie_data['overview'])) > 150 else movie_data['overview'],
                    'recommendation_reason': f'Predicted rating: {predicted_rating:.2f} (based on latent factors)'
                })
        
        print(f"🧠 Generated {len(recommendations)} matrix factorization recommendations")
        return recommendations
    
    def train(self, movies_csv: str = 'data/raw/movies_popular.csv', ratings_data: pd.DataFrame = None):
        """Train the collaborative filtering system"""
        print("🚀 Training collaborative filtering system...")
        
        self.load_data(movies_csv, ratings_data)
        self.create_user_item_matrix()
        self.calculate_user_similarity()
        self.train_matrix_factorization()
        
        self.is_trained = True
        print("🎉 Collaborative filtering system ready!")
        return True

# Demo function
def demo_collaborative_filtering():
    """Test our collaborative filtering system"""
    cf_recommender = CollaborativeFilteringRecommender(n_factors=20)  # Smaller for demo
    cf_recommender.train()
    
    # Test both approaches
    test_user_id = 1
    
    print(f"\n" + "="*80)
    print(f"🎯 COLLABORATIVE FILTERING RECOMMENDATIONS FOR USER {test_user_id}")
    print("="*80)
    
    # Show user's previous ratings
    user_ratings = cf_recommender.ratings_df[cf_recommender.ratings_df['user_id'] == test_user_id]
    print(f"\n📽️  User {test_user_id}'s Previous Ratings:")
    for _, rating in user_ratings.head(5).iterrows():
        print(f"   {rating['movie_title']}: {rating['rating']}/5")
    
    # User-based recommendations
    print(f"\n👥 USER-BASED COLLABORATIVE FILTERING:")
    user_based_recs = cf_recommender.get_user_based_recommendations(test_user_id, 3)
    for i, rec in enumerate(user_based_recs, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   {rec['recommendation_reason']}")
        print(f"   Genres: {rec['genres']}")
    
    # Matrix factorization recommendations
    print(f"\n🧠 MATRIX FACTORIZATION RECOMMENDATIONS:")
    mf_recs = cf_recommender.get_matrix_factorization_recommendations(test_user_id, 3)
    for i, rec in enumerate(mf_recs, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   {rec['recommendation_reason']}")
        print(f"   Actual rating: {rec['vote_average']}/10")

if __name__ == "__main__":
    demo_collaborative_filtering()