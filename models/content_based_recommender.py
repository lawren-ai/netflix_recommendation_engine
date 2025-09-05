import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from typing import List, Dict, Tuple, Optional
import pickle
import os

class ContentBasedRecommender:
    """
    Content-based recommendation system
    
    How it works:
    1. Create feature vectors for each movie (genres, overview text, ratings)
    2. Calculate similarity between movies using these features
    3. Recommend movies most similar to ones the user liked
    """
    
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.genre_encoder = MultiLabelBinarizer()
        self.is_trained = False
        
    def load_data(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Load our movie data"""
        try:
            self.movies_df = pd.read_csv(csv_path)
            print(f"📚 Loaded {len(self.movies_df)} movies")
            return True
        except FileNotFoundError:
            print(f"❌ Could not find {csv_path}. Run data ingestion first!")
            return False
    
    def prepare_features(self):
        """
        Create feature vectors for each movie
        
        Production insight: Feature engineering is where the magic happens!
        We're combining multiple types of data into numerical vectors.
        """
        print("🔧 Engineering features...")
        
        # 1. Text features from movie overviews
        # TF-IDF converts text into numbers based on word importance
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Top 1000 most important words
            stop_words='english',  # Remove common words like "the", "and"
            ngram_range=(1, 2)  # Use single words and word pairs
        )
        
        # Handle missing overviews
        overviews = self.movies_df['overview'].fillna('')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(overviews)
        
        # 2. Genre features
        # Genres are already lists in our data
        genres_lists = []
        for genres in self.movies_df['genres']:
            try:
                if pd.notna(genres):
                    # If it's already a list, use it; if it's a string, evaluate it
                    if isinstance(genres, list):
                        genres_lists.append(genres)
                    elif isinstance(genres, str) and genres.startswith('['):
                        genres_lists.append(eval(genres))
                    else:
                        genres_lists.append([])
                else:
                    genres_lists.append([])
            except:
                genres_lists.append([])
        
        genre_matrix = self.genre_encoder.fit_transform(genres_lists)
        
        # 3. Numerical features (ratings, popularity, etc.)
        numerical_features = self.movies_df[[
            'vote_average', 'vote_count', 'popularity'
        ]].fillna(0)
        
        # Scale numerical features so they don't dominate
        numerical_scaled = self.scaler.fit_transform(numerical_features)
        
        # 4. Combine all features
        # This is the key step - creating one big feature vector per movie
        self.feature_matrix = np.hstack([
            tfidf_matrix.toarray(),  # Text features
            genre_matrix,            # Genre features  
            numerical_scaled         # Numerical features
        ])
        
        print(f"✅ Created feature matrix: {self.feature_matrix.shape}")
        print(f"   • Text features: {tfidf_matrix.shape[1]}")
        print(f"   • Genre features: {genre_matrix.shape[1]}")
        print(f"   • Numerical features: {numerical_scaled.shape[1]}")
    
    def calculate_similarity(self):
        """
        Calculate how similar each movie is to every other movie
        
        We use cosine similarity - it measures the angle between feature vectors
        Values range from 0 (completely different) to 1 (identical)
        """
        print("📐 Calculating movie similarities...")
        
        # This is computationally expensive for large datasets!
        # In production, you'd use approximate methods or precompute this
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        print(f"✅ Similarity matrix created: {self.similarity_matrix.shape}")
        
        # Quick sanity check
        avg_similarity = np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
        print(f"📊 Average similarity between movies: {avg_similarity:.3f}")
    
    def train(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Train the recommendation model"""
        print("🚀 Training content-based recommender...")
        
        if not self.load_data(csv_path):
            return False
        
        self.prepare_features()
        self.calculate_similarity()
        
        self.is_trained = True
        print("🎉 Training complete!")
        return True
    
    def get_recommendations(self, movie_title: str, n_recommendations: int = 5) -> List[Dict]:
        """
        Get movie recommendations based on a movie the user liked
        
        Args:
            movie_title: Title of movie user liked
            n_recommendations: How many recommendations to return
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        # Smart movie search - this is crucial for user experience!
        movie_idx, movie_info = self._find_movie(movie_title)
        
        if movie_idx is None:
            print(f"❌ Could not find a movie matching '{movie_title}'")
            self._suggest_similar_titles(movie_title)
            return []
        
        print(f"🎬 Finding movies similar to: {movie_info['title']}")
        
        # Get similarity scores for this movie with all others
        similarity_scores = self.similarity_matrix[movie_idx]
        
        # Get indices of most similar movies (excluding the movie itself)
        similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
        
        # Build recommendation list
        recommendations = []
        for idx in similar_indices:
            similar_movie = self.movies_df.iloc[idx]
            recommendations.append({
                'title': similar_movie['title'],
                'similarity_score': similarity_scores[idx],
                'vote_average': similar_movie['vote_average'],
                'overview': similar_movie['overview'][:200] + "..." if len(str(similar_movie['overview'])) > 200 else similar_movie['overview'],
                'genres': similar_movie['genres']
            })
        
        return recommendations
    
    def _find_movie(self, search_term: str) -> Tuple[Optional[int], Optional[Dict]]:
        """
        Smart movie search with fuzzy matching
        
        Production tip: Users never type exact titles!
        """
        search_term = search_term.lower().strip()
        
        # Try exact match first
        exact_matches = self.movies_df[
            self.movies_df['title'].str.lower() == search_term
        ]
        if not exact_matches.empty:
            idx = exact_matches.index[0]
            return idx, exact_matches.iloc[0]
        
        # Try partial match
        partial_matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(search_term, na=False)
        ]
        if not partial_matches.empty:
            # Return the most popular match
            best_match_idx = partial_matches['popularity'].idxmax()
            return best_match_idx, partial_matches.loc[best_match_idx]
        
        # Try word-by-word matching (for "toy story" -> "Toy Story 4")
        search_words = search_term.split()
        if len(search_words) > 1:
            for _, movie in self.movies_df.iterrows():
                movie_title_lower = str(movie['title']).lower()
                if all(word in movie_title_lower for word in search_words):
                    return movie.name, movie
        
        return None, None
    
    def _suggest_similar_titles(self, search_term: str, max_suggestions: int = 5):
        """Suggest similar movie titles when search fails"""
        suggestions = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(
                search_term.split()[0], na=False
            )
        ]['title'].head(max_suggestions).tolist()
        
        if suggestions:
            print(f"💡 Did you mean one of these?")
            for title in suggestions:
                print(f"   • {title}")
        else:
            print("💡 Try searching for a different movie title")
            print("   Available movies include:")
            sample_titles = self.movies_df['title'].head(10).tolist()
            for title in sample_titles:
                print(f"   • {title}")
    
    def save_model(self, path: str = 'models/trained/content_recommender.pkl'):
        """Save the trained model for production use"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'similarity_matrix': self.similarity_matrix,
                'movies_df': self.movies_df,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler,
                'genre_encoder': self.genre_encoder,
                'feature_matrix': self.feature_matrix
            }, f)
        
        print(f"💾 Model saved to {path}")
    
    def load_model(self, path: str = 'models/trained/content_recommender.pkl'):
        """Load a pre-trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.similarity_matrix = data['similarity_matrix']
        self.movies_df = data['movies_df']
        self.tfidf_vectorizer = data['tfidf_vectorizer']
        self.scaler = data['scaler']
        self.genre_encoder = data['genre_encoder']
        self.feature_matrix = data['feature_matrix']
        self.is_trained = True
        
        print("📥 Model loaded successfully!")

# Demo function to test our recommender
def demo_recommendations():
    """Test our recommendation system"""
    recommender = ContentBasedRecommender()
    
    # Train the model
    if not recommender.train():
        return
    
    # Save the model
    recommender.save_model()
    
    # Test recommendations
    test_movies = ["Avengers", "Toy Story", "Titanic"]
    
    for movie in test_movies:
        print(f"\n" + "="*60)
        recommendations = recommender.get_recommendations(movie, n_recommendations=3)
        
        if recommendations:
            print(f"Because you liked '{movie}', you might like:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['title']} (Similarity: {rec['similarity_score']:.3f})")
                print(f"   Rating: {rec['vote_average']}/10")
                print(f"   Overview: {rec['overview']}")
        else:
            print(f"❌ No movies found matching '{movie}'")

if __name__ == "__main__":
    demo_recommendations()