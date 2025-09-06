import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from typing import List, Dict, Tuple
import pickle
import os

class ImprovedContentRecommender:
    """
    Improved content-based recommender with production optimizations
    
    Key improvements:
    1. Similarity thresholds - Only recommend truly similar movies
    2. Genre boosting - Give extra weight to genre matches
    3. Quality filtering - Don't recommend low-rated movies
    4. Popularity boosting - Blend in trending content
    """
    
    def __init__(self, min_similarity_threshold=0.3, min_rating_threshold=6.0):
        self.movies_df = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.genre_encoder = MultiLabelBinarizer()
        self.is_trained = False
        
        # Production tuning parameters
        self.min_similarity_threshold = min_similarity_threshold
        self.min_rating_threshold = min_rating_threshold
        
    def load_data(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Load our movie data"""
        try:
            self.movies_df = pd.read_csv(csv_path)
            print(f"📚 Loaded {len(self.movies_df)} movies")
            return True
        except FileNotFoundError:
            print(f"❌ Could not find {csv_path}. Run data ingestion first!")
            return False
    
    def prepare_enhanced_features(self):
        """
        Enhanced feature engineering with production optimizations
        """
        print("🔧 Engineering enhanced features...")
        
        # 1. Enhanced text features - use both overview and title
        combined_text = (
            self.movies_df['title'].fillna('') + ' ' + 
            self.movies_df['overview'].fillna('')
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,  # Increased from 1000
            stop_words='english',
            ngram_range=(1, 3),  # Include 3-word phrases
            min_df=2,  # Must appear in at least 2 movies
            max_df=0.8  # Don't use words that appear in >80% of movies
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
        
        # 2. Enhanced genre features with better parsing
        genres_lists = []
        for genres in self.movies_df['genres']:
            try:
                if pd.notna(genres):
                    if isinstance(genres, str) and genres.startswith('['):
                        genres_lists.append(eval(genres))
                    elif isinstance(genres, list):
                        genres_lists.append(genres)
                    else:
                        genres_lists.append([])
                else:
                    genres_lists.append([])
            except:
                genres_lists.append([])
        
        genre_matrix = self.genre_encoder.fit_transform(genres_lists)
        
        # 3. Enhanced numerical features with better scaling
        numerical_features = self.movies_df[[
            'vote_average', 'vote_count', 'popularity'
        ]].fillna(0)
        
        # Log transform for popularity (reduces extreme values)
        numerical_features = numerical_features.copy()
        numerical_features['popularity'] = np.log1p(numerical_features['popularity'])
        
        numerical_scaled = self.scaler.fit_transform(numerical_features)
        
        # 4. NEW: Quality features
        # Add features that indicate movie quality
        quality_features = []
        for _, movie in self.movies_df.iterrows():
            # High rating + many votes = quality indicator
            quality_score = movie['vote_average'] * np.log1p(movie['vote_count'])
            quality_features.append([quality_score])
        
        quality_features = np.array(quality_features)
        quality_scaled = StandardScaler().fit_transform(quality_features)
        
        # 5. Combine all features with optimized weights
        self.feature_matrix = np.hstack([
            tfidf_matrix.toarray() * 1.0,      # Text features (base weight)
            genre_matrix * 2.0,                # Genre features (2x weight - important!)
            numerical_scaled * 0.8,            # Numerical features (lower weight)
            quality_scaled * 0.5               # Quality features (moderate weight)
        ])
        
        print(f"✅ Enhanced feature matrix: {self.feature_matrix.shape}")
        print(f"   • Text features: {tfidf_matrix.shape[1]} (weight: 1.0)")
        print(f"   • Genre features: {genre_matrix.shape[1]} (weight: 2.0)")
        print(f"   • Numerical features: {numerical_scaled.shape[1]} (weight: 0.8)")
        print(f"   • Quality features: {quality_scaled.shape[1]} (weight: 0.5)")
    
    def calculate_similarity(self):
        """Calculate similarity with enhanced features"""
        print("📐 Calculating enhanced movie similarities...")
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        # Quality check
        avg_similarity = np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
        print(f"📊 Average similarity: {avg_similarity:.3f}")
        
        # Check similarity distribution
        similarities = self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)]
        print(f"📊 Similarity stats:")
        print(f"   • Min: {similarities.min():.3f}")
        print(f"   • Max: {similarities.max():.3f}")
        print(f"   • 90th percentile: {np.percentile(similarities, 90):.3f}")
    
    def get_improved_recommendations(self, movie_title: str, n_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations with quality filtering and similarity thresholds
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first.")
        
        movie_idx, movie_info = self._find_movie(movie_title)
        
        if movie_idx is None:
            print(f"❌ Could not find movie matching '{movie_title}'")
            self._suggest_similar_titles(movie_title)
            return []
        
        print(f"🎬 Finding high-quality similar movies to: {movie_info['title']}")
        
        # Get all similarity scores for this movie
        similarity_scores = self.similarity_matrix[movie_idx]
        
        # Create candidate list with filtering
        candidates = []
        for idx, similarity in enumerate(similarity_scores):
            if idx == movie_idx:  # Skip the movie itself
                continue
                
            candidate_movie = self.movies_df.iloc[idx]
            
            # Quality filters
            if (similarity >= self.min_similarity_threshold and 
                candidate_movie['vote_average'] >= self.min_rating_threshold):
                
                candidates.append({
                    'idx': idx,
                    'similarity': similarity,
                    'vote_average': candidate_movie['vote_average'],
                    'vote_count': candidate_movie['vote_count']
                })
        
        if not candidates:
            print(f"⚠️  No high-quality similar movies found above threshold {self.min_similarity_threshold}")
            print("💡 Try lowering min_similarity_threshold or min_rating_threshold")
            return []
        
        # Sort by similarity, then by rating for ties
        candidates.sort(key=lambda x: (x['similarity'], x['vote_average']), reverse=True)
        
        # Build recommendation list
        recommendations = []
        for candidate in candidates[:n_recommendations]:
            similar_movie = self.movies_df.iloc[candidate['idx']]
            recommendations.append({
                'title': similar_movie['title'],
                'similarity_score': candidate['similarity'],
                'vote_average': similar_movie['vote_average'],
                'vote_count': similar_movie['vote_count'],
                'overview': similar_movie['overview'][:200] + "..." if len(str(similar_movie['overview'])) > 200 else similar_movie['overview'],
                'genres': similar_movie['genres'],
                'quality_reason': f"High similarity ({candidate['similarity']:.3f}) + Good rating ({similar_movie['vote_average']:.1f})"
            })
        
        print(f"✅ Found {len(recommendations)} high-quality recommendations")
        print(f"   Similarity range: {recommendations[0]['similarity_score']:.3f} to {recommendations[-1]['similarity_score']:.3f}")
        
        return recommendations
    
    def _find_movie(self, search_term: str) -> Tuple:
        """Smart movie search with fuzzy matching"""
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
            best_match_idx = partial_matches['popularity'].idxmax()
            return best_match_idx, partial_matches.loc[best_match_idx]
        
        # Try word-by-word matching
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
            print("💡 Available high-rated movies:")
            top_movies = self.movies_df.nlargest(5, 'vote_average')['title'].tolist()
            for title in top_movies:
                print(f"   • {title}")
    
    def train(self, csv_path: str = 'data/raw/movies_popular.csv'):
        """Train the improved recommendation model"""
        print("🚀 Training improved content-based recommender...")
        
        if not self.load_data(csv_path):
            return False
        
        self.prepare_enhanced_features()
        self.calculate_similarity()
        
        self.is_trained = True
        print("🎉 Improved training complete!")
        return True

# Demo function
def demo_improved_recommendations():
    """Test our improved recommender"""
    # Test different threshold settings
    thresholds = [
        (0.3, 6.0, "Balanced"),
        (0.5, 7.0, "High Quality"),
        (0.1, 5.0, "More Variety")
    ]
    
    for sim_thresh, rating_thresh, description in thresholds:
        print(f"\n" + "="*70)
        print(f"🧪 TESTING: {description} (similarity≥{sim_thresh}, rating≥{rating_thresh})")
        print("="*70)
        
        recommender = ImprovedContentRecommender(
            min_similarity_threshold=sim_thresh,
            min_rating_threshold=rating_thresh
        )
        recommender.train()
        
        # Test with Superman
        recommendations = recommender.get_improved_recommendations("Superman", 3)
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['title']}")
                print(f"   {rec['quality_reason']}")
                print(f"   Overview: {rec['overview']}")
        else:
            print("❌ No recommendations found with these thresholds")

if __name__ == "__main__":
    demo_improved_recommendations()