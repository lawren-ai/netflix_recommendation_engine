import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from models.hybrid_recommender import HybridRecommender

class RecommendationEvaluator:
    """
    Production-grade evaluation framework for recommendation systems
    
    Netflix uses these exact metrics to measure recommendation quality:
    - Precision@K: Of the recommendations we made, how many were actually good?
    - Recall@K: Of all the good movies, how many did we recommend?
    - NDCG@K: Are we ranking the best movies highest?
    - Coverage: Are we recommending diverse content?
    - Novelty: Are we showing users new things they haven't seen?
    """
    
    def __init__(self):
        self.recommender = HybridRecommender()
        self.movies_df = None
        self.synthetic_ratings = None
        
    def generate_synthetic_user_data(self, n_users: int = 100) -> pd.DataFrame:
        """
        Create synthetic user viewing data for evaluation
        
        Production note: In real systems, you'd use actual user interaction data
        (views, ratings, clicks, watch time, etc.)
        """
        print(f"🧪 Generating synthetic user data for {n_users} users...")
        
        # Load movie data
        self.movies_df = pd.read_csv('data/raw/movies_popular.csv')
        movie_ids = self.movies_df['movie_id'].tolist()
        movie_titles = self.movies_df['title'].tolist()
        
        # Create realistic user-movie interactions
        user_data = []
        
        for user_id in range(1, n_users + 1):
            # Each user watches between 5-20 movies
            n_movies_watched = random.randint(5, 20)
            
            # Users have genre preferences (realistic behavior)
            user_preferred_genres = random.sample([
                'Action', 'Adventure', 'Comedy', 'Drama', 'Thriller', 
                'Science Fiction', 'Horror', 'Romance'
            ], k=random.randint(2, 4))
            
            # Select movies based on preferences (80%) and random discovery (20%)
            watched_movies = []
            
            for _ in range(n_movies_watched):
                if random.random() < 0.8 and user_preferred_genres:
                    # Prefer movies from liked genres
                    genre_movies = []
                    for _, movie in self.movies_df.iterrows():
                        try:
                            movie_genres = movie['genres']
                            if isinstance(movie_genres, str):
                                movie_genres = eval(movie_genres)
                            
                            if any(genre in movie_genres for genre in user_preferred_genres):
                                genre_movies.append(movie)
                        except:
                            continue
                    
                    if genre_movies:
                        movie = random.choice(genre_movies)
                        watched_movies.append(movie['title'])
                else:
                    # Random discovery
                    movie = random.choice(movie_titles)
                    watched_movies.append(movie)
            
            # Remove duplicates
            watched_movies = list(set(watched_movies))
            
            # Generate ratings (1-5 scale, biased towards higher ratings)
            for movie_title in watched_movies:
                # Users tend to rate movies they finish higher
                if random.random() < 0.7:  # 70% positive ratings
                    rating = random.choice([4, 5])
                else:
                    rating = random.choice([1, 2, 3])
                
                user_data.append({
                    'user_id': user_id,
                    'movie_title': movie_title,
                    'rating': rating,
                    'preferred_genres': user_preferred_genres
                })
        
        self.synthetic_ratings = pd.DataFrame(user_data)
        
        print(f"✅ Generated {len(self.synthetic_ratings)} user-movie interactions")
        print(f"   Average ratings per user: {len(self.synthetic_ratings) / n_users:.1f}")
        
        return self.synthetic_ratings
    
    def create_train_test_split(self, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split user data into training and test sets
        
        Production strategy: For each user, hide some of their ratings
        and see if we can predict them
        """
        train_data = []
        test_data = []
        
        for user_id in self.synthetic_ratings['user_id'].unique():
            user_ratings = self.synthetic_ratings[
                self.synthetic_ratings['user_id'] == user_id
            ].copy()
            
            # For each user, put some ratings in test set
            n_test = max(1, int(len(user_ratings) * test_ratio))
            test_indices = random.sample(range(len(user_ratings)), n_test)
            
            for idx, (_, rating) in enumerate(user_ratings.iterrows()):
                if idx in test_indices:
                    test_data.append(rating)
                else:
                    train_data.append(rating)
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        print(f"📊 Data split created:")
        print(f"   Training interactions: {len(train_df)}")
        print(f"   Test interactions: {len(test_df)}")
        
        return train_df, test_df
    
    def calculate_precision_at_k(self, recommended: List[str], 
                                relevant: List[str], k: int = 10) -> float:
        """
        Precision@K: Of the K recommendations, how many were actually relevant?
        
        Formula: (Relevant items in top-K recommendations) / K
        
        Production significance: Higher precision = users click/watch more recommendations
        """
        if not recommended or k == 0:
            return 0.0
        
        top_k = recommended[:k]
        relevant_in_top_k = len([item for item in top_k if item in relevant])
        
        return relevant_in_top_k / len(top_k)
    
    def calculate_recall_at_k(self, recommended: List[str], 
                             relevant: List[str], k: int = 10) -> float:
        """
        Recall@K: Of all relevant items, how many did we recommend in top-K?
        
        Formula: (Relevant items in top-K recommendations) / (Total relevant items)
        
        Production significance: Higher recall = we're not missing good content
        """
        if not relevant:
            return 0.0
        
        top_k = recommended[:k]
        relevant_in_top_k = len([item for item in top_k if item in relevant])
        
        return relevant_in_top_k / len(relevant)
    
    def calculate_ndcg_at_k(self, recommended: List[str], 
                           relevant_scores: Dict[str, float], k: int = 10) -> float:
        """
        NDCG@K (Normalized Discounted Cumulative Gain)
        
        Why NDCG? It considers both relevance AND ranking position.
        A highly relevant item at position 1 is much better than at position 10.
        
        This is Netflix's holy grail metric!
        """
        if not recommended or not relevant_scores:
            return 0.0
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, movie in enumerate(recommended[:k]):
            if movie in relevant_scores:
                relevance = relevant_scores[movie]
                # Discount factor: log2(position + 1)
                discount = np.log2(i + 2)  # +2 because positions start at 1
                dcg += relevance / discount
        
        # Calculate IDCG (Ideal DCG) - best possible ordering
        ideal_scores = sorted(relevant_scores.values(), reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_scores[:k]):
            discount = np.log2(i + 2)
            idcg += relevance / discount
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_user(self, user_id: int, train_data: pd.DataFrame, 
                     test_data: pd.DataFrame, k: int = 10) -> Dict:
        """Evaluate recommendations for a single user"""
        
        # Get user's training data (what they've watched)
        user_train = train_data[train_data['user_id'] == user_id]
        liked_movies = user_train[user_train['rating'] >= 4]['movie_title'].tolist()
        preferred_genres = user_train['preferred_genres'].iloc[0] if len(user_train) > 0 else []
        
        # Get user's test data (what we're trying to predict)
        user_test = test_data[test_data['user_id'] == user_id]
        relevant_movies = user_test[user_test['rating'] >= 4]['movie_title'].tolist()
        
        # Create relevance scores (rating - 2.5, so 4+ ratings become positive)
        relevant_scores = {}
        for _, row in user_test.iterrows():
            relevant_scores[row['movie_title']] = max(0, row['rating'] - 2.5)
        
        if not relevant_movies:
            return None  # Skip users with no positive test ratings
        
        # Generate recommendations
        user_profile = {
            'user_id': user_id,
            'liked_movies': liked_movies,
            'preferred_genres': preferred_genres
        }
        
        try:
            recommendations = self.recommender.get_smart_recommendations(user_profile)
            
            # Extract movie titles from recommendations
            recommended_movies = []
            for section in recommendations['sections']:
                for movie in section['movies']:
                    recommended_movies.append(movie['title'])
            
            # Calculate metrics
            precision = self.calculate_precision_at_k(recommended_movies, relevant_movies, k)
            recall = self.calculate_recall_at_k(recommended_movies, relevant_movies, k)
            ndcg = self.calculate_ndcg_at_k(recommended_movies, relevant_scores, k)
            
            return {
                'user_id': user_id,
                f'precision@{k}': precision,
                f'recall@{k}': recall,
                f'ndcg@{k}': ndcg,
                'n_relevant': len(relevant_movies),
                'n_recommended': len(recommended_movies),
                'strategy': recommendations['strategy']
            }
            
        except Exception as e:
            print(f"⚠️  Error evaluating user {user_id}: {e}")
            return None
    
    def run_full_evaluation(self, k: int = 10) -> Dict:
        """
        Run complete evaluation on the recommendation system
        
        This is what Netflix runs every day to monitor system performance
        """
        print("🚀 Starting comprehensive evaluation...")
        
        # Setup
        self.recommender.train()
        
        # Generate test data
        user_data = self.generate_synthetic_user_data(n_users=50)  # Start small
        train_data, test_data = self.create_train_test_split()
        
        # Evaluate all users
        user_results = []
        users_to_evaluate = test_data['user_id'].unique()
        
        print(f"📊 Evaluating {len(users_to_evaluate)} users...")
        
        for user_id in users_to_evaluate:
            result = self.evaluate_user(user_id, train_data, test_data, k)
            if result:
                user_results.append(result)
        
        if not user_results:
            print("❌ No valid evaluations completed")
            return {}
        
        # Calculate aggregate metrics
        results_df = pd.DataFrame(user_results)
        
        aggregate_metrics = {
            'total_users_evaluated': len(user_results),
            f'avg_precision@{k}': results_df[f'precision@{k}'].mean(),
            f'avg_recall@{k}': results_df[f'recall@{k}'].mean(),
            f'avg_ndcg@{k}': results_df[f'ndcg@{k}'].mean(),
            'strategy_distribution': results_df['strategy'].value_counts().to_dict()
        }
        
        # Print results
        print(f"\n🎯 EVALUATION RESULTS (K={k})")
        print("="*50)
        print(f"Users evaluated: {aggregate_metrics['total_users_evaluated']}")
        print(f"Average Precision@{k}: {aggregate_metrics[f'avg_precision@{k}']:.3f}")
        print(f"Average Recall@{k}: {aggregate_metrics[f'avg_recall@{k}']:.3f}")
        print(f"Average NDCG@{k}: {aggregate_metrics[f'avg_ndcg@{k}']:.3f}")
        
        print(f"\n📈 Strategy Usage:")
        for strategy, count in aggregate_metrics['strategy_distribution'].items():
            print(f"  {strategy}: {count} users")
        
        return {
            'aggregate_metrics': aggregate_metrics,
            'detailed_results': user_results
        }

def demo_evaluation():
    """Demo our evaluation framework"""
    evaluator = RecommendationEvaluator()
    
    print("🔬 NETFLIX-STYLE RECOMMENDATION EVALUATION")
    print("="*60)
    
    # Run evaluation
    results = evaluator.run_full_evaluation(k=5)
    
    if results:
        print(f"\n💡 INTERPRETATION:")
        precision = results['aggregate_metrics']['avg_precision@5']
        recall = results['aggregate_metrics']['avg_recall@5']
        ndcg = results['aggregate_metrics']['avg_ndcg@5']
        
        if precision > 0.3:
            print("✅ Good precision - users are likely to engage with recommendations")
        else:
            print("⚠️  Low precision - many irrelevant recommendations")
            
        if recall > 0.2:
            print("✅ Good recall - capturing most of what users would like")
        else:
            print("⚠️  Low recall - missing potential good recommendations")
            
        if ndcg > 0.5:
            print("✅ Good ranking - most relevant items appear at the top")
        else:
            print("⚠️  Poor ranking - relevant items buried in recommendations")

if __name__ == "__main__":
    demo_evaluation()