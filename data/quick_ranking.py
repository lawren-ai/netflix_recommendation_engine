from models.improved_content_recommender import ImprovedContentRecommender
import numpy as np

class RankingOptimizedRecommender(ImprovedContentRecommender):
    """
    Final optimization: Smart ranking that considers both similarity AND quality
    """
    
    def get_improved_recommendations(self, movie_title: str, n_recommendations: int = 10):
        """Enhanced recommendations with optimized ranking"""
        
        # Get the base recommendations
        base_recs = super().get_improved_recommendations(movie_title, n_recommendations * 2)  # Get more candidates
        
        if not base_recs:
            return []
        
        # Re-rank using combined score
        for rec in base_recs:
            # Combined ranking score:
            # 60% similarity + 30% normalized rating + 10% popularity boost
            normalized_rating = rec['vote_average'] / 10.0
            popularity_boost = min(1.0, rec['vote_count'] / 1000.0)  # Cap at 1000 votes
            
            rec['combined_score'] = (
                0.6 * rec['similarity_score'] +
                0.3 * normalized_rating +
                0.1 * popularity_boost
            )
        
        # Sort by combined score
        base_recs.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top recommendations with ranking explanation
        final_recs = base_recs[:n_recommendations]
        
        for rec in final_recs:
            rec['ranking_reason'] = f"Combined score: {rec['combined_score']:.3f} (similarity + rating + popularity)"
        
        print(f"✅ Optimized ranking: Top movie scored {final_recs[0]['combined_score']:.3f}")
        
        return final_recs

# Test the ranking optimization
def test_ranking_optimization():
    print("🎯 TESTING RANKING OPTIMIZATION")
    print("="*50)
    
    recommender = RankingOptimizedRecommender(
        min_similarity_threshold=0.3,
        min_rating_threshold=6.0
    )
    recommender.train()
    
    recommendations = recommender.get_improved_recommendations("Superman", 5)
    
    print(f"\n🏆 OPTIMIZED RANKING RESULTS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']}")
        print(f"   {rec['ranking_reason']}")
        print(f"   Similarity: {rec['similarity_score']:.3f} | Rating: {rec['vote_average']:.1f}")

if __name__ == "__main__":
    test_ranking_optimization()