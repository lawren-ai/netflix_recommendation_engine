from models.recommendation_evaluator import RecommendationEvaluator
from models.improved_hybrid_recommender import ImprovedHybridRecommender

# Test our improvements
print("🧪 TESTING IMPROVEMENTS")
print("="*50)

# Create evaluator with improved system
evaluator = RecommendationEvaluator()
evaluator.recommender = ImprovedHybridRecommender(
    similarity_threshold=0.3,  # Reasonable threshold
    rating_threshold=6.0       # Good quality movies only
)

print("🔄 Running evaluation with IMPROVED system...")
results = evaluator.run_full_evaluation(k=5)

if results:
    print(f"\n📊 IMPROVEMENT COMPARISON:")
    print("="*40)
    print("BEFORE (Original System):")
    print("  Precision@5: 0.028 (2.8%)")
    print("  Recall@5:    0.101 (10.1%)")  
    print("  NDCG@5:      0.085 (8.5%)")
    
    print(f"\nAFTER (Improved System):")
    precision = results['aggregate_metrics']['avg_precision@5']
    recall = results['aggregate_metrics']['avg_recall@5'] 
    ndcg = results['aggregate_metrics']['avg_ndcg@5']
    
    print(f"  Precision@5: {precision:.3f} ({precision*100:.1f}%)")
    print(f"  Recall@5:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"  NDCG@5:      {ndcg:.3f} ({ndcg*100:.1f}%)")
    
    # Calculate improvements
    precision_improvement = (precision - 0.028) / 0.028 * 100
    recall_improvement = (recall - 0.101) / 0.101 * 100
    ndcg_improvement = (ndcg - 0.085) / 0.085 * 100
    
    print(f"\n🚀 IMPROVEMENTS:")
    print(f"  Precision: {precision_improvement:+.1f}%")
    print(f"  Recall:    {recall_improvement:+.1f}%") 
    print(f"  NDCG:      {ndcg_improvement:+.1f}%")
    
    if precision > 0.05:
        print("\n✅ Significant precision improvement!")
    if recall > 0.15:
        print("✅ Good recall improvement!")
    if ndcg > 0.15:
        print("✅ Better ranking quality!")

print(f"\n💡 Next steps to improve further:")
print("  • Add collaborative filtering")
print("  • Use more sophisticated ML models") 
print("  • Incorporate user behavior patterns")
print("  • A/B test different parameter settings")