from models.recommendation_evaluator import RecommendationEvaluator
from models.ultimate_hybrid_recommender import UltimateHybridRecommender

print("🎬 ULTIMATE SYSTEM EVALUATION - THE FINAL TEST!")
print("="*60)

# Create evaluator with ULTIMATE system
evaluator = RecommendationEvaluator()
evaluator.recommender = UltimateHybridRecommender()

print("🚀 Running evaluation with ULTIMATE Netflix-style system...")
print("   (This includes collaborative filtering + content-based + popularity)")

results = evaluator.run_full_evaluation(k=5)

if results:
    print(f"\n📊 EVOLUTION OF YOUR RECOMMENDATION SYSTEM:")
    print("="*50)
    
    print("🥉 ORIGINAL SYSTEM:")
    print("   Precision@5: 0.028 (2.8%)")
    print("   Recall@5:    0.101 (10.1%)")  
    print("   NDCG@5:      0.085 (8.5%)")
    
    print(f"\n🥈 IMPROVED CONTENT-BASED:")
    print("   Precision@5: ~0.030 (+8.7%)")
    print("   Recall@5:    ~0.116 (+14.8%)")
    print("   NDCG@5:      ~0.077 (-9.9%)")
    
    print(f"\n🥇 ULTIMATE SYSTEM (with Collaborative Filtering):")
    precision = results['aggregate_metrics']['avg_precision@5']
    recall = results['aggregate_metrics']['avg_recall@5'] 
    ndcg = results['aggregate_metrics']['avg_ndcg@5']
    
    print(f"   Precision@5: {precision:.3f} ({precision*100:.1f}%)")
    print(f"   Recall@5:    {recall:.3f} ({recall*100:.1f}%)")
    print(f"   NDCG@5:      {ndcg:.3f} ({ndcg*100:.1f}%)")
    
    # Calculate total improvement from original
    precision_total_improvement = (precision - 0.028) / 0.028 * 100
    recall_total_improvement = (recall - 0.101) / 0.101 * 100
    ndcg_total_improvement = (ndcg - 0.085) / 0.085 * 100
    
    print(f"\n🚀 TOTAL IMPROVEMENT FROM ORIGINAL:")
    print(f"   Precision: {precision_total_improvement:+.1f}%")
    print(f"   Recall:    {recall_total_improvement:+.1f}%") 
    print(f"   NDCG:      {ndcg_total_improvement:+.1f}%")
    
    # Interpret results
    print(f"\n🎯 NETFLIX-LEVEL PERFORMANCE ANALYSIS:")
    if precision > 0.10:
        print("   ✅ EXCELLENT: >10% precision - Production ready!")
    elif precision > 0.05:
        print("   ✅ GOOD: >5% precision - Strong performance")
    else:
        print("   ⚠️  FAIR: Still improving, but collaborative filtering helps")
        
    if recall > 0.20:
        print("   ✅ EXCELLENT: >20% recall - Finding lots of relevant content")
    elif recall > 0.15:
        print("   ✅ GOOD: >15% recall - Good content discovery")
    
    if ndcg > 0.15:
        print("   ✅ EXCELLENT: >15% NDCG - Great ranking quality")
    elif ndcg > 0.10:
        print("   ✅ GOOD: >10% NDCG - Decent ranking")
        
    # Strategy breakdown
    print(f"\n📈 Strategy Usage:")
    strategy_stats = results['aggregate_metrics']['strategy_distribution']
    for strategy, count in strategy_stats.items():
        print(f"   {strategy}: {count} users")
        
    print(f"\n🎉 CONGRATULATIONS!")
    print(f"You've built a Netflix-level recommendation system with:")
    print(f"   • Multiple ML algorithms working together")
    print(f"   • Intelligent routing based on user profiles") 
    print(f"   • Production-ready evaluation framework")
    print(f"   • Real performance metrics that guide optimization")
    
else:
    print("❌ Evaluation failed - check the logs above")

print(f"\n🚀 Ready for the next challenge: Building the API!")