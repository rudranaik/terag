#!/usr/bin/env python3
"""
Smart deduplication with embedding optimizations
"""

import json
import time
from pathlib import Path
from graph_builder import TERAGGraph
from entity_deduplicator import deduplicate_graph_entities
from entity_merger import apply_deduplication_to_graph

def main():
    # Your current graph file
    graph_file = "terag_data/graph_temp/terag_graph.json"
    output_dir = "terag_data"
    
    print("🧠 SMART DEDUPLICATION WITH EMBEDDINGS")
    print("=" * 60)
    
    try:
        print("📖 Loading graph...")
        graph = TERAGGraph.load_from_file(graph_file)
        original_stats = graph.get_statistics()
        
        print(f"📊 Original: {original_stats['num_concepts']} concepts, {original_stats['num_passages']} passages")
        
        # Smart approach: Start with string matching, then selective embedding
        print(f"\n🚀 PHASE 1: String-based deduplication (fast)")
        
        # Phase 1: High string threshold, no embeddings
        phase1_settings = {
            "string_similarity_threshold": 0.85,  # Good string matching
            "embedding_similarity_threshold": 0.99,  # Disabled
            "graph_similarity_threshold": 0.6
        }
        
        start_time = time.time()
        entity_mapping_1, clusters_1 = deduplicate_graph_entities(graph, **phase1_settings)
        phase1_time = time.time() - start_time
        
        print(f"⏱️  Phase 1 completed in {phase1_time:.2f} seconds")
        print(f"🎯 Found {len(entity_mapping_1)} string-based duplicates")
        
        if entity_mapping_1:
            # Apply phase 1 deduplication
            graph = apply_deduplication_to_graph(graph, entity_mapping_1)
            phase1_stats = graph.get_statistics()
            concepts_reduced_1 = original_stats['num_concepts'] - phase1_stats['num_concepts']
            print(f"✅ Reduced concepts by {concepts_reduced_1} in phase 1")
            print(f"📊 Remaining: {phase1_stats['num_concepts']} concepts")
        
        # Phase 2: Embedding-based on remaining concepts
        print(f"\n🧠 PHASE 2: Embedding-based deduplication")
        print(f"   Processing {graph.get_statistics()['num_concepts']} remaining concepts...")
        
        phase2_settings = {
            "string_similarity_threshold": 0.7,   # Lower string threshold
            "embedding_similarity_threshold": 0.82,  # Good embedding threshold  
            "graph_similarity_threshold": 0.6
        }
        
        start_time = time.time()
        entity_mapping_2, clusters_2 = deduplicate_graph_entities(graph, **phase2_settings)
        phase2_time = time.time() - start_time
        
        print(f"⏱️  Phase 2 completed in {phase2_time:.2f} seconds")
        print(f"🎯 Found {len(entity_mapping_2)} semantic duplicates")
        
        if entity_mapping_2:
            # Apply phase 2 deduplication
            final_graph = apply_deduplication_to_graph(graph, entity_mapping_2)
            
            # Show final results
            final_stats = final_graph.get_statistics()
            total_concepts_reduced = original_stats['num_concepts'] - final_stats['num_concepts']
            total_reduction_pct = (total_concepts_reduced / original_stats['num_concepts']) * 100
            
            print(f"✅ Total concepts reduced: {total_concepts_reduced} ({total_reduction_pct:.1f}%)")
            print(f"📊 Final: {final_stats['num_concepts']} concepts, {final_stats['num_edges']} edges")
            
            # Show some examples
            if clusters_2:
                print(f"\n📋 Semantic duplicate examples:")
                sorted_clusters = sorted(clusters_2, key=lambda c: len(c.duplicate_entities), reverse=True)
                for i, cluster in enumerate(sorted_clusters[:3]):
                    print(f"   {i+1}. '{cluster.canonical_entity}' ← {list(cluster.duplicate_entities)}")
        else:
            print("✨ No additional semantic duplicates found")
            final_graph = graph
        
        # Save final graph
        final_file = f"{output_dir}/terag_graph.json"
        final_graph.save_to_file(final_file)
        
        total_time = phase1_time + phase2_time
        print(f"\n🎉 DEDUPLICATION COMPLETE!")
        print(f"⏱️  Total time: {total_time:.2f}s (Phase 1: {phase1_time:.2f}s, Phase 2: {phase2_time:.2f}s)")
        print(f"💾 Final graph saved: {final_file}")
        print(f"\n🔍 Ready for retrieval:")
        print(f"python retrieval_demo.py --graph-file {final_file} --interactive")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()