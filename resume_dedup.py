#!/usr/bin/env python3
"""
Resume deduplication from existing graph
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
    
    print("🔍 RESUMING DEDUPLICATION")
    print("=" * 50)
    
    try:
        print("📖 Loading graph...")
        graph = TERAGGraph.load_from_file(graph_file)
        original_stats = graph.get_statistics()
        
        print(f"📊 Original: {original_stats['num_concepts']} concepts, {original_stats['num_passages']} passages")
        print(f"🚀 Running FASTER deduplication with relaxed settings...")
        
        # Use embedding settings (will be slower but thorough)
        dedup_settings = {
            "string_similarity_threshold": 0.75,  # Moderate string matching
            "embedding_similarity_threshold": 0.82,  # Enable embeddings  
            "graph_similarity_threshold": 0.6  # Standard validation
        }
        
        print(f"⚡ Fast settings: string=0.9 (high), embeddings=disabled, graph=0.7")
        
        start_time = time.time()
        entity_mapping, clusters = deduplicate_graph_entities(graph, **dedup_settings)
        dedup_time = time.time() - start_time
        
        print(f"⏱️  Deduplication completed in {dedup_time:.2f} seconds")
        
        if entity_mapping:
            print(f"🎯 Found {len(entity_mapping)} duplicate entities in {len(clusters)} clusters")
            deduplicated_graph = apply_deduplication_to_graph(graph, entity_mapping)
            
            # Show improvement
            final_stats = deduplicated_graph.get_statistics()
            concepts_reduced = original_stats['num_concepts'] - final_stats['num_concepts']
            reduction_pct = (concepts_reduced / original_stats['num_concepts']) * 100
            print(f"✅ Reduced concepts by {concepts_reduced} ({reduction_pct:.1f}%)")
        else:
            print("✨ No duplicates found with these settings")
            deduplicated_graph = graph
        
        # Save final graph
        final_file = f"{output_dir}/terag_graph.json"
        deduplicated_graph.save_to_file(final_file)
        print(f"💾 Final graph saved: {final_file}")
        
        print(f"\n🎉 READY FOR RETRIEVAL!")
        print(f"python retrieval_demo.py --graph-file {final_file} --interactive")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()