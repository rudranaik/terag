#!/usr/bin/env python3
"""
TERAG Graph Merger

Merges multiple TERAG graphs and applies deduplication to handle overlapping entities.
Supports incremental document ingestion by combining existing graphs with new ones.
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict

from graph_builder import TERAGGraph
from entity_deduplicator import deduplicate_graph_entities
from entity_merger import apply_deduplication_to_graph


def merge_terag_graphs(graph_paths: List[str], verbose: bool = True) -> TERAGGraph:
    """
    Merge multiple TERAG graphs into a single combined graph
    
    Args:
        graph_paths: List of paths to TERAG graph JSON files
        verbose: Whether to print progress information
        
    Returns:
        Combined TERAGGraph with all passages, concepts, and edges
    """
    if verbose:
        print(f"🔗 Merging {len(graph_paths)} TERAG graphs...")
    
    combined_graph = TERAGGraph()
    total_passages = 0
    total_concepts = 0
    total_edges = 0
    
    for i, graph_path in enumerate(graph_paths):
        if verbose:
            print(f"   📖 Loading graph {i+1}: {graph_path}")
        
        try:
            graph = TERAGGraph.load_from_file(graph_path)
            graph_stats = graph.get_statistics()
            
            if verbose:
                print(f"      📊 {graph_stats['num_passages']} passages, "
                      f"{graph_stats['num_concepts']} concepts, "
                      f"{graph_stats['num_edges']} edges")
            
            # Add all passages from this graph
            passages_added = 0
            for passage in graph.passages.values():
                if passage.passage_id not in combined_graph.passages:
                    combined_graph.add_passage(passage)
                    passages_added += 1
                elif verbose:
                    print(f"      ⚠️  Skipping duplicate passage: {passage.passage_id}")
            
            # Add all concepts from this graph (auto-merges by concept_id)
            concepts_before = len(combined_graph.concepts)
            for concept in graph.concepts.values():
                combined_graph.add_concept(concept)  # This handles merging automatically
            concepts_added = len(combined_graph.concepts) - concepts_before
            
            # Add all edges from this graph
            edges_added = 0
            for passage_id, concepts in graph.passage_to_concepts.items():
                for concept_id, weight in concepts.items():
                    if passage_id in combined_graph.passages and concept_id in combined_graph.concepts:
                        # Only add if we haven't already added this edge
                        existing_weight = combined_graph.passage_to_concepts.get(passage_id, {}).get(concept_id)
                        if existing_weight is None:
                            combined_graph.add_edge(passage_id, concept_id, weight)
                            edges_added += 1
                        else:
                            # Update weight if new weight is higher
                            if weight > existing_weight:
                                combined_graph.add_edge(passage_id, concept_id, weight)
            
            if verbose:
                print(f"      ✅ Added {passages_added} passages, {concepts_added} new concepts, {edges_added} edges")
            
            total_passages += passages_added
            total_concepts += concepts_added
            total_edges += edges_added
            
        except Exception as e:
            print(f"❌ Error loading graph {graph_path}: {e}")
            continue
    
    if verbose:
        final_stats = combined_graph.get_statistics()
        print(f"\n📊 Combined Graph Statistics:")
        print(f"   📄 Total passages: {final_stats['num_passages']}")
        print(f"   🏷️  Total concepts: {final_stats['num_concepts']}")
        print(f"   🔗 Total edges: {final_stats['num_edges']}")
        print(f"   📈 Avg concepts per passage: {final_stats['avg_concepts_per_passage']:.2f}")
    
    return combined_graph


def main():
    parser = argparse.ArgumentParser(description="Merge multiple TERAG graphs with deduplication")
    parser.add_argument(
        "--graphs",
        nargs="+",
        required=True,
        help="Paths to TERAG graph JSON files to merge"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="merged_graph_results",
        help="Directory to save merged graph and deduplication results"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="merged_terag_graph",
        help="Base name for output files"
    )
    parser.add_argument(
        "--skip-deduplication",
        action="store_true",
        help="Skip entity deduplication (faster but may have duplicates)"
    )
    parser.add_argument(
        "--string-threshold",
        type=float,
        default=0.8,
        help="String similarity threshold for deduplication (0.0-1.0)"
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.85,
        help="Embedding similarity threshold for deduplication (0.0-1.0)"
    )
    parser.add_argument(
        "--graph-threshold",
        type=float,
        default=0.6,
        help="Graph co-occurrence threshold for deduplication (0.0-1.0)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding-based deduplication (faster, but less accurate)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    missing_files = []
    for graph_path in args.graphs:
        if not Path(graph_path).exists():
            missing_files.append(graph_path)
    
    if missing_files:
        print("❌ Missing graph files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Make sure all graph files exist before merging.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔗 TERAG GRAPH MERGER")
    print("=" * 70)
    print(f"📁 Input graphs: {len(args.graphs)} files")
    for i, graph_path in enumerate(args.graphs, 1):
        print(f"   {i}. {graph_path}")
    print(f"📁 Output dir: {output_dir}")
    print(f"📁 Output name: {args.output_name}")
    print(f"⚙️  Skip deduplication: {args.skip_deduplication}")
    if not args.skip_deduplication:
        print(f"⚙️  Dedup thresholds: string={args.string_threshold}, "
              f"embedding={args.embedding_threshold}, graph={args.graph_threshold}")
    print()
    
    try:
        start_time = time.time()
        
        # Step 1: Merge all graphs
        print("🔗 STEP 1: MERGING GRAPHS")
        print("-" * 50)
        
        combined_graph = merge_terag_graphs(args.graphs, verbose=True)
        merge_time = time.time() - start_time
        
        # Save raw merged graph
        raw_merged_file = output_dir / f"{args.output_name}_raw.json"
        combined_graph.save_to_file(str(raw_merged_file))
        print(f"\n💾 Raw merged graph saved: {raw_merged_file}")
        
        final_graph = combined_graph
        
        # Step 2: Deduplication (optional)
        if not args.skip_deduplication:
            print(f"\n🔍 STEP 2: ENTITY DEDUPLICATION")
            print("-" * 50)
            
            dedup_start = time.time()
            
            # Configure deduplication parameters
            deduplicator_kwargs = {
                "string_similarity_threshold": args.string_threshold,
                "embedding_similarity_threshold": args.embedding_threshold,
                "graph_similarity_threshold": args.graph_threshold
            }
            
            if args.skip_embeddings:
                print("⚠️  Skipping embeddings - using string + graph similarity only")
                deduplicator_kwargs["embedding_similarity_threshold"] = 0.99
            
            # Run deduplication
            entity_mapping, clusters = deduplicate_graph_entities(combined_graph, **deduplicator_kwargs)
            
            if entity_mapping:
                print(f"\n📊 Deduplication Results:")
                print(f"   Duplicate entities found: {len(entity_mapping)}")
                print(f"   Entity clusters: {len(clusters)}")
                
                # Show top clusters
                if clusters:
                    print(f"\n🎯 Top Entity Clusters:")
                    sorted_clusters = sorted(clusters, key=lambda c: len(c.duplicate_entities), reverse=True)
                    for i, cluster in enumerate(sorted_clusters[:5]):
                        print(f"   {i+1}. '{cluster.canonical_entity}' <- {list(cluster.duplicate_entities)} "
                              f"(confidence: {cluster.confidence_score:.3f})")
                
                # Apply deduplication
                print(f"\n🔄 Applying entity merging...")
                final_graph = apply_deduplication_to_graph(combined_graph, entity_mapping)
                
                # Save entity mapping
                mapping_file = output_dir / f"{args.output_name}_entity_mapping.json"
                with open(mapping_file, 'w') as f:
                    json.dump(entity_mapping, f, indent=2)
                print(f"💾 Entity mapping saved: {mapping_file}")
                
            else:
                print("✨ No duplicate entities found! Graph is already clean.")
            
            dedup_time = time.time() - dedup_start
        else:
            print(f"\n⏭️  STEP 2: SKIPPING DEDUPLICATION")
            dedup_time = 0
        
        # Save final graph
        final_graph_file = output_dir / f"{args.output_name}.json"
        final_graph.save_to_file(str(final_graph_file))
        
        # Generate final statistics
        final_stats = final_graph.get_statistics()
        total_time = time.time() - start_time
        
        print(f"\n📊 FINAL RESULTS")
        print("=" * 70)
        print(f"✅ Merge completed successfully!")
        print(f"📄 Final graph: {final_stats['num_passages']} passages, "
              f"{final_stats['num_concepts']} concepts, "
              f"{final_stats['num_edges']} edges")
        print(f"⏱️  Total time: {total_time:.2f}s (merge: {merge_time:.2f}s, dedup: {dedup_time:.2f}s)")
        print(f"💾 Final graph saved: {final_graph_file}")
        
        # Save processing summary
        summary_file = output_dir / f"{args.output_name}_merge_summary.json"
        summary = {
            "input_graphs": args.graphs,
            "output_file": str(final_graph_file),
            "processing_time": {
                "total_seconds": total_time,
                "merge_seconds": merge_time,
                "deduplication_seconds": dedup_time
            },
            "final_statistics": final_stats,
            "deduplication_applied": not args.skip_deduplication,
            "deduplication_settings": {
                "string_threshold": args.string_threshold,
                "embedding_threshold": args.embedding_threshold,
                "graph_threshold": args.graph_threshold,
                "skip_embeddings": args.skip_embeddings
            } if not args.skip_deduplication else None
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Processing summary: {summary_file}")
        
        print(f"\n🎉 Graph merging complete! Use the merged graph for retrieval:")
        print(f"   python retrieval_demo.py --graph-file {final_graph_file}")
        
    except Exception as e:
        print(f"❌ Error during graph merging: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()