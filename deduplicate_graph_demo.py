#!/usr/bin/env python3
"""
TERAG Entity Deduplication Demo

Runs 3-phase entity deduplication on an existing TERAG graph:
1. String similarity matching
2. Embedding-based semantic similarity
3. Graph-based validation

Then merges entities while preserving graph relationships.
"""

import json
import argparse
from pathlib import Path
from graph_builder import TERAGGraph
from entity_deduplicator import EntityDeduplicator, deduplicate_graph_entities
from entity_merger import apply_deduplication_to_graph, create_deduplication_report


def main():
    parser = argparse.ArgumentParser(description="Deduplicate entities in TERAG graph")
    parser.add_argument(
        "--graph-file",
        type=str,
        default="graph_results/terag_graph.json", 
        help="Path to TERAG graph JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dedup_results",
        help="Directory to save deduplicated graph and reports"
    )
    parser.add_argument(
        "--string-threshold",
        type=float,
        default=0.8,
        help="String similarity threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--embedding-threshold", 
        type=float,
        default=0.85,
        help="Embedding similarity threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--graph-threshold",
        type=float,
        default=0.6,
        help="Graph co-occurrence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding-based similarity (faster, but less accurate)"
    )
    
    args = parser.parse_args()
    
    # Check input file
    graph_file = Path(args.graph_file)
    if not graph_file.exists():
        print(f"❌ Graph file not found: {graph_file}")
        print(f"   Run graph building first with: python build_graph_demo.py")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 TERAG ENTITY DEDUPLICATION")
    print("=" * 60)
    print(f"📁 Input graph: {graph_file}")
    print(f"📁 Output dir: {output_dir}")
    print(f"⚙️  String threshold: {args.string_threshold}")
    print(f"⚙️  Embedding threshold: {args.embedding_threshold}")
    print(f"⚙️  Graph threshold: {args.graph_threshold}")
    print(f"⚙️  Skip embeddings: {args.skip_embeddings}")
    print()
    
    try:
        # Load graph
        print(f"📖 Loading TERAG graph...")
        original_graph = TERAGGraph.load_from_file(str(graph_file))
        original_stats = original_graph.get_statistics()
        
        print(f"   Loaded graph: {original_stats['num_passages']} passages, "
              f"{original_stats['num_concepts']} concepts, "
              f"{original_stats['num_edges']} edges")
        
        # Initialize deduplicator
        deduplicator_kwargs = {
            "string_similarity_threshold": args.string_threshold,
            "embedding_similarity_threshold": args.embedding_threshold,
            "graph_similarity_threshold": args.graph_threshold
        }
        
        if args.skip_embeddings:
            print("⚠️  Skipping embeddings - using string + graph similarity only")
            # Set embedding threshold very high to effectively disable it
            deduplicator_kwargs["embedding_similarity_threshold"] = 0.99
        
        # Run deduplication
        print(f"\n🔍 Running 3-phase entity deduplication...")
        entity_mapping, clusters = deduplicate_graph_entities(original_graph, **deduplicator_kwargs)
        
        if not entity_mapping:
            print("✨ No duplicate entities found! Graph is already clean.")
            return
        
        print(f"\n📊 Deduplication Results:")
        print(f"   Duplicate entities found: {len(entity_mapping)}")
        print(f"   Entity clusters: {len(clusters)}")
        
        # Show top clusters
        print(f"\n🎯 Top Entity Clusters:")
        sorted_clusters = sorted(clusters, key=lambda c: len(c.duplicate_entities), reverse=True)
        for i, cluster in enumerate(sorted_clusters[:5]):
            print(f"   {i+1}. '{cluster.canonical_entity}' <- {list(cluster.duplicate_entities)} "
                  f"(confidence: {cluster.confidence_score:.3f})")
        
        # Apply deduplication
        print(f"\n🔄 Merging duplicate entities...")
        merged_graph = apply_deduplication_to_graph(original_graph, entity_mapping)
        merged_stats = merged_graph.get_statistics()
        
        print(f"   Merged graph: {merged_stats['num_passages']} passages, "
              f"{merged_stats['num_concepts']} concepts, "
              f"{merged_stats['num_edges']} edges")
        
        # Calculate improvements
        concept_reduction = original_stats['num_concepts'] - merged_stats['num_concepts']
        reduction_pct = concept_reduction / original_stats['num_concepts'] * 100
        
        print(f"\n📈 Improvements:")
        print(f"   Concepts reduced: {concept_reduction} ({reduction_pct:.1f}%)")
        print(f"   Graph density: {original_stats.get('density', 'N/A')} -> {merged_stats.get('density', 'N/A')}")
        
        # Save results
        print(f"\n💾 Saving results...")
        
        # Save deduplicated graph
        merged_graph_file = output_dir / "deduplicated_terag_graph.json"
        merged_graph.save_to_file(str(merged_graph_file))
        print(f"   Deduplicated graph: {merged_graph_file}")
        
        # Save entity mapping
        mapping_file = output_dir / "entity_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(entity_mapping, f, indent=2, ensure_ascii=False)
        print(f"   Entity mapping: {mapping_file}")
        
        # Save detailed report
        report = create_deduplication_report(original_graph, merged_graph, entity_mapping, clusters)
        report_file = output_dir / "deduplication_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"   Detailed report: {report_file}")
        
        # Save cluster details
        cluster_details = [
            {
                "canonical_entity": cluster.canonical_entity,
                "duplicate_entities": list(cluster.duplicate_entities),
                "confidence_score": cluster.confidence_score,
                "detection_method": cluster.detection_method
            }
            for cluster in clusters
        ]
        
        clusters_file = output_dir / "entity_clusters.json"
        with open(clusters_file, 'w') as f:
            json.dump(cluster_details, f, indent=2, ensure_ascii=False)
        print(f"   Entity clusters: {clusters_file}")
        
        # Export deduplicated graph to visualization formats
        try:
            import networkx as nx
            print(f"\n📊 Exporting deduplicated graph for visualization...")
            
            # Convert to NetworkX
            nx_graph = merged_graph.to_networkx(
                include_node_attributes=True,
                include_edge_weights=True,
                flatten_metadata=True
            )
            
            # Save GraphML
            graphml_file = output_dir / "deduplicated_graph.graphml"
            nx.write_graphml(nx_graph, str(graphml_file))
            print(f"   GraphML: {graphml_file}")
            
            # Save GML  
            gml_file = output_dir / "deduplicated_graph.gml"
            nx.write_gml(nx_graph, str(gml_file))
            print(f"   GML: {gml_file}")
            
        except ImportError:
            print("⚠️  NetworkX not available - skipping visualization exports")
        except Exception as e:
            print(f"⚠️  Export failed: {e}")
        
        # Print summary
        print(f"\n✅ Deduplication completed successfully!")
        print(f"\n📋 Summary:")
        print(f"   Original entities: {original_stats['num_concepts']}")
        print(f"   Merged entities: {merged_stats['num_concepts']}")
        print(f"   Reduction: {concept_reduction} entities ({reduction_pct:.1f}%)")
        print(f"   Entity clusters: {len(clusters)}")
        print(f"   Average confidence: {sum(c.confidence_score for c in clusters) / len(clusters):.3f}")
        
        print(f"\n🎨 Next steps:")
        print(f"   • Open {output_dir}/deduplicated_graph.graphml in Gephi")
        print(f"   • Review {output_dir}/deduplication_report.json for details")
        print(f"   • Use {output_dir}/deduplicated_terag_graph.json for retrieval")
        
    except Exception as e:
        print(f"❌ Error during deduplication: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()