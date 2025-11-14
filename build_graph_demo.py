#!/usr/bin/env python3
"""
TERAG Graph Building Demo

Builds a weighted NetworkX graph from NER extraction results
and exports it for visualization in Gephi, Cytoscape, etc.
"""

import json
import argparse
import pickle
from pathlib import Path
from graph_builder import GraphBuilder
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Build TERAG graph from NER extractions")
    parser.add_argument(
        "--ner-file", 
        type=str, 
        default="ner_results/ner_extractions.json",
        help="Path to NER extractions JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="graph_results",
        help="Directory to save graph outputs"
    )
    parser.add_argument(
        "--min-concept-freq",
        type=int,
        default=1,
        help="Minimum frequency for concepts to include"
    )
    parser.add_argument(
        "--max-concept-freq-ratio",
        type=float,
        default=0.5,
        help="Maximum frequency ratio for concepts (0.5 = max 50% of passages)"
    )
    parser.add_argument(
        "--disable-edge-weights",
        action="store_true",
        help="Disable hybrid edge weighting (use uniform weights)"
    )
    parser.add_argument(
        "--combine-entity-types",
        action="store_true",
        help="Don't separate entities and concepts into different types"
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    ner_file = Path(args.ner_file)
    if not ner_file.exists():
        print(f"❌ NER extractions file not found: {ner_file}")
        print(f"   Run NER extraction first with: python json_ner_demo.py")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🏗️  TERAG GRAPH BUILDING")
    print("=" * 60)
    print(f"📁 Input: {ner_file}")
    print(f"📁 Output: {output_dir}")
    print(f"⚙️  Edge weights: {'Disabled' if args.disable_edge_weights else 'Hybrid (TF-IDF + Position + Type + Context)'}")
    print(f"⚙️  Entity separation: {'Disabled' if args.combine_entity_types else 'Entities and Concepts separate'}")
    print()
    
    # Initialize graph builder
    builder = GraphBuilder(
        min_concept_freq=args.min_concept_freq,
        max_concept_freq_ratio=args.max_concept_freq_ratio,
        enable_concept_clustering=False,  # TODO: Implement later
        enable_edge_weights=not args.disable_edge_weights,
        separate_entity_concept_types=not args.combine_entity_types
    )
    
    try:
        # Build graph
        graph = builder.build_graph_from_ner_extractions(str(ner_file))
        
        # Save TERAG graph (JSON format)
        terag_graph_file = output_dir / "terag_graph.json"
        print(f"\n💾 Saving TERAG graph: {terag_graph_file}")
        graph.save_to_file(str(terag_graph_file))
        
        # Convert to NetworkX and export
        if NETWORKX_AVAILABLE:
            print("📊 Converting to NetworkX...")
            
            # Create version with full metadata for pickle
            nx_graph_full = graph.to_networkx(
                include_node_attributes=True,
                include_edge_weights=not args.disable_edge_weights,
                flatten_metadata=False
            )
            
            # Create version with flattened metadata for GraphML
            nx_graph_flat = graph.to_networkx(
                include_node_attributes=True,
                include_edge_weights=not args.disable_edge_weights,
                flatten_metadata=True
            )
            
            # Save as GraphML (for Gephi, Cytoscape) - use flattened version
            graphml_file = output_dir / "terag_graph.graphml"
            print(f"💾 Saving GraphML: {graphml_file}")
            try:
                nx.write_graphml(nx_graph_flat, str(graphml_file))
            except Exception as e:
                print(f"⚠️  GraphML export failed: {e}")
                print("   Trying with simplified node attributes...")
                # Create simplified version for problematic GraphML export
                G_simple = nx.Graph()
                for node_id, node_data in nx_graph_flat.nodes(data=True):
                    simple_attrs = {
                        'node_type': node_data.get('node_type', 'unknown'),
                        'label': node_data.get('concept_text', node_data.get('content', ''))[:50]
                    }
                    # Only add simple data types
                    for key, value in node_data.items():
                        if isinstance(value, (str, int, float, bool)) and value is not None:
                            simple_attrs[key] = value
                    G_simple.add_node(node_id, **simple_attrs)
                
                for u, v, edge_data in nx_graph_flat.edges(data=True):
                    simple_edge_attrs = {}
                    for key, value in edge_data.items():
                        if isinstance(value, (str, int, float, bool)) and value is not None:
                            simple_edge_attrs[key] = value
                    G_simple.add_edge(u, v, **simple_edge_attrs)
                
                nx.write_graphml(G_simple, str(graphml_file))
                print(f"   ✅ GraphML saved with simplified attributes")
            
            # Save as NetworkX pickle - use full version
            pickle_file = output_dir / "terag_graph.pkl"
            print(f"💾 Saving NetworkX pickle: {pickle_file}")
            with open(pickle_file, 'wb') as f:
                pickle.dump(nx_graph_full, f)
            
            # Save as GML (alternative format) - use flattened version
            gml_file = output_dir / "terag_graph.gml"
            print(f"💾 Saving GML: {gml_file}")
            nx.write_gml(nx_graph_flat, str(gml_file))
            
            # Get NetworkX statistics
            print("\n📈 NetworkX Graph Statistics:")
            nx_stats = graph.get_networkx_statistics()
            for key, value in nx_stats.items():
                if key != 'top_nodes_by_degree':
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
            # Show top nodes
            if 'top_nodes_by_degree' in nx_stats:
                print(f"\n🔥 Top 5 Nodes by Degree:")
                for node_id, degree in nx_stats['top_nodes_by_degree']:
                    node_data = nx_graph_full.nodes[node_id]
                    node_type = node_data.get('node_type', 'unknown')
                    if node_type == 'passage':
                        display_text = node_data.get('content', '')[:50] + "..."
                    else:
                        display_text = node_data.get('concept_text', node_id)
                    print(f"   {display_text:<55} (degree: {degree}, type: {node_type})")
            
            print(f"\n📋 Summary of Generated Files:")
            print(f"   📄 {terag_graph_file.name:<25} - TERAG JSON format")
            print(f"   📄 {graphml_file.name:<25} - GraphML for Gephi/Cytoscape")
            print(f"   📄 {pickle_file.name:<25} - NetworkX pickle")
            print(f"   📄 {gml_file.name:<25} - GML format")
            
            print(f"\n🎨 Visualization Options:")
            print(f"   • Open {graphml_file.name} in Gephi for interactive visualization")
            print(f"   • Open {graphml_file.name} in Cytoscape for network analysis")
            print(f"   • Load {pickle_file.name} in Python for programmatic analysis")
            
        else:
            print("\n⚠️  NetworkX not available. Install with: pip install networkx")
            print("   Only TERAG JSON format saved.")
        
        # Save processing summary
        summary_file = output_dir / "graph_summary.json"
        summary_data = {
            "input_file": str(ner_file),
            "graph_statistics": graph.get_statistics(),
            "builder_config": {
                "min_concept_freq": args.min_concept_freq,
                "max_concept_freq_ratio": args.max_concept_freq_ratio,
                "edge_weights_enabled": not args.disable_edge_weights,
                "separate_entity_types": not args.combine_entity_types
            }
        }
        
        if NETWORKX_AVAILABLE:
            summary_data["networkx_statistics"] = graph.get_networkx_statistics()
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n✅ Graph building completed successfully!")
        print(f"📊 Generated graph with {graph.get_statistics()['num_passages']} passages, "
              f"{graph.get_statistics()['num_concepts']} concepts, "
              f"and {graph.get_statistics()['num_edges']} edges.")
        
    except Exception as e:
        print(f"❌ Error building graph: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()