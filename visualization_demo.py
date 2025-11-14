"""
TERAG Graph Visualization Demo

Complete example showing how to:
1. Ingest chunked documents into a TERAG graph
2. Convert to NetworkX format
3. Create multiple visualizations
4. Export to various formats for use in external tools

Usage:
    python visualization_demo.py --input chunks.json --output visualizations/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

from graph_builder import GraphBuilder, TERAGGraph
from graph_visualizer import visualize_terag_graph
from ner_extractor import extract_named_entities  # Assuming this exists


def simple_concept_extractor(text: str) -> Tuple[List[str], List[str]]:
    """
    Simple concept extractor for demonstration
    In practice, you'd use more sophisticated NER and concept extraction
    """
    words = text.split()
    
    # Extract potential named entities (capitalized words > 2 chars)
    named_entities = []
    for word in words:
        cleaned = word.strip('.,!?;:"()[]{}')
        if cleaned and len(cleaned) > 2 and cleaned[0].isupper() and cleaned.isalpha():
            named_entities.append(cleaned)
    
    # Extract potential key concepts (long words, technical terms)
    doc_concepts = []
    for word in words:
        cleaned = word.lower().strip('.,!?;:"()[]{}')
        if len(cleaned) > 8 and cleaned.isalpha():
            doc_concepts.append(cleaned)
    
    # Remove duplicates while preserving order
    named_entities = list(dict.fromkeys(named_entities))
    doc_concepts = list(dict.fromkeys(doc_concepts))
    
    return named_entities, doc_concepts


def create_sample_chunks() -> List[dict]:
    """Create sample document chunks for demonstration"""
    return [
        {
            "content": "Apple Inc reported quarterly revenue of $120 billion, marking significant growth in the technology sector. The company's innovation in artificial intelligence and machine learning continues to drive performance.",
            "metadata": {"source": "earnings_q4_2024.pdf", "page": 1, "chunk_id": "chunk_0"}
        },
        {
            "content": "Microsoft Corporation announced Azure cloud services revenue increased by 30% year-over-year. The company's investment in artificial intelligence infrastructure supports enterprise customers globally.",
            "metadata": {"source": "microsoft_earnings.pdf", "page": 2, "chunk_id": "chunk_1"}
        },
        {
            "content": "The technology industry faces increased competition as companies like Apple, Microsoft, and Google invest heavily in artificial intelligence research and development initiatives.",
            "metadata": {"source": "industry_analysis.pdf", "page": 15, "chunk_id": "chunk_2"}
        },
        {
            "content": "California remains the epicenter of technology innovation, with Silicon Valley companies leading developments in machine learning, quantum computing, and sustainable energy solutions.",
            "metadata": {"source": "regional_report.pdf", "page": 8, "chunk_id": "chunk_3"}
        },
        {
            "content": "Enterprise software solutions are increasingly incorporating artificial intelligence capabilities. Companies like Salesforce, Oracle, and ServiceNow are transforming business processes through automation.",
            "metadata": {"source": "enterprise_software.pdf", "page": 22, "chunk_id": "chunk_4"}
        },
        {
            "content": "The quantum computing research conducted by IBM, Google, and Microsoft represents breakthrough innovations that could revolutionize computational capabilities across industries.",
            "metadata": {"source": "quantum_research.pdf", "page": 5, "chunk_id": "chunk_5"}
        },
        {
            "content": "Apple's investment in renewable energy and sustainable manufacturing processes demonstrates corporate responsibility while maintaining profitability in competitive markets.",
            "metadata": {"source": "sustainability_report.pdf", "page": 12, "chunk_id": "chunk_6"}
        },
        {
            "content": "Machine learning algorithms power recommendation systems for companies like Netflix, Amazon, and Spotify, personalizing user experiences through sophisticated data analysis.",
            "metadata": {"source": "ml_applications.pdf", "page": 7, "chunk_id": "chunk_7"}
        }
    ]


def load_chunks_from_file(file_path: str) -> List[dict]:
    """Load chunks from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Ensure chunks have required structure
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                if 'content' in chunk:
                    processed_chunks.append(chunk)
                else:
                    print(f"Warning: Chunk {i} missing 'content' field")
            elif isinstance(chunk, str):
                # If chunk is just a string, create proper structure
                processed_chunks.append({
                    "content": chunk,
                    "metadata": {"chunk_id": f"chunk_{i}"}
                })
            else:
                print(f"Warning: Chunk {i} has invalid format")
        
        return processed_chunks
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="TERAG Graph Visualization Demo")
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON file with document chunks (optional, uses sample data if not provided)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations",
        help="Output directory for visualizations (default: visualizations)"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["matplotlib", "plotly", "pyvis"],
        default=["matplotlib", "plotly", "pyvis"],
        help="Visualization formats to create"
    )
    parser.add_argument(
        "--min-concept-freq",
        type=int,
        default=1,
        help="Minimum concept frequency threshold"
    )
    parser.add_argument(
        "--max-concept-ratio",
        type=float,
        default=0.6,
        help="Maximum concept frequency ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--save-graph",
        action="store_true",
        help="Save the TERAG graph as JSON"
    )
    
    args = parser.parse_args()
    
    print("TERAG Graph Visualization Demo")
    print("="*50)
    
    # Load or create chunks
    if args.input:
        print(f"Loading chunks from: {args.input}")
        chunks = load_chunks_from_file(args.input)
        if not chunks:
            print("Failed to load chunks, using sample data instead")
            chunks = create_sample_chunks()
    else:
        print("Using sample document chunks")
        chunks = create_sample_chunks()
    
    print(f"Loaded {len(chunks)} document chunks")
    
    # Build TERAG graph
    print("\n1. Building TERAG Graph...")
    print("-" * 30)
    
    builder = GraphBuilder(
        min_concept_freq=args.min_concept_freq,
        max_concept_freq_ratio=args.max_concept_ratio,
        enable_concept_clustering=False  # Can be enabled for large datasets
    )
    
    # Try to use the existing NER extractor, fall back to simple extractor
    try:
        from ner_extractor import extract_entities_and_concepts
        extract_concepts_fn = extract_entities_and_concepts
        print("Using advanced NER extractor")
    except ImportError:
        extract_concepts_fn = simple_concept_extractor
        print("Using simple concept extractor")
    
    terag_graph = builder.build_graph_from_chunks(chunks, extract_concepts_fn)
    
    # Save graph if requested
    if args.save_graph:
        graph_path = Path(args.output) / "terag_graph.json"
        graph_path.parent.mkdir(exist_ok=True)
        terag_graph.save_to_file(str(graph_path))
        print(f"\nSaved TERAG graph to: {graph_path}")
    
    # Convert to NetworkX and create visualizations
    print("\n2. Creating Visualizations...")
    print("-" * 30)
    
    try:
        created_files = visualize_terag_graph(
            terag_graph,
            output_dir=args.output,
            formats=args.formats,
            export_data=True
        )
        
        print("\n3. Visualization Summary")
        print("-" * 30)
        print("Your TERAG graph has been successfully converted to NetworkX format!")
        print(f"Created {len(created_files)} output files:")
        
        for format_type, file_path in created_files.items():
            print(f"  • {format_type}: {file_path}")
        
        print("\nVisualization Options:")
        
        if 'matplotlib' in created_files:
            print("  📊 Static plot (PNG) - Good for papers and reports")
        
        if 'plotly' in created_files:
            print("  🌐 Interactive plot (HTML) - Hover for details, zoom, pan")
            print("     Open in browser for best experience")
        
        if 'pyvis' in created_files:
            print("  🕸️  Network visualization (HTML) - Physics simulation, interactive")
            
        if any(f.endswith('.graphml') for f in created_files.values()):
            print("  📁 GraphML export - Compatible with Gephi, Cytoscape, yEd")
            
        if any(f.endswith('.gexf') for f in created_files.values()):
            print("  📁 GEXF export - Native Gephi format")
        
        print("\nExternal Tool Recommendations:")
        print("  • Gephi: Load .gexf or .graphml files for advanced network analysis")
        print("  • Cytoscape: Import .graphml for biological network-style layouts")
        print("  • yEd: Open .graphml for hierarchical and circular layouts")
        print("  • NetworkX: Use the TERAG graph object directly in Python")
        
        # Show example of accessing NetworkX graph
        print("\n4. NetworkX Integration Example")
        print("-" * 30)
        
        try:
            nx_graph = terag_graph.to_networkx()
            nx_stats = terag_graph.get_networkx_statistics()
            
            print("Your NetworkX graph is ready! Example usage:")
            print(f"""
import networkx as nx
from graph_builder import TERAGGraph

# Load your saved graph
terag_graph = TERAGGraph.load_from_file('{args.output}/terag_graph.json')

# Convert to NetworkX
G = terag_graph.to_networkx()

# Network analysis examples:
print(f"Nodes: {{G.number_of_nodes()}}")           # {nx_graph.number_of_nodes()}
print(f"Edges: {{G.number_of_edges()}}")           # {nx_graph.number_of_edges()}
print(f"Density: {{nx.density(G):.4f}}")           # {nx_stats.get('density', 'N/A'):.4f}

# Find most connected concepts
degree_centrality = nx.degree_centrality(G)
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Most connected nodes:", top_nodes)

# Community detection
import networkx.algorithms.community as nx_comm
communities = nx_comm.greedy_modularity_communities(G)
print(f"Found {{len(communities)}} communities")
""")
            
        except Exception as e:
            print(f"NetworkX integration example failed: {e}")
    
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Make sure you have the required packages installed:")
        print("pip install networkx matplotlib plotly pyvis")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("Demo completed successfully! 🎉")
    print("Your document chunks have been transformed into a NetworkX graph")
    print(f"ready for analysis and visualization in your favorite tools.")


if __name__ == "__main__":
    main()