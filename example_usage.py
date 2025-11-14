"""
TERAG Example Usage

Demonstrates how to:
1. Build TERAG from existing chunks
2. Perform retrieval
3. Integrate with existing agentic RAG system
4. Compare TERAG vs dense retrieval
"""

import sys
import os
import json
import argparse
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terag_retriever import TERAG, TERAGConfig, create_terag_from_existing_chunks


def example_1_basic_usage():
    """Example 1: Basic TERAG usage"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic TERAG Usage")
    print("=" * 70)

    # Sample chunks
    chunks = [
        {
            "content": "Apple Inc reported Q4 2024 revenue of $120 billion, a 15% increase year-over-year.",
            "metadata": {"source": "earnings_report", "page": 1}
        },
        {
            "content": "The iPhone division contributed $65 billion to total revenue in Q4 2024.",
            "metadata": {"source": "earnings_report", "page": 2}
        },
        {
            "content": "Apple's services segment grew 20% reaching $25 billion in quarterly revenue.",
            "metadata": {"source": "earnings_report", "page": 3}
        },
        {
            "content": "Microsoft Azure cloud revenue increased 30% in Q4 2024 to $35 billion.",
            "metadata": {"source": "microsoft_report", "page": 1}
        },
        {
            "content": "Competition intensified between Apple and Microsoft in cloud services market.",
            "metadata": {"source": "market_analysis", "page": 1}
        }
    ]

    # Build TERAG
    config = TERAGConfig(
        min_concept_freq=1,
        max_concept_freq_ratio=0.8,
        top_k=3
    )

    terag = TERAG.from_chunks(chunks, config=config, verbose=True)

    # Query
    query = "What was Apple's revenue in Q4 2024?"
    print(f"\n\nQuery: {query}")
    print("-" * 70)

    results, metrics = terag.retrieve(query, verbose=True)

    print("\n\nTop Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [Score: {result.score:.4f}] {result.passage_id}")
        print(f"   Content: {result.content}")
        print(f"   Matched concepts: {result.matched_concepts}")

    print(f"\n\nRetrieval metrics:")
    print(f"  - Query entities: {metrics.num_query_entities}")
    print(f"  - Matched concepts: {metrics.num_matched_concepts}")
    print(f"  - Retrieval time: {metrics.retrieval_time:.3f}s")


def example_2_from_existing_chunks():
    """Example 2: Build TERAG from existing chunks file"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Build from Existing Chunks")
    print("=" * 70)

    chunks_file = "/Users/rudranaik/Documents/custom-rag/chunk.json"

    # Check if file exists
    if not os.path.exists(chunks_file):
        print(f"\nSkipping: {chunks_file} not found")
        print("Run the main chunking pipeline first to generate chunks.")
        return

    # Build TERAG
    config = TERAGConfig(
        min_concept_freq=2,
        max_concept_freq_ratio=0.5,
        ppr_alpha=0.15,
        semantic_weight=0.5,
        frequency_weight=0.5,
        top_k=10
    )

    terag = create_terag_from_existing_chunks(
        chunks_file=chunks_file,
        output_graph_file="terag_graph.json",
        config=config
    )

    # Test queries
    test_queries = [
        "What was the total revenue?",
        "Who are the board members?",
        "What are the main business segments?"
    ]

    for query in test_queries:
        print(f"\n\nQuery: {query}")
        print("-" * 70)

        results, metrics = terag.retrieve(query, top_k=10, verbose=False)

        print(f"\nTop 3 Results:")
        for i, result in enumerate(results, 1):
            content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
            print(f"\n{i}. [Score: {result.score:.4f}]")
            print(f"   {content_preview}")
            print(f"   Matched: {result.matched_concepts[:10]}")  # Show first 5

        print(f"\n   Retrieval time: {metrics.retrieval_time:.3f}s")


def example_3_hybrid_terag_agentic():
    """Example 3: Hybrid TERAG + Agentic RAG"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Hybrid TERAG + Agentic RAG")
    print("=" * 70)

    chunks_file = "../chunks_full_metadata.json"

    if not os.path.exists(chunks_file):
        print(f"\nSkipping: {chunks_file} not found")
        return

    # Build TERAG
    config = TERAGConfig(min_concept_freq=2, top_k=20)
    terag = TERAG.from_chunks_file(chunks_file, config=config, verbose=True)

    # Try importing agentic retrieval
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from agentic_retrieval import create_retriever_from_chunks, RetrievalConfig

        print("\n\nBuilding Agentic RAG...")
        agentic_config = RetrievalConfig(
            max_query_rewrites=2,
            top_k=10
        )
        agentic_retriever = create_retriever_from_chunks(chunks_file, config=agentic_config)

        # Hybrid approach
        query = "What was the revenue growth?"

        print(f"\n\nQuery: {query}")
        print("-" * 70)

        # Step 1: TERAG for broad retrieval (graph-based)
        print("\n1. TERAG Retrieval (Graph-based):")
        terag_results, terag_metrics = terag.retrieve(query, top_k=20, verbose=False)
        print(f"   Retrieved {len(terag_results)} passages in {terag_metrics.retrieval_time:.3f}s")

        # Step 2: Agentic RAG for refinement (dense + sparse + reranking)
        print("\n2. Agentic RAG Retrieval (Dense + Sparse):")
        agentic_results, agentic_metrics = agentic_retriever.retrieve(query, verbose=False)
        print(f"   Retrieved {len(agentic_results)} passages")
        print(f"   Iterations: {agentic_metrics.total_iterations}")

        # Step 3: Combine and deduplicate
        print("\n3. Hybrid Results (TERAG + Agentic):")

        # Simple combination: take top from each
        hybrid_results = []

        # Add TERAG results with graph-based scores
        for r in terag_results[:10]:
            hybrid_results.append({
                "content": r.content,
                "score": r.score,
                "source": "TERAG",
                "matched_concepts": r.matched_concepts
            })

        # Add agentic results
        for chunk in agentic_results[:10]:
            hybrid_results.append({
                "content": chunk.get("content", ""),
                "score": 0.5,  # Placeholder
                "source": "Agentic",
                "matched_concepts": []
            })

        # Deduplicate by content
        seen_content = set()
        unique_results = []
        for r in hybrid_results:
            content_key = r["content"][:100]  # Use first 100 chars as key
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(r)

        print(f"\n   Total unique results: {len(unique_results)}")
        print("\n   Top 3 Hybrid Results:")
        for i, r in enumerate(unique_results[:3], 1):
            preview = r["content"][:120] + "..."
            print(f"\n   {i}. [{r['source']}] [Score: {r['score']:.4f}]")
            print(f"      {preview}")

    except ImportError as e:
        print(f"\nAgentic RAG not available: {e}")
        print("Skipping hybrid example.")


def example_4_save_and_load_graph():
    """Example 4: Save and load pre-built graph"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Save and Load Graph")
    print("=" * 70)

    chunks = [
        {"content": "Apple reported $120B revenue in Q4 2024.", "metadata": {}},
        {"content": "Microsoft Azure grew 30% in Q4.", "metadata": {}},
        {"content": "Tech sector competition intensified in 2024.", "metadata": {}}
    ]

    # Build and save
    print("\n1. Building graph...")
    config = TERAGConfig(min_concept_freq=1)
    terag = TERAG.from_chunks(chunks, config=config, verbose=True)

    graph_file = "test_terag_graph.json"
    terag.save_graph(graph_file)
    print(f"\n   Graph saved to {graph_file}")

    # Load from file
    print(f"\n2. Loading graph from {graph_file}...")
    terag_loaded = TERAG.from_graph_file(graph_file, config=config, verbose=True)

    # Test retrieval
    query = "What was Apple's revenue?"
    results, metrics = terag_loaded.retrieve(query, verbose=False)

    print(f"\n3. Retrieval test:")
    print(f"   Query: {query}")
    print(f"   Results: {len(results)}")
    print(f"   Time: {metrics.retrieval_time:.3f}s")

    # Cleanup
    if os.path.exists(graph_file):
        os.remove(graph_file)
        print(f"\n   Cleaned up {graph_file}")


def main():
    """Run examples"""
    parser = argparse.ArgumentParser(description="TERAG Example Usage")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific example (1-4, default: all)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TERAG: Token-Efficient Graph-Based RAG")
    print("Example Usage Demonstrations")
    print("=" * 70)

    if args.example == 1 or args.example is None:
        example_1_basic_usage()

    if args.example == 2 or args.example is None:
        example_2_from_existing_chunks()

    if args.example == 3 or args.example is None:
        example_3_hybrid_terag_agentic()

    if args.example == 4 or args.example is None:
        example_4_save_and_load_graph()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
