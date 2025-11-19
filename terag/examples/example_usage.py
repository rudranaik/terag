"""
TERAG Example Usage

Demonstrates how to:
1. Build TERAG from existing chunks
2. Save and load graphs
"""

import sys
import os
import json
import argparse
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from terag import TERAG, TERAGConfig
from terag.core import create_terag_from_existing_chunks


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


def example_2_save_and_load_graph():
    """Example 2: Save and load pre-built graph"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Save and Load Graph")
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
        choices=[1, 2],
        help="Run specific example (1-2, default: all)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TERAG: Token-Efficient Graph-Based RAG")
    print("Example Usage Demonstrations")
    print("=" * 70)

    if args.example == 1 or args.example is None:
        example_1_basic_usage()

    if args.example == 2 or args.example is None:
        example_2_save_and_load_graph()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
