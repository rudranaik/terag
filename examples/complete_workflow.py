"""
Complete TERAG Workflow Example

Demonstrates:
1. Loading documents (simulated)
2. Configuring graph construction with auto-save
3. Building the graph with regex-based extraction (no API key needed)
4. Querying the graph
5. Saving results
"""

import json
import os
from terag import TERAG, TERAGConfig

def main():
    # 1. Prepare Data
    # In a real app, you would load these from PDFs or text files
    print("1. Preparing data...")
    chunks = [
        {
            "content": "Apple Inc. released the iPhone 15 in September 2023. It features a new titanium design.",
            "metadata": {"source": "tech_news.txt", "date": "2023-09"}
        },
        {
            "content": "Microsoft Corporation announced new AI features for Azure. Satya Nadella highlighted the importance of cloud computing.",
            "metadata": {"source": "cloud_report.txt", "date": "2023-10"}
        },
        {
            "content": "Google's Pixel 8 introduces enhanced camera capabilities using AI. The phone competes directly with the iPhone.",
            "metadata": {"source": "mobile_review.txt", "date": "2023-10"}
        }
    ]
    
    # 2. Configure TERAG
    # Using regex fallback mode (use_llm_for_ner=False) for easy testing
    print("\n2. Configuring TERAG...")
    config = TERAGConfig(
        # Graph Construction
        min_concept_freq=1,           # Even single occurrences matter for small dataset
        max_concept_freq_ratio=1.0,   
        
        # Retrieval
        top_k=3,
        
        # NER / Entity Extraction
        use_llm_for_ner=False,        # Set to True if you have GROQ_API_KEY
        
        # Persistence
        auto_save_graph=True,
        graph_save_path="example_graph.json"
    )
    
    # 3. Build Graph
    print("\n3. Building Knowledge Graph...")
    terag = TERAG.from_chunks(chunks, config=config, verbose=True)
    
    # 4. Retrieval
    queries = [
        "What did Apple release?",
        "Tell me about AI features",
        "Satya Nadella"
    ]
    
    print("\n4. Querying...")
    for query in queries:
        print(f"\nQuery: '{query}'")
        results, metrics = terag.retrieve(query)
        
        for i, res in enumerate(results, 1):
            print(f"  {i}. [{res.score:.3f}] {res.content[:100]}...")
            print(f"     Matched: {res.matched_concepts}")

    # 5. Reloading (Verification)
    print("\n5. Verifying Persistence...")
    if os.path.exists("example_graph.json"):
        loaded_terag = TERAG.from_graph_file("example_graph.json", verbose=True)
        print("   Graph successfully reloaded from disk!")

if __name__ == "__main__":
    main()
