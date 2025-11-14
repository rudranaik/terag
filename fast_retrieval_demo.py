#!/usr/bin/env python3
"""
Fast TERAG Retrieval Demo with Progress Tracking
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List

from graph_builder import TERAGGraph
from embedding_manager import EmbeddingManager
from hybrid_retriever import create_hybrid_retriever
from query_processor import ProcessedQuery
from query_explainer import explain_query_step_by_step
import dotenv

dotenv.load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Fast TERAG Retrieval Demo with Progress")
    parser.add_argument("--graph-file", type=str, required=True, help="Path to TERAG graph JSON file")
    parser.add_argument("--top-k", type=int, default=15, help="Number of results to return")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")
    
    args = parser.parse_args()
    
    # Check inputs
    graph_file = Path(args.graph_file)
    if not graph_file.exists():
        print(f"❌ Graph file not found: {graph_file}")
        return
    
    print("🔍 FAST TERAG RETRIEVAL DEMO")
    print("=" * 70)
    print(f"📁 Graph file: {graph_file}")
    print()
    
    try:
        # Load graph
        print(f"📖 Loading TERAG graph...")
        start_time = time.time()
        graph = TERAGGraph.load_from_file(str(graph_file))
        load_time = time.time() - start_time
        
        graph_stats = graph.get_statistics()
        print(f"✅ Graph loaded in {load_time:.2f}s:")
        print(f"   📊 {graph_stats['num_passages']} passages, {graph_stats['num_concepts']} concepts")
        
        # Check OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY environment variable not set")
            return
        
        # Initialize with progress tracking
        print(f"\n🚀 Initializing retrieval system...")
        print(f"   📡 This will embed {graph_stats['num_concepts']} concepts + {graph_stats['num_passages']} passages")
        print(f"   ⏳ First run may take 2-5 minutes (cached afterwards)")
        print(f"   🔄 Initializing...")
        
        start_time = time.time()
        
        embedding_manager = EmbeddingManager(batch_size=500)
        retriever = create_hybrid_retriever(
            graph=graph,
            embedding_manager=embedding_manager,
            ppr_weight=0.6,
            semantic_weight=0.4,
            score_fusion_method="weighted_sum"
        )
        
        init_time = time.time() - start_time
        print(f"✅ Retrieval system initialized in {init_time:.2f}s")
        
        if args.interactive:
            run_interactive_mode(retriever, args)
        else:
            print(f"\n💡 Ready for queries! Run with --interactive for query mode.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def run_interactive_mode(retriever, args):
    """Interactive query mode"""
    print(f"\n🎮 INTERACTIVE RETRIEVAL MODE")
    print("=" * 70)
    print("Enter queries to test retrieval (type 'quit' to exit)")
    print("Commands:")
    print("  /stats: Show system statistics")
    print("  /help: Show this help")
    print()
    
    while True:
        try:
            query = input("🔍 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query == '/stats':
                stats = retriever.get_retrieval_statistics()
                print("📊 System Statistics:")
                print(f"   Queries processed: {stats['queries_processed']}")
                print(f"   Fusion method: {stats['fusion_method']}")
                print(f"   Graph entities loaded: {stats['query_processor_stats']['total_graph_entities']}")
                print(f"   Passages embedded: {stats['semantic_retriever_stats']['loaded_passages']}")
                continue
            elif query == '/help':
                print("Commands:")
                print("  /stats: Show system statistics")
                print("  /help: Show this help")
                print("  quit/exit/q: Exit interactive mode")
                continue
            elif not query:
                continue
            
            print("-" * 50)
            start_time = time.time()
            
            # Perform retrieval
            results, analysis = retriever.retrieve(
                query=query,
                top_k=args.top_k,
                enable_analysis=True
            )
            
            retrieval_time = time.time() - start_time
            
            print(f"⏱️  Retrieved {len(results)} results in {retrieval_time:.3f}s")
            
            if analysis:
                print(f"📈 Analysis: {analysis.ppr_only_results} PPR-only, "
                      f"{analysis.semantic_only_results} semantic-only, "
                      f"{analysis.combined_results} combined")
            
            # Show extracted entities first
            if hasattr(retriever.query_processor, '_last_processed_query'):
                processed_query = retriever.query_processor._last_processed_query
                if processed_query and processed_query.extracted_entities:
                    print(f"\n🏷️  Top Extracted Entities:")
                    for idx, entity in enumerate(processed_query.extracted_entities[:5]):
                        print(f"  {idx+1}. '{entity.entity_text}' (similarity: {entity.similarity_score:.3f})")
            
            # Show top retrieved passages  
            print(f"\n📄 Top {min(5, len(results))} Retrieved Passages:")
            for i, result in enumerate(results[:5]):
                print(f"\n  {i+1}. [Hybrid Score: {result.hybrid_score:.4f}] {result.passage_id[:12]}...")
                print(f"     PPR: {result.ppr_score:.4f} | Semantic: {result.semantic_score:.4f}")
                print(f"     {result.explanation}")
                if result.entity_matches:
                    print(f"     🎯 Matched entities: {', '.join(result.entity_matches)}")
                content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
                print(f"     📝 Content: {content_preview}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")


if __name__ == "__main__":
    main()