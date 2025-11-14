#!/usr/bin/env python3
"""
TERAG Dual-Layer Retrieval Demo

Complete demonstration of the TERAG dual-layer retrieval system:
1. Entity-based PPR retrieval 
2. Direct semantic passage retrieval
3. Hybrid fusion of both approaches

Includes query processing, result analysis, and performance evaluation.
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
    parser = argparse.ArgumentParser(description="TERAG Dual-Layer Retrieval Demo")
    parser.add_argument(
        "--graph-file",
        type=str,
        default="dedup_results/deduplicated_terag_graph.json",  # Use deduplicated graph by default
        help="Path to TERAG graph JSON file"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Specific queries to test (space-separated)"
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="JSON file containing test queries"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of results to return per query"
    )
    parser.add_argument(
        "--ppr-weight",
        type=float,
        default=0.6,
        help="Weight for PPR scores in hybrid fusion"
    )
    parser.add_argument(
        "--semantic-weight", 
        type=float,
        default=0.4,
        help="Weight for semantic scores in hybrid fusion"
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "max", "harmonic_mean"],
        help="Score fusion method"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="retrieval_results",
        help="Directory to save retrieval results"
    )
    parser.add_argument(
        "--enable-analysis",
        action="store_true",
        help="Enable detailed retrieval analysis"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON files"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive query mode"
    )
    parser.add_argument(
        "--explain-query",
        type=str,
        default=None,
        help="Explain a specific query step-by-step"
    )
    
    args = parser.parse_args()
    
    # Check input files
    graph_file = Path(args.graph_file)
    if not graph_file.exists():
        print(f"❌ Graph file not found: {graph_file}")
        print("   Available graph files:")
        for potential_file in ["dedup_results/deduplicated_terag_graph.json", "graph_results/terag_graph.json"]:
            if Path(potential_file).exists():
                print(f"     - {potential_file}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 TERAG DUAL-LAYER RETRIEVAL DEMO")
    print("=" * 70)
    print(f"📁 Graph file: {graph_file}")
    print(f"📁 Output dir: {output_dir}")
    print(f"⚙️  PPR weight: {args.ppr_weight}, Semantic weight: {args.semantic_weight}")
    print(f"⚙️  Fusion method: {args.fusion_method}")
    print(f"⚙️  Top-k results: {args.top_k}")
    print()
    
    try:
        # Load graph
        print(f"📖 Loading TERAG graph...")
        start_time = time.time()
        graph = TERAGGraph.load_from_file(str(graph_file))
        load_time = time.time() - start_time
        
        graph_stats = graph.get_statistics()
        print(f"✅ Graph loaded in {load_time:.2f}s:")
        print(f"   📊 {graph_stats['num_passages']} passages, {graph_stats['num_concepts']} concepts, {graph_stats['num_edges']} edges")
        
        # Check OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ OPENAI_API_KEY environment variable not set")
            print("   Set your OpenAI API key to use the retrieval system")
            return
        
        # Initialize hybrid retriever
        print(f"\n🚀 Initializing dual-layer retrieval system...")
        start_time = time.time()
        
        embedding_manager = EmbeddingManager()
        retriever = create_hybrid_retriever(
            graph=graph,
            embedding_manager=embedding_manager,
            ppr_weight=args.ppr_weight,
            semantic_weight=args.semantic_weight,
            score_fusion_method=args.fusion_method
        )
        
        init_time = time.time() - start_time
        print(f"✅ Retrieval system initialized in {init_time:.2f}s")
        
        # Prepare queries
        test_queries = prepare_test_queries(args)
        
        if args.explain_query:
            # Query explanation mode
            run_query_explanation(retriever, args.explain_query, graph, args)
        elif args.interactive:
            # Interactive mode
            run_interactive_mode(retriever, args)
        else:
            # Batch processing mode
            run_batch_retrieval(retriever, test_queries, args, output_dir)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def prepare_test_queries(args) -> List[str]:
    """Prepare test queries from various sources"""
    
    test_queries = []
    
    # From command line arguments
    if args.queries:
        test_queries.extend(args.queries)
    
    # From query file
    if args.query_file and Path(args.query_file).exists():
        try:
            with open(args.query_file, 'r') as f:
                query_data = json.load(f)
                if isinstance(query_data, list):
                    test_queries.extend(query_data)
                elif isinstance(query_data, dict) and "queries" in query_data:
                    test_queries.extend(query_data["queries"])
        except Exception as e:
            print(f"⚠️  Failed to load queries from file: {e}")
    
    # Default test queries if none provided
    if not test_queries:
        test_queries = [
            "How is the music business performing?",
            "What was the revenue growth in Q2?",
            "Saregama financial results and performance",
            "Live events and concert revenue updates",
            "Digital streaming music growth trends",
            "Entertainment industry earnings report",
            "Music album releases this quarter"
        ]
    
    return test_queries


def run_batch_retrieval(retriever, test_queries: List[str], args, output_dir: Path):
    """Run batch retrieval on test queries"""
    
    print(f"\n📋 Running batch retrieval on {len(test_queries)} queries...")
    print(f"{'='*70}")
    
    all_results = []
    all_analyses = []
    
    for i, query in enumerate(test_queries):
        print(f"\n🔍 Query {i+1}/{len(test_queries)}: '{query}'")
        print("-" * 50)
        
        start_time = time.time()
        
        # Perform retrieval
        results, analysis = retriever.retrieve(
            query=query,
            top_k=args.top_k,
            enable_analysis=args.enable_analysis
        )
        
        retrieval_time = time.time() - start_time
        
        print(f"⏱️  Retrieval completed in {retrieval_time:.3f}s")
        print(f"📊 Found {len(results)} results")
        
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
        print(f"\n📄 Top {min(10, len(results))} Retrieved Passages (by relevance):")
        for j, result in enumerate(results[:10]):
            print(f"\n  {j+1}. [Hybrid Score: {result.hybrid_score:.4f}] {result.passage_id[:12]}")
            print(f"     PPR: {result.ppr_score:.4f} | Semantic: {result.semantic_score:.4f}")
            print(f"     {result.explanation}")
            if result.entity_matches:
                print(f"     🎯 Matched entities: {', '.join(result.entity_matches)}")
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            print(f"     📝 Content: {content_preview}")
            print("     " + "─" * 60)
        
        # Store results
        query_result = {
            "query": query,
            "query_index": i,
            "retrieval_time": retrieval_time,
            "num_results": len(results),
            "results": [
                {
                    "rank": j + 1,
                    "passage_id": result.passage_id,
                    "hybrid_score": result.hybrid_score,
                    "ppr_score": result.ppr_score,
                    "semantic_score": result.semantic_score,
                    "confidence": result.confidence,
                    "entity_matches": result.entity_matches,
                    "explanation": result.explanation,
                    "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content
                }
                for j, result in enumerate(results)
            ]
        }
        
        if analysis:
            query_result["analysis"] = {
                "total_results": analysis.total_results,
                "ppr_only_results": analysis.ppr_only_results,
                "semantic_only_results": analysis.semantic_only_results,
                "combined_results": analysis.combined_results,
                "coverage_overlap": analysis.coverage_overlap,
                "query_confidence": analysis.query_confidence
            }
        
        all_results.append(query_result)
        if analysis:
            all_analyses.append(analysis)
    
    # Print summary
    print(f"\n📊 RETRIEVAL SUMMARY")
    print("=" * 70)
    
    avg_retrieval_time = sum(r["retrieval_time"] for r in all_results) / len(all_results)
    avg_results_count = sum(r["num_results"] for r in all_results) / len(all_results)
    
    print(f"📈 Processed {len(test_queries)} queries")
    print(f"⏱️  Average retrieval time: {avg_retrieval_time:.3f}s")
    print(f"📊 Average results per query: {avg_results_count:.1f}")
    
    if all_analyses:
        avg_overlap = sum(a.coverage_overlap for a in all_analyses) / len(all_analyses)
        avg_confidence = sum(a.query_confidence for a in all_analyses) / len(all_analyses)
        print(f"🔗 Average PPR-Semantic overlap: {avg_overlap:.2f}")
        print(f"🎯 Average query confidence: {avg_confidence:.2f}")
    
    # Show retrieval system statistics
    retriever_stats = retriever.get_retrieval_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Fusion method: {retriever_stats['fusion_method']}")
    print(f"   PPR weight: {retriever_stats['ppr_weight']}")
    print(f"   Semantic weight: {retriever_stats['semantic_weight']}")
    print(f"   Graph entities loaded: {retriever_stats['query_processor_stats']['total_graph_entities']}")
    print(f"   Passages embedded: {retriever_stats['semantic_retriever_stats']['loaded_passages']}")
    
    # Save results if requested
    if args.save_results:
        save_retrieval_results(all_results, retriever_stats, output_dir, args)


def run_query_explanation(retriever, query: str, graph: TERAGGraph, args):
    """Run detailed explanation for a specific query"""
    
    print(f"\n🔍 DETAILED QUERY EXPLANATION")
    print("=" * 70)
    print(f"Query: '{query}'")
    print()
    
    # Run step-by-step explanation
    explanation = explain_query_step_by_step(
        query=query,
        graph=graph,
        hybrid_retriever=retriever,
        top_k=args.top_k,
        detailed=True
    )
    
    print(f"\n📊 EXPLANATION SUMMARY")
    print("-" * 40)
    print(f"   📝 Query confidence: {explanation.processed_query.confidence_score:.3f}")
    print(f"   🎯 Entities extracted: {len(explanation.entity_explanations)}")
    print(f"   🕸️  PPR paths found: {len(explanation.ppr_paths)}")
    print(f"   🧠 Semantic matches: {len(explanation.semantic_matches)}")
    print(f"   🔄 Final hybrid results: {len(explanation.hybrid_fusion)}")
    
    if explanation.entity_explanations:
        print(f"\n🏷️  TOP EXTRACTED ENTITIES:")
        for i, entity_exp in enumerate(explanation.entity_explanations[:5]):
            print(f"   {i+1}. '{entity_exp.entity_text}' (similarity: {entity_exp.similarity_score:.3f})")
            print(f"      Connected to {entity_exp.connected_passages} passages in graph")
    
    print(f"\n✅ Query explanation complete!")


def run_interactive_mode(retriever, args):
    """Run interactive query mode"""
    
    print(f"\n🎮 INTERACTIVE RETRIEVAL MODE")
    print("=" * 70)
    print("Enter queries to test retrieval (type 'quit' to exit)")
    print("Commands:")
    print("  - /stats: Show system statistics")
    print("  - /help: Show this help")
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
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif query == '/help':
                print("Commands:")
                print("  - /stats: Show system statistics")
                print("  - /help: Show this help")
                print("  - quit/exit/q: Exit interactive mode")
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
            print(f"\n📄 Top {min(5, len(results))} Retrieved Passages (by relevance):")
            for i, result in enumerate(results[:5]):  # Show top 5 in interactive mode
                print(f"\n  {i+1}. [Hybrid Score: {result.hybrid_score:.4f}] {result.passage_id[:12]}")
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


def save_retrieval_results(all_results: List[dict], retriever_stats: dict, output_dir: Path, args):
    """Save retrieval results to files"""
    
    print(f"\n💾 Saving results to {output_dir}...")
    
    # Save detailed results
    results_file = output_dir / "retrieval_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save system configuration and stats
    config_file = output_dir / "retrieval_config.json"
    config_data = {
        "configuration": {
            "graph_file": args.graph_file,
            "top_k": args.top_k,
            "ppr_weight": args.ppr_weight,
            "semantic_weight": args.semantic_weight,
            "fusion_method": args.fusion_method
        },
        "system_statistics": retriever_stats,
        "summary": {
            "total_queries": len(all_results),
            "avg_retrieval_time": sum(r["retrieval_time"] for r in all_results) / len(all_results),
            "avg_results_per_query": sum(r["num_results"] for r in all_results) / len(all_results)
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Save query-result mapping for analysis
    simple_results = []
    for result in all_results:
        simple_results.append({
            "query": result["query"],
            "top_3_results": [
                {
                    "score": r["hybrid_score"],
                    "content_preview": r["content_preview"],
                    "explanation": r["explanation"]
                }
                for r in result["results"][:3]
            ]
        })
    
    simple_file = output_dir / "simple_results.json"
    with open(simple_file, 'w') as f:
        json.dump(simple_results, f, indent=2)
    
    print(f"   📄 {results_file.name} - Detailed retrieval results")
    print(f"   📄 {config_file.name} - Configuration and statistics") 
    print(f"   📄 {simple_file.name} - Simplified results for review")


if __name__ == "__main__":
    main()