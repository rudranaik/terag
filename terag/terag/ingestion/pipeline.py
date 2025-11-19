#!/usr/bin/env python3
"""
TERAG Smart Document Ingestion Pipeline

Single script that handles the complete pipeline:
1. Extract entities from new chunks
2. Build graph from extractions  
3. Merge with existing graph (if exists)
4. Deduplicate entities
5. Output final ready-to-use graph

Usage: python ingest_chunks.py --chunks new_documents.txt
"""

import os
import json
import argparse
import time
import shutil
from pathlib import Path
from typing import Optional

# Import all required components
from terag.graph.builder import GraphBuilder, TERAGGraph
from terag.graph.deduplication import OpenAIEntityDeduplicator
from terag.graph.merger import apply_deduplication_to_graph


def run_ner_extraction(chunks_file: str, output_dir: str, verbose: bool = True) -> str:
    """Run NER extraction on chunks file"""
    if verbose:
        print(f"🔍 STEP 1: EXTRACTING ENTITIES")
        print(f"   Input: {chunks_file}")
        
        # Show file info
        file_size = os.path.getsize(chunks_file)
        file_size_mb = file_size / (1024 * 1024)
        print(f"   📊 File size: {file_size_mb:.2f} MB")
        
        # Try to preview JSON structure
        try:
            with open(chunks_file, 'r') as f:
                sample = f.read(1000)  # Read first 1KB
                if sample.count('{') > 1:
                    print(f"   📄 Detected JSON with multiple objects/chunks")
                else:
                    print(f"   📄 Detected single JSON object")
        except:
            pass
    
    # Import and run NER extraction
    import subprocess
    import sys
    import threading
    import time
    
    ner_output_dir = output_dir + "/ner_temp"
    Path(ner_output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Run NER extraction
        cmd = [
            sys.executable, "-m", "terag.ingestion.ner_extraction",
            "--files", chunks_file,
            "--output", ner_output_dir
        ]
        
        if verbose:
            print(f"   🚀 Starting NER extraction...")
            print(f"   ⏳ This may take several minutes for large files...")
        
        # Start the subprocess
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True, 
                                 cwd=".")
        
        # Progress indicator in a separate thread
        def show_progress():
            dots = 0
            while process.poll() is None:
                dots = (dots + 1) % 4
                progress_str = "   🔄 Processing" + "." * dots + " " * (3 - dots)
                if verbose:
                    print(f"\r{progress_str}", end="", flush=True)
                time.sleep(2)
        
        if verbose:
            progress_thread = threading.Thread(target=show_progress)
            progress_thread.daemon = True
            progress_thread.start()
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        if verbose:
            print(f"\r   ✅ NER extraction completed!" + " " * 20)  # Clear progress line
        
        if process.returncode != 0:
            raise RuntimeError(f"NER extraction failed: {stderr}")
        
        ner_file = os.path.join(ner_output_dir, "ner_extractions.json")
        
        if not os.path.exists(ner_file):
            raise RuntimeError(f"NER extraction did not produce expected output file: {ner_file}")
        
        if verbose:
            # Load and show stats
            with open(ner_file, 'r') as f:
                extractions = json.load(f)
            print(f"   📊 Extracted entities from {len(extractions)} chunks")
            
            # Show first few extractions as preview
            if extractions:
                print(f"   👁️  Preview first extraction:")
                first = extractions[0]
                content_preview = first.get('content', '')[:100] + "..." if len(first.get('content', '')) > 100 else first.get('content', '')
                print(f"      Content: {content_preview}")
                print(f"      Entities: {len(first.get('entities', []))}, Concepts: {len(first.get('concepts', []))}")
        
        return ner_file
        
    except Exception as e:
        if verbose:
            print(f"\r   ❌ NER extraction failed: {e}" + " " * 20)
        raise


def build_graph_from_ner(ner_file: str, output_dir: str, verbose: bool = True) -> str:
    """Build TERAG graph from NER extractions"""
    if verbose:
        print(f"\n🏗️  STEP 2: BUILDING GRAPH")
        print(f"   Input: {ner_file}")
        
        # Show input stats
        try:
            with open(ner_file, 'r') as f:
                extractions = json.load(f)
            print(f"   📊 Input: {len(extractions)} chunks to process")
        except:
            print(f"   📊 Loading input file...")
    
    try:
        if verbose:
            print(f"   🚀 Initializing graph builder...")
        
        # Initialize graph builder
        builder = GraphBuilder(
            min_concept_freq=1,
            max_concept_freq_ratio=0.5,
            enable_concept_clustering=False,
            enable_edge_weights=True,
            separate_entity_concept_types=True
        )
        
        if verbose:
            print(f"   🔄 Building graph from extractions...")
        
        # Build graph
        graph = builder.build_graph_from_ner_extractions(ner_file)
        
        # Save graph
        graph_output_dir = output_dir + "/graph_temp"
        Path(graph_output_dir).mkdir(parents=True, exist_ok=True)
        
        graph_file = os.path.join(graph_output_dir, "terag_graph.json")
        
        if verbose:
            print(f"   💾 Saving graph to {graph_file}")
        
        graph.save_to_file(graph_file)
        
        if verbose:
            stats = graph.get_statistics()
            print(f"   ✅ Built graph: {stats['num_passages']} passages, "
                  f"{stats['num_concepts']} concepts, {stats['num_edges']} edges")
            print(f"   📈 Avg concepts per passage: {stats['avg_concepts_per_passage']:.2f}")
        
        return graph_file
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Graph building failed: {e}")
        raise


def merge_graphs(new_graph_file: str, existing_graph_file: Optional[str], output_dir: str, verbose: bool = True) -> str:
    """Merge new graph with existing graph if it exists"""
    if existing_graph_file and os.path.exists(existing_graph_file):
        if verbose:
            print(f"\n🔗 STEP 3: MERGING GRAPHS")
            print(f"   Existing: {existing_graph_file}")
            print(f"   New: {new_graph_file}")
        
        try:
            # Load both graphs
            existing_graph = TERAGGraph.load_from_file(existing_graph_file)
            new_graph = TERAGGraph.load_from_file(new_graph_file)
            
            existing_stats = existing_graph.get_statistics()
            new_stats = new_graph.get_statistics()
            
            if verbose:
                print(f"   📊 Existing: {existing_stats['num_passages']} passages, {existing_stats['num_concepts']} concepts")
                print(f"   📊 New: {new_stats['num_passages']} passages, {new_stats['num_concepts']} concepts")
            
            # Create combined graph
            combined_graph = TERAGGraph()
            
            # Add all passages from both graphs
            passages_added = 0
            for passage in existing_graph.passages.values():
                combined_graph.add_passage(passage)
                passages_added += 1
                
            for passage in new_graph.passages.values():
                if passage.passage_id not in combined_graph.passages:
                    combined_graph.add_passage(passage)
                    passages_added += 1
                elif verbose:
                    print(f"      ⚠️  Skipping duplicate passage: {passage.passage_id}")
            
            # Add all concepts (auto-merges by concept_id)
            concepts_before = len(combined_graph.concepts)
            for concept in existing_graph.concepts.values():
                combined_graph.add_concept(concept)
            for concept in new_graph.concepts.values():
                combined_graph.add_concept(concept)
            concepts_total = len(combined_graph.concepts)
            
            # Add all edges
            edges_added = 0
            for passage_id, concepts in existing_graph.passage_to_concepts.items():
                for concept_id, weight in concepts.items():
                    combined_graph.add_edge(passage_id, concept_id, weight)
                    edges_added += 1
                    
            for passage_id, concepts in new_graph.passage_to_concepts.items():
                for concept_id, weight in concepts.items():
                    if passage_id in combined_graph.passages and concept_id in combined_graph.concepts:
                        existing_weight = combined_graph.passage_to_concepts.get(passage_id, {}).get(concept_id)
                        if existing_weight is None:
                            combined_graph.add_edge(passage_id, concept_id, weight)
                            edges_added += 1
                        elif weight > existing_weight:
                            combined_graph.add_edge(passage_id, concept_id, weight)
            
            # Save merged graph
            merge_output_dir = output_dir + "/merge_temp"
            Path(merge_output_dir).mkdir(parents=True, exist_ok=True)
            
            merged_file = os.path.join(merge_output_dir, "merged_graph.json")
            combined_graph.save_to_file(merged_file)
            
            if verbose:
                final_stats = combined_graph.get_statistics()
                print(f"   ✅ Merged: {final_stats['num_passages']} passages, "
                      f"{final_stats['num_concepts']} concepts, {final_stats['num_edges']} edges")
            
            return merged_file
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Graph merging failed: {e}")
            raise
    else:
        if verbose:
            print(f"\n⏭️  STEP 3: SKIPPING MERGE (no existing graph)")
        return new_graph_file


def deduplicate_graph(graph_file: str, output_dir: str, dedup_settings: dict, verbose: bool = True) -> str:
    """Run deduplication on the graph"""
    if verbose:
        print(f"\n🔍 STEP 4: DEDUPLICATING ENTITIES")
        print(f"   Input: {graph_file}")
    
    try:
        if verbose:
            print(f"   📖 Loading graph...")
        
        # Load graph
        graph = TERAGGraph.load_from_file(graph_file)
        original_stats = graph.get_statistics()
        
        if verbose:
            print(f"   📊 Original: {original_stats['num_concepts']} concepts, {original_stats['num_passages']} passages")
            print(f"   🚀 Running 3-phase deduplication...")
            print(f"      Phase 1: String similarity (threshold: {dedup_settings['string_similarity_threshold']})")
            print(f"      Phase 2: Embedding similarity (threshold: {dedup_settings['embedding_similarity_threshold']})")
            print(f"      Phase 3: Graph co-occurrence (threshold: {dedup_settings['graph_similarity_threshold']})")
        
        # Initialize deduplicator
        deduplicator = OpenAIEntityDeduplicator(
            embedding_manager=None, # Will be initialized inside if needed, or we should pass it
            string_similarity_threshold=dedup_settings['string_similarity_threshold'],
            embedding_similarity_threshold=dedup_settings['embedding_similarity_threshold'],
            graph_similarity_threshold=dedup_settings['graph_similarity_threshold']
        )
        
        # Run deduplication
        # Note: OpenAIEntityDeduplicator needs an embedding manager. 
        # We should initialize it here or let the class handle it.
        # Let's check how OpenAIEntityDeduplicator is implemented.
        # It takes embedding_manager in __init__.
        
        from terag.embeddings.manager import EmbeddingManager
        embedding_manager = EmbeddingManager()
        deduplicator.embedding_manager = embedding_manager
        
        entity_mapping, clusters = deduplicator.deduplicate_entities_from_graph(graph)
        
        if entity_mapping:
            if verbose:
                print(f"   🎯 Found {len(entity_mapping)} duplicate entities in {len(clusters)} clusters")
                
                # Show some examples
                if clusters and len(clusters) > 0:
                    print(f"   📋 Top duplicate clusters:")
                    sorted_clusters = sorted(clusters, key=lambda c: len(c.duplicate_entities), reverse=True)
                    for i, cluster in enumerate(sorted_clusters[:3]):
                        print(f"      {i+1}. '{cluster.canonical_entity}' ← {list(cluster.duplicate_entities)}")
                
                print(f"   🔄 Merging duplicate entities...")
            
            # Apply deduplication
            deduplicated_graph = apply_deduplication_to_graph(graph, entity_mapping)
            final_stats = deduplicated_graph.get_statistics()
            
            if verbose:
                concepts_reduced = original_stats['num_concepts'] - final_stats['num_concepts']
                reduction_pct = (concepts_reduced / original_stats['num_concepts']) * 100
                print(f"   ✅ Reduced concepts by {concepts_reduced} ({reduction_pct:.1f}%)")
                print(f"   📊 Final: {final_stats['num_concepts']} concepts, {final_stats['num_edges']} edges")
        else:
            if verbose:
                print(f"   ✨ No duplicates found - graph is already clean")
            deduplicated_graph = graph
        
        if verbose:
            print(f"   💾 Saving final graph...")
        
        # Save deduplicated graph
        final_file = os.path.join(output_dir, "terag_graph.json")
        deduplicated_graph.save_to_file(final_file)
        
        # Also save entity mapping if any
        if entity_mapping:
            mapping_file = os.path.join(output_dir, "entity_mapping.json")
            with open(mapping_file, 'w') as f:
                json.dump(entity_mapping, f, indent=2)
            if verbose:
                print(f"   💾 Entity mapping saved: {mapping_file}")
        
        return final_file
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Deduplication failed: {e}")
        raise


def cleanup_temp_dirs(output_dir: str, verbose: bool = True):
    """Clean up temporary directories"""
    temp_dirs = [
        output_dir + "/ner_temp",
        output_dir + "/graph_temp", 
        output_dir + "/merge_temp"
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if verbose:
                print(f"   🗑️  Cleaned up: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(description="TERAG Smart Document Ingestion Pipeline")
    parser.add_argument(
        "--chunks",
        type=str,
        required=True,
        help="Path to text file containing chunks to ingest (one chunk per line)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="terag_data",
        help="Directory to store the final graph and results"
    )
    parser.add_argument(
        "--existing-graph",
        type=str,
        default=None,
        help="Path to existing graph to merge with (auto-detected if not provided)"
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
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary files for debugging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Validate inputs
    chunks_file = Path(args.chunks)
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect existing graph if not provided
    existing_graph_file = args.existing_graph
    if not existing_graph_file:
        potential_existing = output_dir / "terag_graph.json"
        if potential_existing.exists():
            existing_graph_file = str(potential_existing)
            if verbose:
                print(f"🔍 Auto-detected existing graph: {existing_graph_file}")
    
    if verbose:
        print("🚀 TERAG SMART DOCUMENT INGESTION PIPELINE")
        print("=" * 80)
        print(f"📁 Input chunks: {chunks_file}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔗 Existing graph: {existing_graph_file or 'None (fresh ingestion)'}")
        print(f"⚙️  Dedup settings: string={args.string_threshold}, "
              f"embedding={args.embedding_threshold}, graph={args.graph_threshold}")
        if args.skip_embeddings:
            print("⚙️  Embedding deduplication: DISABLED")
        print()
    
    try:
        start_time = time.time()
        
        # Step 1: NER Extraction
        ner_file = run_ner_extraction(str(chunks_file), str(output_dir), verbose)
        
        # Step 2: Build Graph
        new_graph_file = build_graph_from_ner(ner_file, str(output_dir), verbose)
        
        # Step 3: Merge with existing (if exists)
        merged_graph_file = merge_graphs(new_graph_file, existing_graph_file, str(output_dir), verbose)
        
        # Step 4: Deduplicate
        dedup_settings = {
            "string_similarity_threshold": args.string_threshold,
            "embedding_similarity_threshold": args.embedding_threshold if not args.skip_embeddings else 0.99,
            "graph_similarity_threshold": args.graph_threshold
        }
        
        final_graph_file = deduplicate_graph(merged_graph_file, str(output_dir), dedup_settings, verbose)
        
        # Step 5: Cleanup (unless keeping temp files)
        if not args.keep_temp_files:
            if verbose:
                print(f"\n🧹 CLEANUP")
            cleanup_temp_dirs(str(output_dir), verbose)
        
        # Final summary
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n🎉 INGESTION COMPLETE!")
            print("=" * 80)
            
            # Load final graph for stats
            final_graph = TERAGGraph.load_from_file(final_graph_file)
            final_stats = final_graph.get_statistics()
            
            print(f"✅ Final graph: {final_stats['num_passages']} passages, "
                  f"{final_stats['num_concepts']} concepts, {final_stats['num_edges']} edges")
            print(f"⏱️  Total processing time: {total_time:.2f} seconds")
            print(f"💾 Ready-to-use graph: {final_graph_file}")
            print()
            print("🔍 Next steps:")
            print(f"   # Test retrieval system:")
            print(f"   python retrieval_demo.py --graph-file {final_graph_file} --interactive")
            print()
            print(f"   # Ingest more chunks:")
            print(f"   python ingest_chunks.py --chunks new_chunks.txt")
        
        return 0
        
    except Exception as e:
        if verbose:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())