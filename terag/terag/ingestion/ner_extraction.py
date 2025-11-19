"""
JSON to NER Demo

Complete demonstration of ingesting JSON files and running NER extraction
with file-based caching and progress tracking.

Usage:
    python json_ner_demo.py --files data/*.json --output results/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from terag.embeddings.cache import ExtractionCache, print_progress_report
from .groq_client import GroqClient, LLMResponse

from .json_ingestion import JSONIngestionAdapter, IngestionRule
from .ner_extractor import NERExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_json_files(file_paths: List[str]) -> Dict[str, Any]:
    """Analyze JSON file structures before processing"""
    adapter = JSONIngestionAdapter()
    
    analysis_results = {}
    total_estimated_passages = 0
    
    print("\n🔍 JSON STRUCTURE ANALYSIS")
    print("="*80)
    
    for file_path in file_paths:
        print(f"\n📄 Analyzing: {Path(file_path).name}")
        print("-"*60)
        
        analysis = adapter.preview_json_structure(file_path, max_items=2)
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            analysis_results[file_path] = {"error": analysis["error"]}
            continue
        
        # Estimate passage count
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                estimated_passages = len(data)
            elif isinstance(data, dict) and "chunks" in data:
                estimated_passages = len(data["chunks"])
            else:
                estimated_passages = 1
        except:
            estimated_passages = 0
        
        total_estimated_passages += estimated_passages
        
        print(f"📊 Data Type: {analysis['data_type']}")
        print(f"🏗️  Detected Structure: {analysis['detected_structure']}")
        print(f"📈 Estimated Passages: {estimated_passages}")
        print(f"🏷️  Available Fields: {', '.join(analysis['available_fields'][:8])}")
        
        if len(analysis['available_fields']) > 8:
            print(f"    ... and {len(analysis['available_fields']) - 8} more")
        
        # Show sample content
        if analysis['sample_items']:
            sample = analysis['sample_items'][0]
            content_fields = ["content", "question", "answer", "text", "body"]
            
            for field in content_fields:
                if field in sample:
                    content = str(sample[field])
                    preview = content[:150] + "..." if len(content) > 150 else content
                    print(f"📝 Content Preview ({field}): {repr(preview)}")
                    break
        
        analysis_results[file_path] = {
            "structure": analysis["detected_structure"],
            "estimated_passages": estimated_passages,
            "fields": analysis["available_fields"]
        }
    
    print(f"\n📊 TOTAL ESTIMATED PASSAGES: {total_estimated_passages}")
    print("="*80)
    
    return analysis_results


def run_json_ner_extraction(
    file_paths: List[str],
    output_dir: str = "ner_results",
    cache_dir: str = "json_ner_cache",
    progress_interval: int = 5,
    dry_run: bool = False
):
    """
    Run complete JSON to NER extraction pipeline
    
    Args:
        file_paths: List of JSON file paths to process
        output_dir: Directory to save results
        cache_dir: Directory for extraction caching
        progress_interval: Report progress every N passages
        dry_run: If True, only analyze structure without running NER
    """
    
    # Setup directories
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Step 1: Analyze JSON structures
    analysis = analyze_json_files(file_paths)
    
    if dry_run:
        print("\n✅ Dry run completed - structure analysis only")
        return analysis
    
    # Step 2: Initialize NER extractor with caching
    print("\n🚀 INITIALIZING NER SYSTEM")
    print("="*80)
    
    try:
        extractor = NERExtractor(
            cache_dir=cache_dir,
            use_fallback=True,  # Use regex fallback if Groq API fails
            enable_progress_reporting=True
        )
        print("✅ NER extractor initialized successfully")
        print(f"📁 Cache directory: {cache_dir}")
        
    except Exception as e:
        print(f"❌ Failed to initialize NER extractor: {e}")
        print("💡 Make sure GROQ_API_KEY environment variable is set")
        return None
    
    # Step 3: Ingest JSON files
    print(f"\n📥 INGESTING JSON FILES")
    print("="*80)
    
    adapter = JSONIngestionAdapter()
    all_passages = []
    
    for file_path in file_paths:
        print(f"\n📄 Processing: {Path(file_path).name}")
        
        try:
            # Get common metadata for this file
            file_name = Path(file_path).stem
            common_metadata = {
                "source_file": Path(file_path).name,
                "batch_id": f"json_batch_{len(all_passages)}",
                "processing_timestamp": "2024-11-14"
            }
            
            # Ingest with auto-detection
            passages = adapter.ingest_json_file(
                file_path,
                global_document_metadata=common_metadata
            )
            
            print(f"  ✅ Extracted {len(passages)} passages")
            
            # Convert to NER format
            ner_passages = [p.to_ner_format() for p in passages]
            all_passages.extend(ner_passages)
            
        except Exception as e:
            print(f"  ❌ Failed to process {file_path}: {e}")
            continue
    
    print(f"\n📊 TOTAL PASSAGES FOR NER: {len(all_passages)}")
    
    if not all_passages:
        print("❌ No passages to process")
        return None
    
    # Step 4: Run NER extraction with caching
    print(f"\n🧠 RUNNING NER EXTRACTION")
    print("="*80)
    
    try:
        # Process all passages with progress tracking
        results = extractor.extract_from_passages(
            all_passages,
            progress_interval=progress_interval
        )
        
        print(f"✅ NER extraction completed!")
        print(f"📊 Processed {len(results)} passages")
        
    except Exception as e:
        print(f"❌ NER extraction failed: {e}")
        return None
    
    # Step 5: Save results and generate reports
    print(f"\n💾 SAVING RESULTS")
    print("="*80)
    
    try:
        # Export extraction results
        results_file = output_path / "ner_extractions.json"
        extractor.export_results(str(results_file), include_content=True)
        print(f"📁 Exported extractions to: {results_file}")
        
        # Generate summary report
        cache_summary = extractor.get_cache_summary()
        
        summary_report = {
            "files_processed": file_paths,
            "total_passages": len(all_passages),
            "extraction_results": len(results),
            "cache_summary": cache_summary,
            "structure_analysis": analysis
        }
        
        summary_file = output_path / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Summary report saved to: {summary_file}")
        
        # Print final statistics
        print(f"\n📈 FINAL STATISTICS")
        print("="*80)
        
        progress = cache_summary["cache_status"]
        print(f"✅ Successfully processed: {progress['completed_passages']} passages")
        print(f"♻️  Used cached results: {progress['skipped_passages']} passages")
        print(f"❌ Failed extractions: {progress['failed_passages']} passages")
        print(f"💰 Total estimated cost: {progress['total_cost_estimate']}")
        print(f"📊 Unique entities found: {cache_summary['unique_entities']}")
        print(f"📊 Unique concepts found: {cache_summary['unique_concepts']}")
        
        # Show sample extractions
        print(f"\n🔍 SAMPLE EXTRACTIONS")
        print("="*80)
        
        sample_count = min(3, len(results))
        for i in range(sample_count):
            entities, concepts = results[i]
            passage_content = all_passages[i]["content"][:100] + "..."
            
            print(f"\nSample {i+1}:")
            print(f"📝 Content: {passage_content}")
            print(f"🏷️  Entities: {entities[:5]}")  # Show first 5
            print(f"💭 Concepts: {concepts[:5]}")   # Show first 5
        
        return {
            "results_file": str(results_file),
            "summary_file": str(summary_file),
            "total_passages": len(all_passages),
            "cache_summary": cache_summary
        }
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="JSON to NER Extraction Demo")
    parser.add_argument(
        "--files", 
        nargs="+",
        required=True,
        help="JSON files to process"
    )
    parser.add_argument(
        "--output",
        default="ner_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache",
        default="json_ner_cache", 
        help="Cache directory for extractions"
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5,
        help="Progress reporting interval"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze structure only, don't run NER"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze JSON structures"
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    valid_files = []
    for file_path in args.files:
        if Path(file_path).exists():
            valid_files.append(file_path)
        else:
            print(f"⚠️ Warning: File not found: {file_path}")
    
    if not valid_files:
        print("❌ No valid JSON files provided")
        return
    
    print(f"🎯 JSON-to-NER EXTRACTION DEMO")
    print(f"📁 Processing {len(valid_files)} files")
    print(f"📂 Output directory: {args.output}")
    print(f"🗃️ Cache directory: {args.cache}")
    
    if args.analyze_only:
        analyze_json_files(valid_files)
        return
    
    # Run the complete pipeline
    result = run_json_ner_extraction(
        valid_files,
        output_dir=args.output,
        cache_dir=args.cache,
        progress_interval=args.progress_interval,
        dry_run=args.dry_run
    )
    
    if result:
        print(f"\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved in: {args.output}")
        if not args.dry_run:
            print(f"🔄 Cache saved in: {args.cache}")
            print(f"💡 Rerun to resume from cache if interrupted")
    else:
        print(f"\n❌ Processing failed or was skipped")


if __name__ == "__main__":
    main()