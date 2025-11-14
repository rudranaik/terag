"""
Extraction Cache System for TERAG

Provides persistent file-based caching for LLM extractions with:
- Document-level metadata preservation
- Passage-level tracking
- Progress monitoring
- Resume capability
- Cost tracking
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of entity/concept extraction for a single passage"""
    passage_hash: str
    content: str
    entities: List[str]
    concepts: List[str]
    document_metadata: Dict[str, Any]  # Document-level metadata
    passage_metadata: Dict[str, Any]   # Passage-specific metadata
    timestamp: str
    model_used: str
    processing_time_seconds: float
    api_cost_estimate: float = 0.0


@dataclass
class ProcessingProgress:
    """Track overall processing progress"""
    total_passages: int
    processed_passages: int
    skipped_passages: int  # Already in cache
    failed_passages: int
    start_time: str
    last_update: str
    estimated_completion: Optional[str] = None
    total_api_calls: int = 0
    total_cost_estimate: float = 0.0


class ExtractionCache:
    """
    File-based cache for LLM extractions with metadata preservation
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache system
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache file paths
        self.extractions_file = self.cache_dir / "extractions.json"
        self.progress_file = self.cache_dir / "progress.json"
        self.document_metadata_file = self.cache_dir / "document_metadata.json"
        self.api_usage_file = self.cache_dir / "api_usage.json"
        
        # In-memory cache
        self.extractions: Dict[str, ExtractionResult] = {}
        self.document_metadata: Dict[str, Dict] = {}  # document_id -> metadata
        self.progress: Optional[ProcessingProgress] = None
        self.api_usage_log: List[Dict] = []
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"Cache initialized: {len(self.extractions)} extractions loaded")
    
    def _load_cache(self):
        """Load all cache files on startup"""
        
        # Load extractions
        if self.extractions_file.exists():
            try:
                with open(self.extractions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for passage_hash, extract_data in data.items():
                    self.extractions[passage_hash] = ExtractionResult(**extract_data)
                    
                logger.info(f"Loaded {len(self.extractions)} cached extractions")
            except Exception as e:
                logger.warning(f"Failed to load extractions cache: {e}")
        
        # Load document metadata
        if self.document_metadata_file.exists():
            try:
                with open(self.document_metadata_file, 'r', encoding='utf-8') as f:
                    self.document_metadata = json.load(f)
                    
                logger.info(f"Loaded metadata for {len(self.document_metadata)} documents")
            except Exception as e:
                logger.warning(f"Failed to load document metadata: {e}")
        
        # Load progress
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.progress = ProcessingProgress(**data)
                    
                logger.info(f"Loaded processing progress: {self.progress.processed_passages}/{self.progress.total_passages}")
            except Exception as e:
                logger.warning(f"Failed to load progress: {e}")
        
        # Load API usage log
        if self.api_usage_file.exists():
            try:
                with open(self.api_usage_file, 'r', encoding='utf-8') as f:
                    self.api_usage_log = json.load(f)
                    
                logger.info(f"Loaded {len(self.api_usage_log)} API usage records")
            except Exception as e:
                logger.warning(f"Failed to load API usage log: {e}")
    
    def _save_extractions(self):
        """Save extractions to file"""
        data = {
            passage_hash: asdict(result)
            for passage_hash, result in self.extractions.items()
        }
        
        with open(self.extractions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_document_metadata(self):
        """Save document metadata to file"""
        with open(self.document_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.document_metadata, f, indent=2, ensure_ascii=False)
    
    def _save_progress(self):
        """Save progress to file"""
        if self.progress:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.progress), f, indent=2)
    
    def _save_api_usage(self):
        """Save API usage log to file"""
        with open(self.api_usage_file, 'w', encoding='utf-8') as f:
            json.dump(self.api_usage_log, f, indent=2)
    
    def get_passage_hash(self, content: str) -> str:
        """Generate consistent hash for passage content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def is_passage_cached(self, content: str) -> bool:
        """Check if passage is already processed"""
        passage_hash = self.get_passage_hash(content)
        return passage_hash in self.extractions
    
    def get_cached_extraction(self, content: str) -> Optional[ExtractionResult]:
        """Get cached extraction result for passage"""
        passage_hash = self.get_passage_hash(content)
        return self.extractions.get(passage_hash)
    
    def cache_extraction(
        self,
        content: str,
        entities: List[str],
        concepts: List[str],
        document_metadata: Dict[str, Any],
        passage_metadata: Dict[str, Any],
        model_used: str,
        processing_time_seconds: float,
        api_cost_estimate: float = 0.0
    ):
        """
        Cache extraction result for a passage
        
        Args:
            content: Passage text content
            entities: Extracted entities
            concepts: Extracted concepts
            document_metadata: Document-level metadata (source, author, etc.)
            passage_metadata: Passage-level metadata (page, chunk_id, etc.)
            model_used: LLM model name
            processing_time_seconds: Time taken for extraction
            api_cost_estimate: Estimated API cost
        """
        passage_hash = self.get_passage_hash(content)
        
        # Create extraction result
        result = ExtractionResult(
            passage_hash=passage_hash,
            content=content,
            entities=entities,
            concepts=concepts,
            document_metadata=document_metadata,
            passage_metadata=passage_metadata,
            timestamp=datetime.now().isoformat(),
            model_used=model_used,
            processing_time_seconds=processing_time_seconds,
            api_cost_estimate=api_cost_estimate
        )
        
        # Store in memory
        self.extractions[passage_hash] = result
        
        # Store document metadata separately (for easy lookup)
        doc_id = document_metadata.get('document_id') or document_metadata.get('source', 'unknown')
        if doc_id not in self.document_metadata:
            self.document_metadata[doc_id] = document_metadata
        
        # Log API usage
        self.api_usage_log.append({
            "timestamp": result.timestamp,
            "passage_hash": passage_hash,
            "model": model_used,
            "processing_time": processing_time_seconds,
            "cost_estimate": api_cost_estimate,
            "tokens_estimate": len(content.split()) * 1.3  # Rough token estimate
        })
        
        # Update progress
        if self.progress:
            self.progress.processed_passages += 1
            self.progress.total_api_calls += 1
            self.progress.total_cost_estimate += api_cost_estimate
            self.progress.last_update = datetime.now().isoformat()
        
        # Save to disk immediately
        self._save_extractions()
        self._save_document_metadata()
        self._save_progress()
        self._save_api_usage()
        
        logger.info(f"Cached extraction for passage {passage_hash[:8]}... ({len(entities)} entities, {len(concepts)} concepts)")
    
    def initialize_processing(self, total_passages: int):
        """Initialize processing session"""
        existing_processed = len(self.extractions)
        
        self.progress = ProcessingProgress(
            total_passages=total_passages,
            processed_passages=existing_processed,
            skipped_passages=0,
            failed_passages=0,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat()
        )
        
        self._save_progress()
        
        if existing_processed > 0:
            logger.info(f"Resuming processing: {existing_processed} passages already completed")
        else:
            logger.info(f"Starting fresh processing of {total_passages} passages")
    
    def mark_passage_skipped(self):
        """Mark a passage as skipped (already cached)"""
        if self.progress:
            self.progress.skipped_passages += 1
            self.progress.last_update = datetime.now().isoformat()
            self._save_progress()
    
    def mark_passage_failed(self):
        """Mark a passage as failed to process"""
        if self.progress:
            self.progress.failed_passages += 1
            self.progress.last_update = datetime.now().isoformat()
            self._save_progress()
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get detailed progress report"""
        if not self.progress:
            return {"status": "Not initialized"}
        
        completed = self.progress.processed_passages
        total = self.progress.total_passages
        completion_rate = (completed / total * 100) if total > 0 else 0
        
        return {
            "total_passages": total,
            "completed_passages": completed,
            "skipped_passages": self.progress.skipped_passages,
            "failed_passages": self.progress.failed_passages,
            "remaining_passages": total - completed,
            "completion_percentage": f"{completion_rate:.1f}%",
            "total_api_calls": self.progress.total_api_calls,
            "total_cost_estimate": f"${self.progress.total_cost_estimate:.4f}",
            "start_time": self.progress.start_time,
            "last_update": self.progress.last_update,
            "avg_processing_time": self._calculate_avg_processing_time(),
            "estimated_remaining_time": self._estimate_remaining_time()
        }
    
    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time per passage"""
        if not self.api_usage_log:
            return 0.0
        
        total_time = sum(entry["processing_time"] for entry in self.api_usage_log)
        return total_time / len(self.api_usage_log)
    
    def _estimate_remaining_time(self) -> str:
        """Estimate remaining processing time"""
        if not self.progress or self.progress.total_passages == 0:
            return "Unknown"
        
        remaining = self.progress.total_passages - self.progress.processed_passages
        avg_time = self._calculate_avg_processing_time()
        
        if avg_time == 0:
            return "Unknown"
        
        remaining_seconds = remaining * avg_time
        
        if remaining_seconds < 60:
            return f"{remaining_seconds:.0f} seconds"
        elif remaining_seconds < 3600:
            return f"{remaining_seconds / 60:.1f} minutes"
        else:
            return f"{remaining_seconds / 3600:.1f} hours"
    
    def get_document_passages(self, document_id: str) -> List[ExtractionResult]:
        """Get all passages for a specific document"""
        passages = []
        for result in self.extractions.values():
            doc_id = result.document_metadata.get('document_id') or result.document_metadata.get('source')
            if doc_id == document_id:
                passages.append(result)
        
        # Sort by passage order if available
        passages.sort(key=lambda x: x.passage_metadata.get('chunk_index', 0))
        return passages
    
    def export_extractions(self, output_file: str, include_content: bool = False):
        """
        Export all extractions to a file
        
        Args:
            output_file: Path to export file
            include_content: Whether to include full passage content
        """
        export_data = []
        
        for passage_hash, result in self.extractions.items():
            entry = {
                "passage_hash": passage_hash,
                "entities": result.entities,
                "concepts": result.concepts,
                "document_metadata": result.document_metadata,
                "passage_metadata": result.passage_metadata,
                "timestamp": result.timestamp,
                "model_used": result.model_used,
                "processing_time": result.processing_time_seconds
            }
            
            if include_content:
                entry["content"] = result.content
            
            export_data.append(entry)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} extractions to {output_file}")
    
    def clear_cache(self):
        """Clear all cached data (use with caution!)"""
        self.extractions.clear()
        self.document_metadata.clear()
        self.progress = None
        self.api_usage_log.clear()
        
        # Remove files
        for file_path in [self.extractions_file, self.progress_file, 
                         self.document_metadata_file, self.api_usage_file]:
            if file_path.exists():
                file_path.unlink()
        
        logger.info("Cache cleared completely")


def print_progress_report(cache: ExtractionCache):
    """Pretty print progress report"""
    report = cache.get_progress_report()
    
    print("\n" + "="*60)
    print("EXTRACTION PROGRESS REPORT")
    print("="*60)
    
    for key, value in report.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"{formatted_key:<25}: {value}")
    
    print("="*60)


if __name__ == "__main__":
    # Test the cache system
    import time
    
    # Initialize cache
    cache = ExtractionCache("test_cache")
    
    # Simulate processing
    test_passages = [
        {
            "content": "Apple Inc reported strong revenue growth in Q4 2024.",
            "document_metadata": {
                "document_id": "earnings_2024.pdf",
                "source": "earnings_2024.pdf", 
                "author": "Apple Inc",
                "date_created": "2024-01-15",
                "document_type": "earnings_report"
            },
            "passage_metadata": {
                "chunk_id": "chunk_0",
                "page": 1,
                "chunk_index": 0
            }
        },
        {
            "content": "Microsoft Azure cloud services revenue increased by 30%.",
            "document_metadata": {
                "document_id": "microsoft_q4.pdf",
                "source": "microsoft_q4.pdf",
                "author": "Microsoft Corp",
                "date_created": "2024-01-20",
                "document_type": "earnings_report"
            },
            "passage_metadata": {
                "chunk_id": "chunk_1", 
                "page": 2,
                "chunk_index": 0
            }
        }
    ]
    
    cache.initialize_processing(len(test_passages))
    
    print("Testing cache system...")
    
    for i, passage in enumerate(test_passages):
        # Check if already cached
        if cache.is_passage_cached(passage["content"]):
            print(f"Passage {i+1}: Already cached, skipping")
            cache.mark_passage_skipped()
            continue
        
        print(f"Processing passage {i+1}...")
        
        # Simulate LLM processing
        time.sleep(0.5)  # Simulate API call
        
        # Mock extraction results
        entities = ["Apple Inc", "Q4 2024"] if "Apple" in passage["content"] else ["Microsoft", "Azure"]
        concepts = ["revenue", "growth"] if "Apple" in passage["content"] else ["cloud", "services"]
        
        # Cache the result
        cache.cache_extraction(
            content=passage["content"],
            entities=entities,
            concepts=concepts,
            document_metadata=passage["document_metadata"],
            passage_metadata=passage["passage_metadata"],
            model_used="openai/gpt-oss-20b",
            processing_time_seconds=0.5,
            api_cost_estimate=0.002
        )
        
        print_progress_report(cache)
    
    print("\nCache test completed!")
    print(f"Cache directory: {cache.cache_dir}")