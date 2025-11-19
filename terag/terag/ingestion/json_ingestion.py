"""
Universal JSON Ingestion System for TERAG

Handles various JSON structures and converts them to standardized passage format
for NER extraction with document metadata preservation.

Key Features:
1. Auto-detects JSON structure
2. Flexible content field mapping  
3. Metadata extraction and normalization
4. Configurable extraction rules
5. Support for nested and flat structures
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass  
class IngestionRule:
    """Rules for extracting content and metadata from JSON"""
    content_fields: List[str]  # Fields to treat as main content
    metadata_fields: Dict[str, str]  # JSON field -> standardized metadata name
    document_metadata_path: Optional[str] = None  # Path to document-level metadata
    passage_metadata_path: Optional[str] = None   # Path to passage-level metadata
    combine_content_fields: bool = True  # Whether to combine multiple content fields
    content_separator: str = "\n\n"  # Separator when combining content
    min_content_length: int = 10  # Minimum content length to include


@dataclass
class StandardizedPassage:
    """Standardized passage format for NER processing"""
    content: str
    document_metadata: Dict[str, Any]
    passage_metadata: Dict[str, Any]
    original_structure: Dict[str, Any]  # Keep original for reference
    content_hash: str
    
    def to_ner_format(self) -> Dict[str, Any]:
        """Convert to format expected by NER extractor"""
        return {
            "content": self.content,
            "document_metadata": self.document_metadata,
            "passage_metadata": self.passage_metadata
        }


class JSONIngestionAdapter:
    """
    Universal JSON ingestion adapter for TERAG NER system
    """
    
    def __init__(self):
        """Initialize with common ingestion rules"""
        self.predefined_rules = {
            "chunk_array": IngestionRule(
                content_fields=["content"],
                metadata_fields={
                    "chunk_id": "chunk_id",
                    "chunk_type": "chunk_type", 
                    "token_count": "token_count",
                    "artifact_type": "artifact_type",
                    "document_position": "document_position"
                },
                document_metadata_path="document_metadata",
                passage_metadata_path=None,  # Metadata is at chunk level
                min_content_length=30
            ),
            "qa_format": IngestionRule(
                content_fields=["question", "answer", "text"],
                metadata_fields={
                    "type": "interaction_type",
                    "questioner_name": "questioner",
                    "answerer_name": "answerer",
                    "speaker_name": "speaker"
                },
                combine_content_fields=True,
                content_separator="\n\nQ: ",
                min_content_length=30
            ),
            "simple_content": IngestionRule(
                content_fields=["content", "text", "body", "message"],
                metadata_fields={
                    "id": "id",
                    "title": "title",
                    "author": "author",
                    "timestamp": "timestamp",
                    "source": "source"
                },
                min_content_length=30
            ),
            "nested_chunks": IngestionRule(
                content_fields=["content"],
                metadata_fields={
                    "id": "chunk_id",
                    "type": "content_type"
                },
                document_metadata_path="metadata.document",
                passage_metadata_path="metadata.passage",
                min_content_length=30
            )
        }
    
    def detect_json_structure(self, data: Union[Dict, List]) -> str:
        """
        Auto-detect JSON structure to choose appropriate ingestion rule
        
        Returns:
            Rule name to use for processing
        """
        if isinstance(data, list):
            if len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    # Check for chunk-like structure
                    if "content" in first_item and "document_metadata" in first_item:
                        return "chunk_array"
                    # Check for Q&A structure
                    elif any(field in first_item for field in ["question", "answer"]):
                        return "qa_format"
                    else:
                        return "simple_content"
            return "simple_content"
        
        elif isinstance(data, dict):
            # Check for wrapper with chunks
            if "chunks" in data:
                chunks = data["chunks"]
                if isinstance(chunks, list) and len(chunks) > 0:
                    first_chunk = chunks[0]
                    if any(field in first_chunk for field in ["question", "answer", "type"]):
                        return "qa_format" 
                    else:
                        return "chunk_array"
            # Check for nested structure
            elif any(key in data for key in ["metadata", "document_info"]):
                return "nested_chunks"
            else:
                return "simple_content"
        
        return "simple_content"
    
    def extract_nested_field(self, data: Dict, field_path: str) -> Any:
        """
        Extract nested field using dot notation
        e.g., "metadata.document.title" -> data["metadata"]["document"]["title"]
        """
        try:
            keys = field_path.split(".")
            value = data
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def _combine_content_with_speakers(
        self, 
        content_parts: List[str], 
        rule: IngestionRule, 
        item: Dict[str, Any]
    ) -> str:
        """
        Combine content parts with speaker attribution for better graph connectivity
        
        Args:
            content_parts: List of content strings extracted from fields
            rule: Ingestion rule being applied
            item: Original JSON item with speaker information
            
        Returns:
            Combined content with speaker attribution
        """
        
        # Handle Q&A format with speakers
        if (rule.combine_content_fields and 
            len(content_parts) >= 2 and 
            "question" in rule.content_fields and 
            "answer" in rule.content_fields):
            
            question_text = content_parts[0] if len(content_parts) > 0 else ""
            answer_text = content_parts[1] if len(content_parts) > 1 else ""
            
            # Get speaker names
            questioner = item.get("questioner_name", "Unknown")
            answerer = item.get("answerer_name", "Unknown")
            
            # Format with speaker attribution
            content = f"{questioner} asked: {question_text}\n\n{answerer} answered: {answer_text}"
            
            logger.debug(f"Q&A with speakers: {questioner} -> {answerer}")
            return content
        
        # Handle commentary format with speaker
        elif (len(content_parts) == 1 and 
              "text" in rule.content_fields and 
              "speaker_name" in item):
            
            speaker = item.get("speaker_name", "Unknown")
            text = content_parts[0]
            
            # Format as attributed statement
            content = f"{speaker} said: {text}"
            
            logger.debug(f"Commentary by: {speaker}")
            return content
        
        # Handle multiple content fields without specific speaker format
        elif rule.combine_content_fields and len(content_parts) > 1:
            # Check if we can identify a speaker for general content
            speaker = item.get("speaker_name") or item.get("author") or item.get("answerer_name")
            
            if speaker:
                combined_content = rule.content_separator.join(content_parts)
                content = f"{speaker}: {combined_content}"
            else:
                content = rule.content_separator.join(content_parts)
            
            return content
        
        # Single content field - check for speaker attribution
        else:
            text = content_parts[0]
            speaker = (item.get("speaker_name") or 
                      item.get("author") or 
                      item.get("answerer_name") or 
                      item.get("questioner_name"))
            
            if speaker and not text.startswith(speaker):
                content = f"{speaker}: {text}"
            else:
                content = text
            
            return content
    
    def apply_ingestion_rule(
        self, 
        data: Union[Dict, List], 
        rule: IngestionRule,
        global_document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[StandardizedPassage]:
        """
        Apply ingestion rule to extract standardized passages
        
        Args:
            data: JSON data to process
            rule: Ingestion rule to apply
            global_document_metadata: Document-level metadata to apply to all passages
            
        Returns:
            List of standardized passages
        """
        passages = []
        
        # Handle different top-level structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "chunks" in data:
                items = data["chunks"]
            else:
                items = [data]  # Treat single object as list of one
        else:
            logger.warning(f"Unsupported data type: {type(data)}")
            return []
        
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
                
            try:
                # Extract content
                content_parts = []
                for field in rule.content_fields:
                    if field in item:
                        content_parts.append(str(item[field]))
                
                if not content_parts:
                    logger.warning(f"No content found in item {i}")
                    continue
                
                # Combine content with speaker attribution
                content = self._combine_content_with_speakers(content_parts, rule, item)
                
                # Skip if content too short
                if len(content.strip()) < rule.min_content_length:
                    logger.debug(f"Skipping short content in item {i}: {len(content)} chars")
                    continue
                
                # Extract document metadata
                document_metadata = global_document_metadata or {}
                if rule.document_metadata_path:
                    doc_meta = self.extract_nested_field(item, rule.document_metadata_path)
                    if doc_meta and isinstance(doc_meta, dict):
                        document_metadata.update(doc_meta)
                
                # Extract passage metadata
                passage_metadata = {"original_index": i}
                if rule.passage_metadata_path:
                    pass_meta = self.extract_nested_field(item, rule.passage_metadata_path)
                    if pass_meta and isinstance(pass_meta, dict):
                        passage_metadata.update(pass_meta)
                
                # Apply metadata field mappings
                for json_field, std_field in rule.metadata_fields.items():
                    if json_field in item:
                        passage_metadata[std_field] = item[json_field]
                
                # Add default metadata if missing
                if "source_file" not in document_metadata and "source" not in document_metadata:
                    document_metadata["source"] = "unknown"
                
                # Generate content hash
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                
                # Create standardized passage
                passage = StandardizedPassage(
                    content=content,
                    document_metadata=document_metadata,
                    passage_metadata=passage_metadata,
                    original_structure=item,
                    content_hash=content_hash
                )
                
                passages.append(passage)
                
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                continue
        
        logger.info(f"Extracted {len(passages)} passages using rule: {rule}")
        return passages
    
    def ingest_json_file(
        self, 
        file_path: str,
        rule_name: Optional[str] = None,
        custom_rule: Optional[IngestionRule] = None,
        global_document_metadata: Optional[Dict[str, Any]] = None
    ) -> List[StandardizedPassage]:
        """
        Ingest JSON file and extract standardized passages
        
        Args:
            file_path: Path to JSON file
            rule_name: Name of predefined rule to use
            custom_rule: Custom ingestion rule
            global_document_metadata: Document-level metadata for all passages
            
        Returns:
            List of standardized passages ready for NER
        """
        # Load JSON data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return []
        
        # Add file metadata
        if global_document_metadata is None:
            global_document_metadata = {}
        
        file_path_obj = Path(file_path)
        global_document_metadata.update({
            "source_file": file_path_obj.name,
            "source_path": str(file_path_obj),
            "file_size": file_path_obj.stat().st_size if file_path_obj.exists() else 0
        })
        
        # Choose ingestion rule
        if custom_rule:
            rule = custom_rule
            logger.info(f"Using custom ingestion rule for {file_path}")
        elif rule_name and rule_name in self.predefined_rules:
            rule = self.predefined_rules[rule_name]
            logger.info(f"Using predefined rule '{rule_name}' for {file_path}")
        else:
            # Auto-detect structure
            detected_structure = self.detect_json_structure(data)
            rule = self.predefined_rules[detected_structure]
            logger.info(f"Auto-detected structure '{detected_structure}' for {file_path}")
        
        # Apply rule and extract passages
        passages = self.apply_ingestion_rule(data, rule, global_document_metadata)
        
        logger.info(f"Successfully ingested {len(passages)} passages from {file_path}")
        return passages
    
    def ingest_multiple_files(
        self,
        file_paths: List[str],
        common_metadata: Optional[Dict[str, Any]] = None
    ) -> List[StandardizedPassage]:
        """
        Ingest multiple JSON files and combine passages
        
        Args:
            file_paths: List of JSON file paths
            common_metadata: Common document metadata for all files
            
        Returns:
            Combined list of standardized passages
        """
        all_passages = []
        
        for file_path in file_paths:
            logger.info(f"Processing file: {file_path}")
            
            # Create file-specific metadata
            file_metadata = common_metadata.copy() if common_metadata else {}
            file_metadata["batch_processing"] = True
            
            passages = self.ingest_json_file(file_path, global_document_metadata=file_metadata)
            all_passages.extend(passages)
        
        logger.info(f"Total passages from {len(file_paths)} files: {len(all_passages)}")
        return all_passages
    
    def create_custom_rule(
        self,
        content_fields: List[str],
        metadata_mappings: Dict[str, str],
        **kwargs
    ) -> IngestionRule:
        """Helper to create custom ingestion rules"""
        return IngestionRule(
            content_fields=content_fields,
            metadata_fields=metadata_mappings,
            **kwargs
        )
    
    def preview_json_structure(self, file_path: str, max_items: int = 3) -> Dict[str, Any]:
        """
        Preview JSON structure to help design custom rules
        
        Returns:
            Dictionary with structure analysis
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {"error": f"Failed to load file: {e}"}
        
        analysis = {
            "file_path": file_path,
            "data_type": type(data).__name__,
            "detected_structure": self.detect_json_structure(data),
            "recommended_rule": None,
            "sample_items": [],
            "available_fields": set(),
            "metadata_fields": set()
        }
        
        # Get sample items
        if isinstance(data, list):
            items = data[:max_items]
        elif isinstance(data, dict):
            if "chunks" in data:
                items = data["chunks"][:max_items]
            else:
                items = [data]
        else:
            items = []
        
        # Analyze fields
        for item in items:
            if isinstance(item, dict):
                analysis["available_fields"].update(item.keys())
                analysis["sample_items"].append(item)
                
                # Identify potential metadata fields
                for key, value in item.items():
                    if isinstance(value, (str, int, float, bool)) and key not in ["content", "text", "body"]:
                        analysis["metadata_fields"].add(key)
        
        analysis["available_fields"] = list(analysis["available_fields"])
        analysis["metadata_fields"] = list(analysis["metadata_fields"])
        analysis["recommended_rule"] = analysis["detected_structure"]
        
        return analysis


def run_json_analysis_demo(json_files: List[str]):
    """Demo function to analyze JSON structures"""
    adapter = JSONIngestionAdapter()
    
    print("🔍 JSON STRUCTURE ANALYSIS")
    print("="*70)
    
    for file_path in json_files:
        print(f"\n📄 Analyzing: {file_path}")
        print("-"*50)
        
        analysis = adapter.preview_json_structure(file_path)
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            continue
        
        print(f"Data Type: {analysis['data_type']}")
        print(f"Detected Structure: {analysis['detected_structure']}")
        print(f"Recommended Rule: {analysis['recommended_rule']}")
        print(f"Available Fields: {', '.join(analysis['available_fields'][:10])}")
        
        if len(analysis['available_fields']) > 10:
            print(f"  ... and {len(analysis['available_fields']) - 10} more")
        
        print(f"Metadata Fields: {', '.join(analysis['metadata_fields'][:8])}")
        
        # Show sample structure
        if analysis['sample_items']:
            print(f"\nSample Item Structure:")
            sample = analysis['sample_items'][0]
            for key, value in list(sample.items())[:5]:
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {type(value).__name__} = {repr(value)}")


if __name__ == "__main__":
    # Example usage with your JSON files
    json_files = [
        "examples/sample_data.json"
    ]
    
    run_json_analysis_demo(json_files)