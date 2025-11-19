"""
TERAG Named Entity Recognition and Concept Extraction

LLM-based extraction of:
1. Named Entities (people, organizations, locations, dates)
2. Document-level Concepts (key terms, topics)

Uses Groq API with GPT OSS 20B for high-accuracy extraction
with file-based caching for resumable processing
"""

import re
from typing import List, Tuple, Set, Dict, Optional
from collections import Counter
from dataclasses import dataclass
import json
import logging
import time

# Import our custom components
from terag.embeddings.cache import ExtractionCache
from .groq_client import GroqClient, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    entity_type: str  # "PERSON", "ORG", "LOC", "DATE", "CONCEPT"
    confidence: float = 1.0
    context: str = ""  # Surrounding context


class NERExtractor:
    """
    LLM-based NER and concept extraction for TERAG

    Strategy:
    1. Primary: LLM-based extraction using Groq API
    2. Caching: File-based persistence for resumable processing
    3. Fallback: Basic regex patterns if LLM fails
    """

    def __init__(
        self,
        cache_dir: str = "extraction_cache",
        groq_api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-20b",
        use_fallback: bool = True,
        enable_progress_reporting: bool = True,
        use_llm: bool = True
    ):
        """
        Initialize LLM-based NER extractor
        
        Args:
            cache_dir: Directory for caching extractions
            groq_api_key: Groq API key (if None, loads from environment)
            model: Groq model to use
            use_fallback: Whether to use regex fallback if LLM fails
            enable_progress_reporting: Whether to print progress reports
            use_llm: Whether to use LLM for extraction
        """
        # Initialize caching system
        self.cache = ExtractionCache(cache_dir)
        self.use_fallback = use_fallback
        self.enable_progress_reporting = enable_progress_reporting
        self.use_llm = use_llm
        
        # Initialize LLM client
        self.groq_client = None
        if use_llm:
            try:
                self.groq_client = GroqClient(
                    api_key=groq_api_key,
                    model=model
                )
                logger.info(f"Groq client initialized with model: {model}")
            except Exception as e:
                if use_fallback:
                    logger.warning(f"Failed to initialize Groq client: {e}. Will use fallback extraction.")
                    self.groq_client = None
                else:
                    raise e

        # Common patterns for different entity types
        self.patterns = {
            "DATE": [
                r'\b\d{4}\b',  # Year
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # MM/DD/YYYY
                r'\bQ[1-4]\s+\d{4}\b',  # Q1 2024
                r'\b(?:FY|fiscal year)\s*\d{4}\b'
            ],
            "ORG": [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Corporation|Company|Ltd|Limited|LLC)\b',
                r'\b(?:the\s+)?[A-Z][a-z]+\s+(?:Bank|Group|Association|Foundation|Institute)\b'
            ],
            "MONEY": [
                r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|T))?\b',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD|INR|rupees)\b'
            ],
            "PERCENT": [
                r'\b\d+(?:\.\d+)?%\b',
                r'\b\d+(?:\.\d+)?\s*percent\b'
            ]
        }

        # Common stopwords to filter
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these", "those", 
            "it", "its", "he", "she", "him", "her", "his", "hers", "them", "their", "they",
            "they're", "they've", "they'll", "they'd", "they're", "they've", "they'll", "they'd",
            "who", "whom", "whose", "what", "which", "that", "this", "these", "those"
        }

    def extract_entities_and_concepts(
        self,
        text: str,
        document_metadata: Optional[Dict[str, any]] = None,
        passage_metadata: Optional[Dict[str, any]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Extract named entities and document concepts from text

        Args:
            text: Input text
            document_metadata: Document-level metadata for context
            passage_metadata: Passage-level metadata for caching

        Returns:
            (named_entities, document_concepts)
        """
        # Check cache first
        if self.cache.is_passage_cached(text):
            cached_result = self.cache.get_cached_extraction(text)
            if cached_result:
                if self.enable_progress_reporting:
                    logger.info(f"Using cached extraction for passage {cached_result.passage_hash[:8]}...")
                return cached_result.entities, cached_result.concepts
        
        # Use LLM for extraction
        if self.groq_client:
            entities, concepts = self._extract_with_llm(text, document_metadata, passage_metadata)
        elif self.use_fallback:
            logger.warning("LLM not available, using fallback extraction")
            entities, concepts = self._extract_fallback(text)
        else:
            raise RuntimeError("LLM client not available and fallback disabled")
            
        return entities, concepts
    
    def _extract_with_llm(
        self, 
        text: str,
        document_metadata: Optional[Dict[str, any]] = None,
        passage_metadata: Optional[Dict[str, any]] = None
    ) -> Tuple[List[str], List[str]]:
        """Extract entities and concepts using LLM"""
        start_time = time.time()
        
        try:
            # Call LLM
            entities, concepts, llm_response = self.groq_client.extract_entities_and_concepts(
                text, document_metadata or {}
            )
            
            processing_time = time.time() - start_time
            
            # Cache the result
            self.cache.cache_extraction(
                content=text,
                entities=entities,
                concepts=concepts,
                document_metadata=document_metadata or {},
                passage_metadata=passage_metadata or {},
                model_used=llm_response.model,
                processing_time_seconds=processing_time,
                api_cost_estimate=llm_response.cost_estimate
            )
            
            if self.enable_progress_reporting:
                logger.info(f"LLM extraction: {len(entities)} entities, {len(concepts)} concepts ({processing_time:.1f}s, ${llm_response.cost_estimate:.4f})")
            
            return entities, concepts
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            if self.use_fallback:
                return self._extract_fallback(text)
            else:
                raise e
    
    def _extract_fallback(self, text: str) -> Tuple[List[str], List[str]]:
        """Fallback extraction using regex patterns"""
        # Extract named entities using patterns
        entities = self._extract_entities_pattern_based(text)
        
        # Extract document concepts
        concepts = self._extract_document_concepts(text)
        
        # Convert to text lists
        entity_texts = list(set(e.text for e in entities))
        concept_texts = list(set(concepts))
        
        return entity_texts, concept_texts
    
    def extract_from_passages(
        self,
        passages: List[Dict[str, any]],
        progress_interval: int = 10
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Extract entities and concepts from multiple passages with progress tracking
        
        Args:
            passages: List of passage dictionaries with 'content', 'document_metadata', 'passage_metadata'
            progress_interval: Print progress every N passages
            
        Returns:
            List of (entities, concepts) tuples for each passage
        """
        # Initialize processing
        self.cache.initialize_processing(len(passages))
        
        results = []
        processed_count = 0
        
        if self.enable_progress_reporting:
            print(f"\n🚀 Starting extraction for {len(passages)} passages...")
            
        for i, passage in enumerate(passages):
            content = passage.get('content', '')
            document_metadata = passage.get('document_metadata', {})
            passage_metadata = passage.get('passage_metadata', {})
            passage_metadata['original_index'] = i  # Track original order
            
            # Check if already cached
            if self.cache.is_passage_cached(content):
                cached_result = self.cache.get_cached_extraction(content)
                results.append((cached_result.entities, cached_result.concepts))
                self.cache.mark_passage_skipped()
                
                if self.enable_progress_reporting and (i + 1) % progress_interval == 0:
                    progress = self.cache.get_progress_report()
                    print(f"📊 Progress: {progress['completion_percentage']} complete "
                          f"({progress['completed_passages']}/{progress['total_passages']} passages)")
                continue
            
            # Extract with LLM
            try:
                entities, concepts = self.extract_entities_and_concepts(
                    content, document_metadata, passage_metadata
                )
                results.append((entities, concepts))
                processed_count += 1
                
                # Print progress
                if self.enable_progress_reporting and (i + 1) % progress_interval == 0:
                    progress = self.cache.get_progress_report()
                    print(f"📊 Progress: {progress['completion_percentage']} complete "
                          f"({progress['completed_passages']}/{progress['total_passages']} passages, "
                          f"~{progress['estimated_remaining_time']} remaining)")
                    
            except Exception as e:
                logger.error(f"Failed to extract from passage {i}: {e}")
                self.cache.mark_passage_failed()
                results.append(([], []))  # Empty result for failed passage
        
        # Final report
        if self.enable_progress_reporting:
            final_progress = self.cache.get_progress_report()
            print(f"\n✅ Extraction completed!")
            print(f"📈 Final stats: {final_progress['completed_passages']} successful, "
                  f"{final_progress['skipped_passages']} cached, "
                  f"{final_progress['failed_passages']} failed")
            print(f"💰 Total cost: {final_progress['total_cost_estimate']}")
            
        return results
    
    def get_cache_summary(self) -> Dict[str, any]:
        """Get summary of cached extractions"""
        progress = self.cache.get_progress_report()
        
        # Calculate document coverage
        doc_count = len(self.cache.document_metadata)
        
        # Calculate entity/concept statistics
        all_entities = set()
        all_concepts = set()
        
        for result in self.cache.extractions.values():
            all_entities.update(result.entities)
            all_concepts.update(result.concepts)
        
        return {
            "cache_status": progress,
            "documents_processed": doc_count,
            "unique_entities": len(all_entities),
            "unique_concepts": len(all_concepts),
            "cache_directory": str(self.cache.cache_dir)
        }
    
    def export_results(self, output_file: str, include_content: bool = False):
        """Export cached results to file"""
        self.cache.export_extractions(output_file, include_content)
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear_cache()

    def _extract_entities_pattern_based(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []

        # Extract by pattern type
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_text = match.group().strip()
                    if len(entity_text) > 1:  # Filter single chars
                        entities.append(Entity(
                            text=entity_text,
                            entity_type=entity_type,
                            confidence=0.8,  # Pattern-based confidence
                            context=self._get_context(text, match.start(), match.end())
                        ))

        # Extract capitalized phrases (potential PERSON/ORG/LOC)
        entities.extend(self._extract_capitalized_phrases(text))

        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _extract_capitalized_phrases(self, text: str) -> List[Entity]:
        """Extract capitalized phrases as potential named entities"""
        entities = []

        # Pattern: Capitalized words (2-5 words max)
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b'
        matches = re.finditer(pattern, text)

        for match in matches:
            phrase = match.group().strip()

            # Filter out common false positives
            words = phrase.split()
            if all(w.lower() in self.stopwords for w in words):
                continue

            # Heuristic: Longer phrases more likely to be entities
            confidence = min(0.9, 0.5 + len(words) * 0.1)

            entities.append(Entity(
                text=phrase,
                entity_type="UNKNOWN",  # Could be PERSON, ORG, or LOC
                confidence=confidence,
                context=self._get_context(text, match.start(), match.end())
            ))

        return entities

    def _extract_document_concepts(self, text: str) -> List[str]:
        """
        Extract key document concepts (important terms/phrases)

        Strategy:
        1. Extract noun phrases
        2. Filter by term frequency (TF)
        3. Filter by term length and capitalization
        """
        concepts = []

        # Extract potential concepts:
        # 1. Multi-word capitalized terms
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        capitalized_matches = re.findall(capitalized_pattern, text)
        concepts.extend(capitalized_matches)

        # 2. Technical terms (camelCase, hyphenated, acronyms)
        technical_pattern = r'\b[A-Z]{2,}\b|\b[a-z]+[A-Z][a-zA-Z]*\b|\b[a-z]+-[a-z]+\b'
        technical_matches = re.findall(technical_pattern, text)
        concepts.extend(technical_matches)

        # 3. Important domain terms (long words)
        words = re.findall(r'\b[a-zA-Z]{6,}\b', text)
        word_freq = Counter(w.lower() for w in words)

        # Keep terms that appear 2+ times and aren't stopwords
        for word, freq in word_freq.items():
            if freq >= 2 and word.lower() not in self.stopwords:
                concepts.append(word)

        # Deduplicate and normalize
        concepts = list(set(c.strip() for c in concepts if len(c) > 3))

        return concepts

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for an entity"""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _enhance_with_llm(self, text: str, entities: List[Entity]) -> List[Entity]:
        """
        Enhance entity extraction with LLM (optional)

        Uses few-shot prompting to extract additional entities
        """
        if not self.llm_interface:
            return entities

        # Create few-shot prompt
        prompt = self._create_ner_prompt(text)

        try:
            # Call LLM
            response = self.llm_interface.generate(prompt)

            # Parse LLM response
            llm_entities = self._parse_llm_entities(response)

            # Merge with pattern-based entities
            merged = self._merge_entity_lists(entities, llm_entities)

            return merged

        except Exception as e:
            print(f"Warning: LLM enhancement failed: {e}")
            return entities

    def _create_ner_prompt(self, text: str) -> str:
        """Create few-shot NER prompt"""
        prompt = f"""Extract named entities from the following text. Identify:
- PERSON: People's names
- ORG: Organizations, companies
- LOC: Locations, places
- DATE: Dates and time periods
- CONCEPT: Key technical terms or domain concepts

Text: {text[:1000]}

Return entities in JSON format:
[{{"text": "...", "type": "..."}}, ...]

Entities:"""
        return prompt

    def _parse_llm_entities(self, response: str) -> List[Entity]:
        """Parse entities from LLM response"""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            entities = []
            for item in data:
                entities.append(Entity(
                    text=item["text"],
                    entity_type=item["type"],
                    confidence=0.9  # LLM confidence
                ))
            return entities
        except:
            # Fallback: parse as simple list
            return []

    def _merge_entity_lists(
        self,
        pattern_entities: List[Entity],
        llm_entities: List[Entity]
    ) -> List[Entity]:
        """Merge entities from different sources"""
        entity_dict = {}

        # Add pattern-based entities
        for entity in pattern_entities:
            key = entity.text.lower()
            entity_dict[key] = entity

        # Add LLM entities (higher confidence, so override)
        for entity in llm_entities:
            key = entity.text.lower()
            if key in entity_dict:
                # Update type and confidence
                entity_dict[key].entity_type = entity.entity_type
                entity_dict[key].confidence = max(
                    entity_dict[key].confidence,
                    entity.confidence
                )
            else:
                entity_dict[key] = entity

        return list(entity_dict.values())


class QueryNER:
    """
    Lightweight NER specifically for queries (few-shot prompting)
    """

    def __init__(self, llm_interface: Optional[object] = None):
        self.llm_interface = llm_interface
        self.extractor = NERExtractor(use_llm=False)

    def extract_query_entities(self, query: str) -> List[str]:
        """
        Extract named entities from a user query

        Args:
            query: User query text

        Returns:
            List of entity texts
        """
        # Use pattern-based extraction (fast)
        entities, _ = self.extractor.extract_entities_and_concepts(
            query
        )

        # Optionally enhance with LLM for better accuracy
        if self.llm_interface:
            entities = self._enhance_query_ner_with_llm(query, entities)

        return entities

    def _enhance_query_ner_with_llm(
        self,
        query: str,
        pattern_entities: List[str]
    ) -> List[str]:
        """Use few-shot LLM to extract query entities"""
        prompt = f"""Extract the key named entities and concepts from this query:

Examples:
Query: "What was Apple's revenue in 2024?"
Entities: ["Apple", "revenue", "2024"]

Query: "How did Microsoft perform in Q4?"
Entities: ["Microsoft", "Q4"]

Query: "{query}"
Entities:"""

        try:
            response = self.llm_interface.generate(prompt)
            # Parse response (simplified)
            llm_entities = eval(response.strip())  # Caution: eval in production
            return list(set(pattern_entities + llm_entities))
        except:
            return pattern_entities


# Convenience function
def extract_concepts_from_text(text: str, use_cache: bool = True) -> Tuple[List[str], List[str]]:
    """
    Quick function to extract entities and concepts using LLM

    Returns:
        (named_entities, document_concepts)
    """
    cache_dir = "quick_extraction_cache" if use_cache else None
    extractor = NERExtractor(cache_dir=cache_dir, use_fallback=True)
    return extractor.extract_entities_and_concepts(text)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test passages with metadata
    test_passages = [
        {
            "content": """Apple Inc announced strong revenue growth in Q4 2024, reaching $120 billion. 
            The company expanded operations in California and India. CEO Tim Cook highlighted 
            the success of new products and the growing market share.""",
            "document_metadata": {
                "document_id": "apple_earnings_2024.pdf",
                "source": "apple_earnings_2024.pdf",
                "author": "Apple Inc",
                "document_type": "earnings_report",
                "date_created": "2024-01-15"
            },
            "passage_metadata": {
                "chunk_id": "chunk_0",
                "page": 1,
                "chunk_index": 0
            }
        },
        {
            "content": """Microsoft Corporation reported significant achievements in cloud computing 
            and artificial intelligence segments during fiscal year 2024. Azure revenue 
            increased 30% year-over-year, reaching $35 billion.""",
            "document_metadata": {
                "document_id": "microsoft_earnings_2024.pdf", 
                "source": "microsoft_earnings_2024.pdf",
                "author": "Microsoft Corp",
                "document_type": "earnings_report",
                "date_created": "2024-01-20"
            },
            "passage_metadata": {
                "chunk_id": "chunk_0",
                "page": 1,
                "chunk_index": 0
            }
        }
    ]

    print("Testing LLM-based NER Extraction with Caching")
    print("=" * 70)
    print("⚠️  Note: Requires GROQ_API_KEY environment variable")
    print()

    try:
        # Initialize extractor with caching
        extractor = NERExtractor(
            cache_dir="test_extraction_cache",
            use_fallback=True,
            enable_progress_reporting=True
        )
        
        # Test single extraction
        print("🧪 Testing single passage extraction...")
        entities, concepts = extractor.extract_entities_and_concepts(
            test_passages[0]["content"],
            test_passages[0]["document_metadata"],
            test_passages[0]["passage_metadata"]
        )
        
        print(f"\nExtracted from passage 1:")
        print(f"  Entities ({len(entities)}): {entities}")
        print(f"  Concepts ({len(concepts)}): {concepts}")
        
        # Test batch processing
        print(f"\n🚀 Testing batch extraction...")
        results = extractor.extract_from_passages(test_passages, progress_interval=1)
        
        # Show results
        for i, (entities, concepts) in enumerate(results):
            print(f"\nPassage {i+1} results:")
            print(f"  Entities: {entities}")
            print(f"  Concepts: {concepts}")
        
        # Show cache summary
        print(f"\n📊 Cache Summary:")
        summary = extractor.get_cache_summary()
        for key, value in summary.items():
            if key != "cache_status":
                print(f"  {key}: {value}")
        
        # Test query NER
        print("\n" + "=" * 70)
        print("Testing Query NER")
        query_ner = QueryNER()
        test_query = "What was Apple's revenue in Q4 2024?"
        query_entities = query_ner.extract_query_entities(test_query)

        print(f"\nQuery: {test_query}")
        print(f"Entities: {query_entities}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTo run this test, you need:")
        print("1. Set GROQ_API_KEY environment variable")
        print("2. Install dependencies: pip install groq python-dotenv")
        print("\nFalling back to regex-only test...")
        
        # Fallback test without LLM
        from collections import Counter
        test_text = test_passages[0]["content"]
        
        # Simple regex extraction
        import re
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', test_text)
        concepts = [w for w in test_text.split() if len(w) > 8]
        
        print(f"\nFallback extraction results:")
        print(f"  Entities: {list(set(entities))}")
        print(f"  Concepts: {list(set(concepts))}")
