"""
TERAG Named Entity Recognition and Concept Extraction

Lightweight extraction of:
1. Named Entities (people, organizations, locations, dates)
2. Document-level Concepts (key terms, topics)

Designed for minimal LLM usage - uses pattern-based + optional LLM enhancement
"""

import re
from typing import List, Tuple, Set, Dict, Optional
from collections import Counter
from dataclasses import dataclass
import json


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    entity_type: str  # "PERSON", "ORG", "LOC", "DATE", "CONCEPT"
    confidence: float = 1.0
    context: str = ""  # Surrounding context


class NERExtractor:
    """
    Lightweight NER and concept extraction for TERAG

    Strategy:
    1. Pattern-based extraction (fast, no LLM)
    2. Optional LLM enhancement for higher accuracy
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_interface: Optional[object] = None
    ):
        self.use_llm = use_llm
        self.llm_interface = llm_interface

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
            "should", "may", "might", "must", "can", "this", "that", "these", "those"
        }

    def extract_entities_and_concepts(
        self,
        text: str,
        extract_concepts: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Extract named entities and document concepts from text

        Args:
            text: Input text
            extract_concepts: Whether to extract document-level concepts

        Returns:
            (named_entities, document_concepts)
        """
        # Extract named entities using patterns
        entities = self._extract_entities_pattern_based(text)

        # Optionally enhance with LLM
        if self.use_llm and self.llm_interface:
            entities = self._enhance_with_llm(text, entities)

        # Extract document concepts (key terms)
        concepts = []
        if extract_concepts:
            concepts = self._extract_document_concepts(text)

        # Convert to text lists
        entity_texts = list(set(e.text for e in entities))
        concept_texts = list(set(concepts))

        return entity_texts, concept_texts

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
            query,
            extract_concepts=False
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
def extract_concepts_from_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Quick function to extract entities and concepts

    Returns:
        (named_entities, document_concepts)
    """
    extractor = NERExtractor(use_llm=False)
    return extractor.extract_entities_and_concepts(text)


if __name__ == "__main__":
    # Test NER extraction
    test_text = """
    Apple Inc announced strong revenue growth in Q4 2024, reaching $120 billion.
    The company expanded operations in California and India. CEO Tim Cook
    highlighted the success of new products and the growing market share.
    Microsoft Corporation also reported significant achievements in cloud computing
    and artificial intelligence segments during fiscal year 2024.
    """

    print("Testing NER Extraction")
    print("=" * 60)

    extractor = NERExtractor()
    entities, concepts = extractor.extract_entities_and_concepts(test_text)

    print("\nNamed Entities:")
    for entity in sorted(entities):
        print(f"  - {entity}")

    print(f"\nDocument Concepts:")
    for concept in sorted(concepts):
        print(f"  - {concept}")

    # Test query NER
    print("\n" + "=" * 60)
    print("Testing Query NER")
    query_ner = QueryNER()
    test_query = "What was Apple's revenue in Q4 2024?"
    query_entities = query_ner.extract_query_entities(test_query)

    print(f"\nQuery: {test_query}")
    print(f"Entities: {query_entities}")
