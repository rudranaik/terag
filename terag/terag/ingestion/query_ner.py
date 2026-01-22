"""
Improved QueryNER for TERAG

LLM-first entity extraction with fallback to regex patterns
"""

import json
import logging
import re
from typing import List, Optional

from .groq_client import GroqClient # Keep for backward compat
from .llm_providers import get_llm_provider
from .ner_extractor import NERExtractor

logger = logging.getLogger(__name__)


class ImprovedQueryNER:
    """
    Lightweight NER specifically for queries
    
    Strategy:
    1. Primary: LLM-based extraction for high accuracy
    2. Fallback: Regex patterns if LLM unavailable
    """

    def __init__(
        self, 
        groq_api_key: Optional[str] = None, 
        use_llm: bool = True,
        # New args
        provider: str = "groq",
        api_key: Optional[str] = None
    ):
        """
        Initialize QueryNER
        """
        self.llm_provider = None
        self.groq_client = None
        self.use_llm = use_llm
        
        # Try to initialize LLM client
        if use_llm:
            try:
                effective_api_key = api_key or groq_api_key
                self.llm_provider = get_llm_provider(
                    provider_name=provider,
                    api_key=effective_api_key,
                    silent_fallback=not use_llm
                )
                
                # Alias for potential backward compatibility
                self.groq_client = self.llm_provider
                
                if self.llm_provider.client is not None:
                    logger.info(f"QueryNER: LLM provider '{provider}' initialized")
                else:
                    logger.debug("QueryNER: LLM not available (no API key), will use regex fallback")
                    self.llm_provider = None
                    self.groq_client = None
            except Exception as e:
                logger.debug(f"QueryNER: LLM not available ({e}), will use regex fallback")
                self.llm_provider = None
                self.groq_client = None
        
        # Initialize fallback extractor (regex-based)
        self.fallback_extractor = NERExtractor(use_llm=False)

    def extract_query_entities(self, query: str, verbose: bool = False) -> List[str]:
        """
        Extract named entities from a user query
        
        Strategy:
        1. Try LLM-based extraction first (if available)
        2. Fall back to regex patterns if LLM fails or unavailable
        
        Args:
            query: User query text
            verbose: Print extraction method used

        Returns:
            List of entity texts
        """
        # Try LLM-based extraction first
        if self.llm_provider:
            try:
                entities = self._extract_with_llm(query)
                if verbose:
                    logger.info(f"QueryNER: Extracted {len(entities)} entities using LLM")
                return entities
            except Exception as e:
                if verbose:
                    logger.warning(f"QueryNER: LLM extraction failed ({e}), falling back to regex")
        
        # Fallback to regex-based extraction (silent unless verbose)
        entities = self._extract_with_regex(query)
        if verbose:
            logger.info(f"QueryNER: Extracted {len(entities)} entities using regex fallback")
        
        return entities

    def _extract_with_llm(self, query: str) -> List[str]:
        """
        Extract query entities using LLM (Groq)
        
        Uses few-shot prompting optimized for query understanding
        """
        system_prompt = """You are an expert at extracting key entities and concepts from search queries.

Extract ALL important terms that would help retrieve relevant documents:
- Named entities (people, organizations, products, locations)
- Key concepts and topics
- Important terms (technical terms, domain-specific words)
- Dates and time periods

Return ONLY a JSON array of strings, nothing else.
Example: ["Apple", "revenue", "Q4 2024"]"""

        user_prompt = f"""Extract key entities and concepts from this query:

Query: "{query}"

Return a JSON array of the most important terms for document retrieval."""

        # Call LLM
        response = self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        if not response.success:
            raise RuntimeError(f"LLM call failed: {response.error_message}")
        
        # Parse response
        entities = self._parse_llm_response(response.content)
        
        # Filter and clean
        entities = [e.strip() for e in entities if e and len(e.strip()) > 1]
        
        return list(set(entities))  # Deduplicate

    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response to extract entity list"""
        # Clean response
        cleaned = response.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        
        cleaned = cleaned.strip()
        
        try:
            # Try to parse as JSON array
            entities = json.loads(cleaned)
            if isinstance(entities, list):
                return entities
            elif isinstance(entities, dict):
                # Handle {"entities": [...]} format
                return entities.get('entities', [])
            else:
                return []
        except json.JSONDecodeError:
            # Fallback: extract quoted strings
            matches = re.findall(r'["\']([^"\']+)["\']', response)
            return matches if matches else []

    def _extract_with_regex(self, query: str) -> List[str]:
        """
        Fallback extraction using regex patterns
        
        Extracts:
        - Capitalized words/phrases
        - Dates and numbers
        - Important keywords
        """
        entities, _ = self.fallback_extractor.extract_entities_and_concepts(query)
        
        # Also extract important keywords (non-stopwords)
        words = re.findall(r'\b[a-zA-Z]+\b', query)
        
        stopwords = {
            "what", "when", "where", "who", "why", "how", "is", "are", "was", "were",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "about", "tell", "me", "show", "find"
        }
        
        keywords = [w for w in words if w.lower() not in stopwords and len(w) > 2]
        
        # Combine entities and keywords
        all_entities = list(set(entities + keywords))
        
        return all_entities
