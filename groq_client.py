"""
Groq API Client for TERAG

Provides a wrapper for Groq API calls with:
- GPT OSS 20B model support
- Retry logic and error handling
- Cost tracking
- Response validation
"""

import json
import time
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM API call"""
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    processing_time: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None


class GroqClient:
    """
    Groq API client wrapper for TERAG entity extraction
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-20b",  # Default model
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30
    ):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (if None, loads from env GROQ_API_KEY)
            model: Model name to use
            max_retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
        """
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not available. Install with: pip install groq")
        
        # Get API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not provided. Set GROQ_API_KEY environment variable or pass api_key parameter")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Cost tracking (approximate values - update based on actual Groq pricing)
        self.pricing = {
            "openai/gpt-oss-20b": {"input": 0.0002, "output": 0.0002}  # Placeholder pricing
        }
        
        logger.info(f"Groq client initialized with model: {self.model}")
    
    def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Make a call to the Groq LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLMResponse object with results and metadata
        """
        start_time = time.time()
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Groq API call attempt {attempt + 1}/{self.max_retries}")
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=self.timeout
                )
                
                # Extract response content
                content = response.choices[0].message.content
                
                # Calculate metrics
                processing_time = time.time() - start_time
                tokens_used = self._estimate_tokens(prompt, content)
                cost_estimate = self._calculate_cost(tokens_used)
                
                logger.debug(f"Groq API call successful: {tokens_used} tokens, ${cost_estimate:.4f}")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    tokens_used=tokens_used,
                    cost_estimate=cost_estimate,
                    processing_time=processing_time,
                    timestamp=datetime.now().isoformat(),
                    success=True
                )
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Groq API call attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
        # All retries failed
        processing_time = time.time() - start_time
        logger.error(f"All Groq API call attempts failed. Last error: {last_error}")
        
        return LLMResponse(
            content="",
            model=self.model,
            tokens_used=0,
            cost_estimate=0.0,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            success=False,
            error_message=last_error
        )
    
    def _estimate_tokens(self, prompt: str, response: str) -> int:
        """
        Estimate token count for prompt and response
        Rough approximation: 1 token ≈ 0.75 words
        """
        prompt_words = len(prompt.split())
        response_words = len(response.split())
        return int((prompt_words + response_words) * 1.33)  # Convert words to tokens
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate estimated cost for API call"""
        pricing = self.pricing.get(self.model, {"input": 0.0002, "output": 0.0002})
        
        # Assume roughly 60% input, 40% output (rough estimate)
        input_tokens = int(tokens * 0.6)
        output_tokens = int(tokens * 0.4)
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def extract_entities_and_concepts(
        self,
        text: str,
        document_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[str], LLMResponse]:
        """
        Extract entities and concepts from text using LLM
        
        Args:
            text: Text passage to analyze
            document_context: Optional document metadata for context
            
        Returns:
            (entities, concepts, llm_response)
        """
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Create user prompt with context
        user_prompt = self._create_extraction_prompt(text, document_context)
        
        # Call LLM
        response = self.call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        if not response.success:
            logger.error(f"LLM call failed: {response.error_message}")
            return [], [], response
        
        # Parse response
        entities, concepts = self._parse_extraction_response(response.content)
        
        logger.info(f"Extracted {len(entities)} entities and {len(concepts)} concepts")
        return entities, concepts, response
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for entity extraction"""
        return """You are an expert at extracting named entities and key concepts from text documents.

Your task is to identify:
1. ENTITIES: Named entities like people, organizations, locations, dates, products, type of financial figures (not financial amounts or figures themselves)
2. CONCEPTS: Important domain-specific terms, technical concepts, and key topics

Guidelines:
- Be precise and avoid generic terms.
- Focus on business-relevant entities and concepts
- Normalize similar entities (e.g., "Apple Inc" and "Apple" → "Apple Inc")
- Disambigute proper nouns from generic nouns
- Extract concepts that would be useful for document retrieval

Return your response as valid JSON in this exact format:
{
  "entities": ["entity1", "entity2", ...],
  "concepts": ["concept1", "concept2", ...]
}"""

    def _create_extraction_prompt(
        self, 
        text: str, 
        document_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create user prompt for extraction"""
        context_info = ""
        if document_context:
            doc_type = document_context.get('document_type', 'unknown')
            source = document_context.get('source', 'unknown')
            context_info = f"\n\nDocument Context:\n- Type: {doc_type}\n- Source: {source}"
        
        return f"""Extract named entities and key concepts from this text passage: {context_info}

Text:
{text}

Extract entities and concepts that would be useful for:
1. Finding similar passages in other documents
2. Answering questions about this content
3. Understanding the main topics and subjects discussed

Remember to return valid JSON with "entities" and "concepts" arrays."""

    def _parse_extraction_response(self, response: str) -> Tuple[List[str], List[str]]:
        """
        Parse LLM response to extract entities and concepts
        
        Returns:
            (entities, concepts)
        """
        # Clean response - remove markdown code blocks if present
        cleaned_response = response.strip()
        
        # Remove markdown code blocks
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]   # Remove ```
            
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
        cleaned_response = cleaned_response.strip()
        
        try:
            # Try to parse as JSON
            data = json.loads(cleaned_response)
            
            entities = data.get("entities", [])
            concepts = data.get("concepts", [])
            
            # Clean and validate
            entities = [e.strip() for e in entities if e and isinstance(e, str) and len(e.strip()) > 1]
            concepts = [c.strip() for c in concepts if c and isinstance(c, str) and len(c.strip()) > 1]
            
            logger.debug(f"Successfully parsed JSON: {len(entities)} entities, {len(concepts)} concepts")
            return entities, concepts
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}. Attempting fallback parsing")
            
            # Fallback: Try to extract from text format
            entities, concepts = self._fallback_parse(response)
            return entities, concepts
    
    def _fallback_parse(self, response: str) -> Tuple[List[str], List[str]]:
        """Fallback parser for non-JSON responses"""
        entities = []
        concepts = []
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for section headers
            if 'entities' in line.lower():
                current_section = 'entities'
                continue
            elif 'concepts' in line.lower():
                current_section = 'concepts'
                continue
            
            # Extract items (look for bullets, dashes, or quoted items)
            if line.startswith(('- ', '• ', '* ', '"')) or line.endswith('"'):
                item = line.lstrip('- •*"').rstrip('"').strip()
                if item and len(item) > 1:
                    if current_section == 'entities':
                        entities.append(item)
                    elif current_section == 'concepts':
                        concepts.append(item)
        
        return entities, concepts
    
    def batch_extract(
        self,
        passages: List[Dict[str, Any]],
        batch_size: int = 1
    ) -> List[Tuple[List[str], List[str], LLMResponse]]:
        """
        Process multiple passages in batches
        
        Args:
            passages: List of passage dictionaries with 'content' and optional metadata
            batch_size: Number of passages to process per batch (currently 1 for individual processing)
            
        Returns:
            List of (entities, concepts, response) tuples
        """
        results = []
        
        for i, passage in enumerate(passages):
            logger.info(f"Processing passage {i+1}/{len(passages)}")
            
            content = passage.get('content', '')
            document_metadata = passage.get('document_metadata', {})
            
            entities, concepts, response = self.extract_entities_and_concepts(
                content, document_metadata
            )
            
            results.append((entities, concepts, response))
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        return results


def test_groq_client():
    """Test the Groq client with sample text"""
    # Test text
    test_text = """
    Apple Inc announced strong quarterly revenue of $120 billion in Q4 2024, 
    representing a 15% increase year-over-year. The technology giant's iPhone division 
    contributed significantly to this growth, while services revenue reached $25 billion. 
    CEO Tim Cook highlighted the company's expansion in artificial intelligence and 
    machine learning capabilities during the earnings call.
    """
    
    try:
        # Initialize client
        client = GroqClient()
        
        print("Testing Groq client...")
        print(f"Model: {client.model}")
        print(f"Test text length: {len(test_text)} characters")
        
        # Extract entities and concepts
        entities, concepts, response = client.extract_entities_and_concepts(test_text)
        
        print(f"\nExtraction Results:")
        print(f"Success: {response.success}")
        print(f"Processing time: {response.processing_time:.2f}s")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Cost estimate: ${response.cost_estimate:.4f}")
        
        print(f"\nEntities ({len(entities)}):")
        for entity in entities:
            print(f"  - {entity}")
        
        print(f"\nConcepts ({len(concepts)}):")
        for concept in concepts:
            print(f"  - {concept}")
            
        print(f"\nRaw LLM response:")
        print(response.content)
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_groq_client()