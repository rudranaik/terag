"""
LLM Providers for TERAG

Abstracts different LLM APIs (Groq, OpenAI, Anthropic) into a common interface.
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized response from LLM"""
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    processing_time: float
    timestamp: str
    success: bool
    error_message: Optional[str] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = None, silent_fallback: bool = False):
        self.api_key = api_key
        self.model = model
        self.silent_fallback = silent_fallback

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate text from LLM"""
        pass
    
    def extract_entities_and_concepts(
        self, 
        text: str, 
        document_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[str], LLMResponse]:
        """
        Extract entities/concepts using this provider.
        Default implementation uses the prompt templates from GroqClient.
        """
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_extraction_prompt(text, document_context)
        
        response = self.generate(user_prompt, system_prompt)
        
        if not response.success:
            return [], [], response
            
        entities, concepts = self._parse_extraction_response(response.content)
        return entities, concepts, response

    def _create_system_prompt(self) -> str:
        return """You are an expert at extracting named entities and key concepts from text documents.

Your task is to identify:
1. ENTITIES: Named entities like people, organizations, locations, dates, products
2. CONCEPTS: Important domain-specific terms, technical concepts, and key topics

Guidelines:
- Be precise and avoid generic terms.
- Focus on business-relevant entities and concepts
- Normalize similar entities (e.g., "Apple Inc" and "Apple" -> "Apple Inc")
- Disambigute proper nouns from generic nouns

Return your response as valid JSON in this exact format:
{
  "entities": ["entity1", "entity2", ...],
  "concepts": ["concept1", "concept2", ...]
}"""

    def _create_extraction_prompt(self, text: str, document_context: Optional[Dict] = None) -> str:
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
        """Parse JSON response"""
        cleaned = response.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        elif cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        try:
            data = json.loads(cleaned)
            entities = data.get("entities", [])
            concepts = data.get("concepts", [])
            return entities, concepts
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return [], []


class GroqProvider(LLMProvider):
    """Provider for Groq API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b", silent_fallback: bool = False):
        super().__init__(api_key, model, silent_fallback)
        try:
            from groq import Groq
            self.api_key = api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                if not silent_fallback:
                    raise ValueError("Groq API key not found")
                self.client = None
            else:
                self.client = Groq(api_key=self.api_key)
        except ImportError:
            if not silent_fallback:
                raise ImportError("groq package not installed")
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if not self.client:
            return LLMResponse("", self.model, 0, 0, 0, "", False, "Client not initialized")
            
        try:
            start_time = time.time()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            content = completion.choices[0].message.content
            duration = time.time() - start_time
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=0, # Need actual token counting if critical
                cost_estimate=0.0,
                processing_time=duration,
                timestamp=datetime.now().isoformat(),
                success=True
            )
        except Exception as e:
            return LLMResponse("", self.model, 0, 0, 0, "", False, str(e))


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo", silent_fallback: bool = False):
        super().__init__(api_key, model, silent_fallback)
        try:
            from openai import OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                if not silent_fallback:
                    raise ValueError("OpenAI API key not found")
                self.client = None
            else:
                self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            if not silent_fallback:
                raise ImportError("openai package not installed")
            self.client = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if not self.client:
            return LLMResponse("", self.model, 0, 0, 0, "", False, "Client not initialized")
            
        try:
            start_time = time.time()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            content = completion.choices[0].message.content
            duration = time.time() - start_time
            
            # Simple cost tracking
            tokens = completion.usage.total_tokens if completion.usage else 0
            
            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens,
                cost_estimate=0.0, # Implement pricing map if needed
                processing_time=duration,
                timestamp=datetime.now().isoformat(),
                success=True
            )
        except Exception as e:
            return LLMResponse("", self.model, 0, 0, 0, "", False, str(e))


def get_llm_provider(
    provider_name: str = "groq", 
    api_key: Optional[str] = None, 
    model: Optional[str] = None,
    silent_fallback: bool = False
) -> LLMProvider:
    """Factory to get LLM provider"""
    if provider_name == "openai":
        return OpenAIProvider(api_key, model or "gpt-3.5-turbo", silent_fallback)
    elif provider_name == "groq":
        return GroqProvider(api_key, model or "openai/gpt-oss-20b", silent_fallback)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
