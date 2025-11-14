"""
TERAG Embedding Manager

Manages embeddings for passages, entities, and concepts using OpenAI's text-embedding-3-small.
Provides persistent caching to avoid re-computing embeddings.
"""

import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import time

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embeddings with OpenAI text-embedding-3-small and persistent caching
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        cache_dir: str = "embeddings_cache",
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize embedding manager
        
        Args:
            api_key: OpenAI API key (if None, loads from environment)
            model: OpenAI embedding model name
            cache_dir: Directory for caching embeddings
            batch_size: Batch size for API calls
            max_retries: Maximum retries for failed API calls
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Initialize OpenAI client
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache files
        self.embeddings_cache_file = self.cache_dir / "embeddings_cache.pkl"
        self.metadata_cache_file = self.cache_dir / "embedding_metadata.json"
        
        # In-memory cache
        self.embeddings_cache = {}
        self.metadata = {
            "model": self.model,
            "cache_created": datetime.now().isoformat(),
            "total_embeddings": 0,
            "total_api_calls": 0,
            "total_cost_estimate": 0.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"Embedding manager initialized with model: {self.model}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Cached embeddings: {len(self.embeddings_cache)}")
    
    def _load_cache(self):
        """Load embeddings and metadata from disk"""
        # Load embeddings cache
        if self.embeddings_cache_file.exists():
            try:
                with open(self.embeddings_cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")
                self.embeddings_cache = {}
        
        # Load metadata
        if self.metadata_cache_file.exists():
            try:
                with open(self.metadata_cache_file, 'r') as f:
                    cached_metadata = json.load(f)
                    self.metadata.update(cached_metadata)
                logger.info(f"Loaded embedding metadata")
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")
    
    def _save_cache(self):
        """Save embeddings and metadata to disk"""
        try:
            # Save embeddings
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            
            # Update and save metadata
            self.metadata.update({
                "total_embeddings": len(self.embeddings_cache),
                "last_updated": datetime.now().isoformat()
            })
            
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                
            logger.debug("Embedding cache saved")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding as numpy array, or None if failed
        """
        text_hash = self._get_text_hash(text)
        
        # Check cache first
        if text_hash in self.embeddings_cache:
            logger.debug(f"Using cached embedding for text: {text[:50]}...")
            return self.embeddings_cache[text_hash]
        
        # Get embedding from API
        embedding = self._call_openai_embedding([text])
        if embedding is not None and len(embedding) > 0:
            embedding_vector = np.array(embedding[0])
            
            # Cache the result
            self.embeddings_cache[text_hash] = embedding_vector
            self._save_cache()
            
            return embedding_vector
        
        return None
    
    def embed_texts_batch(self, texts: List[str], show_progress: bool = True) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple texts efficiently
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress
            
        Returns:
            Dict mapping text -> embedding vector
        """
        if not texts:
            return {}
        
        results = {}
        uncached_texts = []
        uncached_hashes = []
        
        # Check cache for all texts
        for text in texts:
            text_hash = self._get_text_hash(text)
            if text_hash in self.embeddings_cache:
                results[text] = self.embeddings_cache[text_hash]
            else:
                uncached_texts.append(text)
                uncached_hashes.append(text_hash)
        
        if show_progress:
            logger.info(f"Embedding texts: {len(uncached_texts)} new, {len(results)} cached")
        
        # Process uncached texts in batches
        for i in range(0, len(uncached_texts), self.batch_size):
            batch_texts = uncached_texts[i:i + self.batch_size]
            batch_hashes = uncached_hashes[i:i + self.batch_size]
            
            if show_progress:
                logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(uncached_texts) - 1) // self.batch_size + 1}")
            
            # Get embeddings for batch
            embeddings = self._call_openai_embedding(batch_texts)
            
            if embeddings is not None and len(embeddings) == len(batch_texts):
                # Store results
                for text, text_hash, embedding in zip(batch_texts, batch_hashes, embeddings):
                    embedding_vector = np.array(embedding)
                    results[text] = embedding_vector
                    self.embeddings_cache[text_hash] = embedding_vector
                
                # Save cache periodically
                self._save_cache()
            else:
                logger.error(f"Failed to get embeddings for batch {i // self.batch_size + 1}")
        
        if show_progress:
            logger.info(f"Completed embedding {len(texts)} texts")
        
        return results
    
    def _call_openai_embedding(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Call OpenAI embedding API with retries
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors, or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    encoding_format="float"
                )
                
                # Extract embeddings
                embeddings = [data.embedding for data in response.data]
                
                # Update metadata
                self.metadata["total_api_calls"] += 1
                # Estimate cost (text-embedding-3-small: ~$0.00002 per 1K tokens)
                total_tokens = sum(len(text.split()) * 1.3 for text in texts)  # Rough token estimate
                cost_estimate = (total_tokens / 1000) * 0.00002
                self.metadata["total_cost_estimate"] += cost_estimate
                
                logger.debug(f"OpenAI embedding API call successful: {len(texts)} texts, ~${cost_estimate:.6f}")
                
                return embeddings
                
            except Exception as e:
                logger.warning(f"OpenAI API call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"All {self.max_retries} attempts failed for embedding API call")
        
        return None
    
    def embed_graph_entities(self, graph) -> Dict[str, np.ndarray]:
        """
        Embed all entities/concepts in a TERAG graph
        
        Args:
            graph: TERAGGraph instance
            
        Returns:
            Dict mapping entity_text -> embedding
        """
        logger.info("Embedding all graph entities/concepts...")
        
        # Get all unique entity/concept texts
        entity_texts = []
        for concept in graph.concepts.values():
            entity_texts.append(concept.concept_text)
        
        # Remove duplicates while preserving order
        unique_texts = list(dict.fromkeys(entity_texts))
        
        logger.info(f"Found {len(unique_texts)} unique entities/concepts to embed")
        
        # Embed all texts
        embeddings = self.embed_texts_batch(unique_texts)
        
        logger.info(f"Successfully embedded {len(embeddings)} entities/concepts")
        return embeddings
    
    def embed_graph_passages(self, graph) -> Dict[str, np.ndarray]:
        """
        Embed all passages in a TERAG graph
        
        Args:
            graph: TERAGGraph instance
            
        Returns:
            Dict mapping passage_id -> embedding
        """
        logger.info("Embedding all graph passages...")
        
        # Get all passage texts
        passage_texts = []
        passage_ids = []
        
        for passage_id, passage in graph.passages.items():
            passage_texts.append(passage.content)
            passage_ids.append(passage_id)
        
        logger.info(f"Found {len(passage_texts)} passages to embed")
        
        # Embed all passages
        embeddings_by_text = self.embed_texts_batch(passage_texts)
        
        # Map back to passage IDs
        embeddings_by_id = {}
        for passage_id, passage_text in zip(passage_ids, passage_texts):
            if passage_text in embeddings_by_text:
                embeddings_by_id[passage_id] = embeddings_by_text[passage_text]
        
        logger.info(f"Successfully embedded {len(embeddings_by_id)} passages")
        return embeddings_by_id
    
    def get_cache_statistics(self) -> Dict:
        """Get statistics about the embedding cache"""
        return {
            "total_cached_embeddings": len(self.embeddings_cache),
            "cache_size_mb": self.embeddings_cache_file.stat().st_size / (1024 * 1024) if self.embeddings_cache_file.exists() else 0,
            "model": self.metadata.get("model"),
            "total_api_calls": self.metadata.get("total_api_calls", 0),
            "total_cost_estimate": self.metadata.get("total_cost_estimate", 0.0),
            "cache_created": self.metadata.get("cache_created"),
            "last_updated": self.metadata.get("last_updated")
        }
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        self.embeddings_cache.clear()
        
        if self.embeddings_cache_file.exists():
            self.embeddings_cache_file.unlink()
        if self.metadata_cache_file.exists():
            self.metadata_cache_file.unlink()
        
        logger.info("Embedding cache cleared")


# Convenience functions
def create_embedding_manager(**kwargs) -> EmbeddingManager:
    """Create embedding manager with default settings"""
    return EmbeddingManager(**kwargs)


def embed_graph_data(graph, embedding_manager: Optional[EmbeddingManager] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Embed all entities and passages in a graph
    
    Args:
        graph: TERAGGraph instance
        embedding_manager: Optional pre-configured embedding manager
        
    Returns:
        (entity_embeddings, passage_embeddings)
    """
    if embedding_manager is None:
        embedding_manager = EmbeddingManager()
    
    logger.info("Embedding graph data...")
    
    # Embed entities
    entity_embeddings = embedding_manager.embed_graph_entities(graph)
    
    # Embed passages  
    passage_embeddings = embedding_manager.embed_graph_passages(graph)
    
    logger.info("Graph embedding completed")
    
    return entity_embeddings, passage_embeddings


if __name__ == "__main__":
    # Test embedding manager
    print("🔗 EMBEDDING MANAGER TEST")
    print("=" * 60)
    
    try:
        # Initialize manager
        manager = EmbeddingManager()
        
        # Test single embedding
        test_text = "Apple Inc reported strong revenue growth in Q4 2024"
        print(f"Testing single embedding...")
        embedding = manager.embed_text(test_text)
        
        if embedding is not None:
            print(f"✅ Single embedding successful: shape {embedding.shape}")
        else:
            print("❌ Single embedding failed")
        
        # Test batch embedding
        test_texts = [
            "Microsoft Corporation announced AI initiatives",
            "Revenue growth exceeded expectations",
            "Tim Cook emphasized innovation strategy",
            "Q4 financial results were strong"
        ]
        
        print(f"\nTesting batch embedding...")
        embeddings = manager.embed_texts_batch(test_texts)
        
        if len(embeddings) == len(test_texts):
            print(f"✅ Batch embedding successful: {len(embeddings)} embeddings")
            
            # Test embedding similarity
            emb1 = embeddings[test_texts[0]]
            emb2 = embeddings[test_texts[1]]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"Similarity between text 1 and 2: {similarity:.3f}")
        else:
            print("❌ Batch embedding failed")
        
        # Show cache statistics
        stats = manager.get_cache_statistics()
        print(f"\n📊 Cache Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure to set OPENAI_API_KEY environment variable")