"""
Text embedding module using OpenAI embeddings with caching.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from .utils import batch_process, ensure_dir

# Load environment variables from .env file
load_dotenv()


class Embedder:
    """
    Text embedder using OpenAI embeddings with caching support.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        batch_size: int = 100
    ):
        """
        Initialize the embedder.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            cache_dir: Directory to cache embeddings
            batch_size: Batch size for API calls
        """
        self.model = model
        # Use provided API key, or get from environment variable
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        if cache_dir:
            ensure_dir(cache_dir)
            self.cache_file = cache_dir / "embedding_cache.pkl"
            self._load_cache()
        else:
            self.cache = {}
    
    def _load_cache(self) -> None:
        """Load embedding cache from disk."""
        if self.cache_dir and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if self.cache_dir and self.cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model}:{text}".encode()).hexdigest()
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache
        cache_key = self._get_cache_key(text)
        if cache_key in self.cache:
            return np.array(self.cache[cache_key])
        
        # Call API
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Cache result
            self.cache[cache_key] = embedding
            if self.cache_dir:
                self._save_cache()
            
            return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to embed text: {e}")
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for all texts
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append((i, np.array(self.cache[cache_key])))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Embed uncached texts in batches
        if uncached_texts:
            for batch in batch_process(uncached_texts, self.batch_size):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    
                    for j, item in enumerate(response.data):
                        embedding = np.array(item.embedding, dtype=np.float32)
                        orig_idx = uncached_indices[len(embeddings) - len(uncached_texts) + j]
                        embeddings.append((orig_idx, embedding))
                        
                        # Cache
                        cache_key = self._get_cache_key(batch[j])
                        self.cache[cache_key] = embedding
                except Exception as e:
                    raise RuntimeError(f"Failed to embed batch: {e}")
            
            if self.cache_dir:
                self._save_cache()
        
        # Sort by original index and return just embeddings
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings from this model.
        
        Returns:
            Embedding dimension
        """
        # text-embedding-3-small has 1536 dimensions
        if "3-small" in self.model:
            return 1536
        elif "3-large" in self.model:
            return 3072
        elif "ada-002" in self.model:
            return 1536
        else:
            # Default or test with a dummy call
            test_emb = self.embed("test")
            return len(test_emb)

