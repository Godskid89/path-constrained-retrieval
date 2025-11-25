"""
Vector index module for semantic search using FAISS or sklearn.
"""

from typing import List, Optional, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorIndex:
    """
    Vector index for efficient similarity search.
    """
    
    def __init__(self, embedding_dim: int, use_faiss: bool = True):
        """
        Initialize the vector index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            use_faiss: Whether to use FAISS (if available) or sklearn
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss
        self.node_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        
        if use_faiss:
            try:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
                self.faiss_available = True
            except ImportError:
                print("FAISS not available, falling back to sklearn")
                self.faiss_available = False
                self.faiss_index = None
        else:
            self.faiss_available = False
            self.faiss_index = None
    
    def add_vector(self, node_id: str, embedding: np.ndarray) -> None:
        """
        Add a vector to the index.
        
        Args:
            node_id: Node identifier
            embedding: Embedding vector (must match embedding_dim)
        """
        if len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {len(embedding)} does not match "
                f"expected dimension {self.embedding_dim}"
            )
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        embedding = embedding.astype(np.float32).reshape(1, -1)
        
        self.node_ids.append(node_id)
        
        if self.faiss_available and self.faiss_index is not None:
            self.faiss_index.add(embedding)
        else:
            if self.embeddings is None:
                self.embeddings = embedding
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
    
    def search(
        self,
        query_emb: np.ndarray,
        candidates: Optional[Set[str]] = None,
        k: int = 10
    ) -> List[tuple]:
        """
        Search for similar vectors.
        
        Args:
            query_emb: Query embedding vector
            candidates: Optional set of candidate node IDs to restrict search to
            k: Number of results to return
            
        Returns:
            List of (node_id, score) tuples, sorted by score descending
        """
        if len(self.node_ids) == 0:
            return []
        
        # Normalize query
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.astype(np.float32).reshape(1, -1)
        
        if self.faiss_available and self.faiss_index is not None:
            # FAISS search
            scores, indices = self.faiss_index.search(query_emb, min(k * 2, len(self.node_ids)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.node_ids):
                    node_id = self.node_ids[idx]
                    if candidates is None or node_id in candidates:
                        results.append((node_id, float(score)))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
        else:
            # sklearn search
            if self.embeddings is None:
                return []
            
            similarities = cosine_similarity(query_emb, self.embeddings)[0]
            
            # Create list of (node_id, score) pairs
            results = [(self.node_ids[i], float(similarities[i])) 
                      for i in range(len(self.node_ids))]
            
            # Filter by candidates if provided
            if candidates is not None:
                results = [(node_id, score) for node_id, score in results 
                          if node_id in candidates]
            
            # Sort by score and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
    
    def get_vector(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Embedding vector or None if not found
        """
        if node_id not in self.node_ids:
            return None
        
        idx = self.node_ids.index(node_id)
        
        if self.faiss_available and self.faiss_index is not None:
            # FAISS doesn't support direct retrieval, need to reconstruct
            # For now, return None - embeddings should be stored in graph
            return None
        else:
            return self.embeddings[idx]
    
    def size(self) -> int:
        """
        Get the number of vectors in the index.
        
        Returns:
            Number of indexed vectors
        """
        return len(self.node_ids)
    
    def clear(self) -> None:
        """Clear all vectors from the index."""
        self.node_ids = []
        self.embeddings = None
        if self.faiss_available and self.faiss_index is not None:
            self.faiss_index.reset()

