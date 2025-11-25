"""
Path-Constrained Retrieval (PCR) algorithm implementation.
"""

from typing import List, Tuple, Optional, Set, Dict
import numpy as np
from .graph_builder import GraphBuilder
from .vector_index import VectorIndex
from .embedder import Embedder


class PathConstrainedRetrieval:
    """
    Core Path-Constrained Retrieval algorithm.
    
    Given an anchor node and a query, restricts candidate set to reachable nodes
    before performing semantic search.
    """
    
    def __init__(
        self,
        graph: GraphBuilder,
        vector_index: VectorIndex,
        embedder: Optional[Embedder] = None
    ):
        """
        Initialize PCR system.
        
        Args:
            graph: GraphBuilder instance with the knowledge graph
            vector_index: VectorIndex instance with node embeddings
            embedder: Optional embedder for query encoding
        """
        self.graph = graph
        self.vector_index = vector_index
        self.embedder = embedder
    
    def retrieve(
        self,
        anchor: str,
        query: str,
        k: int = 10,
        max_depth: Optional[int] = None,
        fallback: bool = True,
        hybrid: bool = False
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve nodes using path-constrained retrieval.
        
        Args:
            anchor: Anchor node ID to start from
            query: Query text
            k: Number of results to return
            max_depth: Maximum depth for reachability (None for unlimited)
            fallback: If True, fall back to global search if no reachable nodes
            hybrid: If True, combine vector and keyword search
            
        Returns:
            List of (node_id, score, metadata) tuples
        """
        # Get reachable nodes from anchor
        try:
            reachable_nodes = self.graph.reachable(anchor, max_depth=max_depth)
        except ValueError:
            if fallback:
                reachable_nodes = None
            else:
                return []
        
        # If no reachable nodes and fallback enabled, search globally
        if not reachable_nodes and fallback:
            reachable_nodes = None
        
        # Embed query
        if self.embedder:
            query_emb = self.embedder.embed(query)
        else:
            raise ValueError("Embedder required for query encoding")
        
        # Perform semantic search on candidates
        if hybrid:
            results = self._hybrid_search(query, query_emb, reachable_nodes, k)
        else:
            results = self._vector_search(query_emb, reachable_nodes, k)
        
        # Enrich with metadata
        enriched_results = []
        for node_id, score in results:
            node_data = self.graph.get_node(node_id)
            metadata = {
                'text': node_data.get('text', ''),
                'metadata': node_data.get('metadata', {}),
                'path_length': self.graph.get_shortest_path_length(anchor, node_id)
            }
            enriched_results.append((node_id, score, metadata))
        
        return enriched_results
    
    def _vector_search(
        self,
        query_emb: np.ndarray,
        candidates: Optional[Set[str]],
        k: int
    ) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search.
        
        Args:
            query_emb: Query embedding
            candidates: Optional candidate set
            k: Number of results
            
        Returns:
            List of (node_id, score) tuples
        """
        return self.vector_index.search(query_emb, candidates, k)
    
    def _hybrid_search(
        self,
        query: str,
        query_emb: np.ndarray,
        candidates: Optional[Set[str]],
        k: int
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid vector + keyword search.
        
        Args:
            query: Query text
            query_emb: Query embedding
            candidates: Optional candidate set
            k: Number of results
            
        Returns:
            List of (node_id, score) tuples
        """
        # Get vector search results
        vector_results = self._vector_search(query_emb, candidates, k * 2)
        
        # Simple keyword matching (can be enhanced with BM25, etc.)
        query_terms = set(query.lower().split())
        
        scored_results = []
        for node_id, vector_score in vector_results:
            node_data = self.graph.get_node(node_id)
            text = node_data.get('text', '').lower()
            
            # Count matching terms
            text_terms = set(text.split())
            keyword_score = len(query_terms & text_terms) / max(1, len(query_terms))
            
            # Combine scores (weighted average)
            combined_score = 0.7 * vector_score + 0.3 * keyword_score
            scored_results.append((node_id, combined_score))
        
        # Sort and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:k]
    
    def retrieve_with_paths(
        self,
        anchor: str,
        query: str,
        k: int = 10,
        max_depth: Optional[int] = None,
        include_paths: bool = True
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve nodes and include path information.
        
        Args:
            anchor: Anchor node ID
            query: Query text
            k: Number of results
            max_depth: Maximum depth
            include_paths: Whether to include path information
            
        Returns:
            List of (node_id, score, metadata) tuples with path info
        """
        results = self.retrieve(anchor, query, k, max_depth, fallback=True)
        
        if include_paths:
            enriched = []
            for node_id, score, metadata in results:
                paths = self.graph.get_all_paths(anchor, node_id, max_length=10)
                metadata['paths'] = paths[:5]  # Limit to 5 paths
                enriched.append((node_id, score, metadata))
            return enriched
        
        return results

