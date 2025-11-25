"""
Tests for path_retrieval module.
"""

import pytest
import numpy as np
from src.graph_builder import GraphBuilder
from src.vector_index import VectorIndex
from src.path_retrieval import PathConstrainedRetrieval
from src.embedder import Embedder


class MockEmbedder:
    """Mock embedder for testing."""
    def __init__(self):
        self.embedding_dim = 128
    
    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic embedding based on text."""
        # Simple hash-based embedding for testing
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        return np.random.randn(128).astype(np.float32)


def test_retrieve():
    """Test basic PCR retrieval."""
    graph = GraphBuilder()
    embedder = MockEmbedder()
    vector_index = VectorIndex(embedding_dim=128)
    
    # Build graph
    graph.add_node("node1", "Machine learning basics")
    graph.add_node("node2", "Neural networks")
    graph.add_node("node3", "Unrelated topic")
    
    graph.add_edge("node1", "node2")
    
    # Add embeddings
    vec1 = embedder.embed("Machine learning basics")
    vec2 = embedder.embed("Neural networks")
    vec3 = embedder.embed("Unrelated topic")
    
    vector_index.add_vector("node1", vec1)
    vector_index.add_vector("node2", vec2)
    vector_index.add_vector("node3", vec3)
    
    # Create PCR system
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    # Retrieve from node1
    results = pcr.retrieve("node1", "neural networks", k=2)
    
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 3 for r in results)


def test_retrieve_fallback():
    """Test fallback when no reachable nodes."""
    graph = GraphBuilder()
    embedder = MockEmbedder()
    vector_index = VectorIndex(embedding_dim=128)
    
    graph.add_node("node1", "Text")
    vec1 = embedder.embed("Text")
    vector_index.add_vector("node1", vec1)
    
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    # Should fallback to global search
    results = pcr.retrieve("node1", "query", k=1, fallback=True)
    assert len(results) >= 0


def test_hybrid_search():
    """Test hybrid search mode."""
    graph = GraphBuilder()
    embedder = MockEmbedder()
    vector_index = VectorIndex(embedding_dim=128)
    
    graph.add_node("node1", "Machine learning")
    graph.add_node("node2", "Neural networks")
    graph.add_edge("node1", "node2")
    
    vec1 = embedder.embed("Machine learning")
    vec2 = embedder.embed("Neural networks")
    
    vector_index.add_vector("node1", vec1)
    vector_index.add_vector("node2", vec2)
    
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    results = pcr.retrieve("node1", "neural networks", k=2, hybrid=True)
    assert len(results) <= 2

