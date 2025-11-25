"""
Tests for vector_index module.
"""

import pytest
import numpy as np
from src.vector_index import VectorIndex


def test_add_vector():
    """Test adding vectors to index."""
    index = VectorIndex(embedding_dim=128)
    vec = np.random.randn(128).astype(np.float32)
    
    index.add_vector("node1", vec)
    assert index.size() == 1


def test_search():
    """Test vector search."""
    index = VectorIndex(embedding_dim=128)
    
    # Add some vectors
    vec1 = np.random.randn(128).astype(np.float32)
    vec2 = np.random.randn(128).astype(np.float32)
    vec3 = np.random.randn(128).astype(np.float32)
    
    index.add_vector("node1", vec1)
    index.add_vector("node2", vec2)
    index.add_vector("node3", vec3)
    
    # Search with query similar to vec1
    query = vec1 + 0.1 * np.random.randn(128).astype(np.float32)
    results = index.search(query, k=2)
    
    assert len(results) == 2
    assert results[0][0] in ["node1", "node2", "node3"]


def test_search_with_candidates():
    """Test search with candidate filtering."""
    index = VectorIndex(embedding_dim=128)
    
    vec1 = np.random.randn(128).astype(np.float32)
    vec2 = np.random.randn(128).astype(np.float32)
    vec3 = np.random.randn(128).astype(np.float32)
    
    index.add_vector("node1", vec1)
    index.add_vector("node2", vec2)
    index.add_vector("node3", vec3)
    
    query = vec1
    candidates = {"node1", "node2"}
    
    results = index.search(query, candidates=candidates, k=10)
    
    assert all(node_id in candidates for node_id, _ in results)


def test_clear():
    """Test clearing the index."""
    index = VectorIndex(embedding_dim=128)
    vec = np.random.randn(128).astype(np.float32)
    
    index.add_vector("node1", vec)
    assert index.size() == 1
    
    index.clear()
    assert index.size() == 0

