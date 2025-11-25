"""
Tests for graph_builder module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from src.graph_builder import GraphBuilder


def test_add_node():
    """Test adding nodes to graph."""
    graph = GraphBuilder()
    graph.add_node("node1", "Test text", metadata={"type": "test"})
    
    assert "node1" in graph.graph
    assert graph.get_node("node1")["text"] == "Test text"
    assert graph.get_node("node1")["metadata"]["type"] == "test"


def test_add_edge():
    """Test adding edges to graph."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_edge("node1", "node2")
    
    assert graph.graph.has_edge("node1", "node2")


def test_reachable():
    """Test reachability computation."""
    graph = GraphBuilder()
    
    # Create a chain: node1 -> node2 -> node3
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_node("node3", "Text 3")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    
    reachable = graph.reachable("node1")
    assert "node1" in reachable
    assert "node2" in reachable
    assert "node3" in reachable
    
    # Test max_depth
    reachable_depth1 = graph.reachable("node1", max_depth=1)
    assert "node1" in reachable_depth1
    assert "node2" in reachable_depth1
    assert "node3" not in reachable_depth1


def test_induced_subgraph():
    """Test induced subgraph extraction."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_node("node3", "Text 3")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    
    subgraph = graph.induced_subgraph("node1")
    assert subgraph.number_of_nodes() == 3
    assert subgraph.has_edge("node1", "node2")


def test_shortest_path():
    """Test shortest path computation."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_node("node3", "Text 3")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    
    assert graph.get_shortest_path_length("node1", "node3") == 2
    assert graph.get_shortest_path_length("node1", "node1") == 0


def test_save_load():
    """Test graph serialization."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1", embedding=np.array([1.0, 2.0, 3.0]))
    graph.add_node("node2", "Text 2")
    graph.add_edge("node1", "node2")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_graph.pkl"
        graph.save_graph(path)
        
        # Load into new graph
        new_graph = GraphBuilder()
        new_graph.load_graph(path)
        
        assert new_graph.graph.number_of_nodes() == 2
        assert new_graph.graph.has_edge("node1", "node2")
        assert new_graph.get_node("node1")["text"] == "Text 1"


def test_get_stats():
    """Test graph statistics."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_edge("node1", "node2")
    
    stats = graph.get_stats()
    assert stats["num_nodes"] == 2
    assert stats["num_edges"] == 1

