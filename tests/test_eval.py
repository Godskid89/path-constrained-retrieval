"""
Tests for evaluation module.
"""

import pytest
from src.graph_builder import GraphBuilder
from src.evaluation import Evaluator


def test_relevance_at_k():
    """Test relevance@k metric."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_node("node3", "Text 3")
    
    evaluator = Evaluator(graph, ground_truth={"q1": ["node1", "node2"]})
    
    retrieved = ["node1", "node3", "node2"]
    relevant = ["node1", "node2"]
    
    rel_at_1 = evaluator.relevance_at_k(retrieved, relevant, k=1)
    assert rel_at_1 == 1.0  # node1 is relevant
    
    rel_at_2 = evaluator.relevance_at_k(retrieved, relevant, k=2)
    assert rel_at_2 == 0.5  # Top 2 are [node1, node3], only node1 is relevant, so 1/2 = 0.5


def test_hallucination_score():
    """Test structural inconsistency score (formerly hallucination score)."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_node("node3", "Text 3")
    graph.add_edge("node1", "node2")
    
    evaluator = Evaluator(graph)
    
    # node2 is reachable, node3 is not
    retrieved = ["node2", "node3"]
    # Note: method is still called hallucination_score internally
    score = evaluator.hallucination_score(retrieved, "node1")
    
    assert 0.0 <= score <= 1.0
    assert score > 0.0  # node3 is unreachable


def test_multi_hop_consistency():
    """Test multi-hop consistency."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_node("node3", "Text 3")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    
    evaluator = Evaluator(graph)
    
    retrieved = ["node2", "node3"]  # Both at consistent distances
    consistency = evaluator.multi_hop_consistency(retrieved, "node1")
    
    assert 0.0 <= consistency <= 1.0


def test_graph_distance_penalty():
    """Test graph distance penalty."""
    graph = GraphBuilder()
    graph.add_node("node1", "Text 1")
    graph.add_node("node2", "Text 2")
    graph.add_node("node3", "Text 3")
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    
    evaluator = Evaluator(graph)
    
    retrieved = ["node2", "node3"]
    penalty = evaluator.graph_distance_penalty(retrieved, "node1")
    
    assert penalty >= 0.0

