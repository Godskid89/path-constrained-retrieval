#!/usr/bin/env python3
"""
Real-world case studies demonstrating PCR effectiveness.
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.embedder import Embedder
from src.graph_builder import GraphBuilder
from src.vector_index import VectorIndex
from src.path_retrieval import PathConstrainedRetrieval
from src.dataset_loader import DatasetLoader
from src.baselines import BM25Retriever, HybridRetriever

def case_study_1():
    """Case Study 1: Multi-hop reasoning in cloud computing."""
    print("="*60)
    print("Case Study 1: Multi-hop Reasoning")
    print("="*60)
    
    data_dir = Path("data")
    embedder = Embedder(cache_dir=Path("cache"))
    loader = DatasetLoader(data_dir, embedder=embedder)
    
    graph, vector_index, _ = loader.load_domain("tech")
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    anchor = "tech_node_000"
    query = "What are the advanced concepts in cloud computing?"
    
    print(f"\nQuery: {query}")
    print(f"Anchor: {anchor}")
    
    # PCR retrieval
    pcr_results = pcr.retrieve(anchor, query, k=5)
    
    print("\nPCR Results:")
    for i, (node_id, score, metadata) in enumerate(pcr_results, 1):
        path_len = metadata.get('path_length', 'N/A')
        print(f"{i}. {node_id} (score: {score:.3f}, path: {path_len})")
        print(f"   {metadata['text'][:80]}...")
    
    # Baseline comparison
    query_emb = embedder.embed(query)
    baseline_results = vector_index.search(query_emb, candidates=None, k=5)
    
    print("\nBaseline Vector Search Results:")
    for i, (node_id, score) in enumerate(baseline_results, 1):
        reachable = node_id in graph.reachable(anchor)
        print(f"{i}. {node_id} (score: {score:.3f}, reachable: {reachable})")
    
    return pcr_results, baseline_results

def case_study_2():
    """Case Study 2: Hallucination prevention."""
    print("\n" + "="*60)
    print("Case Study 2: Hallucination Prevention")
    print("="*60)
    
    data_dir = Path("data")
    embedder = Embedder(cache_dir=Path("cache"))
    loader = DatasetLoader(data_dir, embedder=embedder)
    
    graph, vector_index, _ = loader.load_domain("tech")
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    anchor = "tech_node_000"
    query = "What are the key principles of cloud computing?"
    
    print(f"\nQuery: {query}")
    print(f"Anchor: {anchor}")
    
    # PCR retrieval
    pcr_results = pcr.retrieve(anchor, query, k=5)
    pcr_nodes = [node_id for node_id, _, _ in pcr_results]
    
    # Baseline retrieval
    query_emb = embedder.embed(query)
    baseline_results = vector_index.search(query_emb, candidates=None, k=5)
    baseline_nodes = [node_id for node_id, _ in baseline_results]
    
    # Check reachability
    reachable = graph.reachable(anchor)
    pcr_hallucinations = [n for n in pcr_nodes if n not in reachable]
    baseline_hallucinations = [n for n in baseline_nodes if n not in reachable]
    
    print(f"\nPCR Results: {len(pcr_nodes)} nodes")
    print(f"  Hallucinations: {len(pcr_hallucinations)} ({len(pcr_hallucinations)/len(pcr_nodes)*100:.1f}%)")
    if pcr_hallucinations:
        print(f"  Unreachable nodes: {pcr_hallucinations}")
    
    print(f"\nBaseline Results: {len(baseline_nodes)} nodes")
    print(f"  Hallucinations: {len(baseline_hallucinations)} ({len(baseline_hallucinations)/len(baseline_nodes)*100:.1f}%)")
    if baseline_hallucinations:
        print(f"  Unreachable nodes: {baseline_hallucinations}")
    
    return pcr_results, baseline_results

def case_study_3():
    """Case Study 3: Path-aware retrieval."""
    print("\n" + "="*60)
    print("Case Study 3: Path-Aware Retrieval")
    print("="*60)
    
    data_dir = Path("data")
    embedder = Embedder(cache_dir=Path("cache"))
    loader = DatasetLoader(data_dir, embedder=embedder)
    
    graph, vector_index, _ = loader.load_domain("tech")
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    anchor = "tech_node_000"
    query = "What are cloud computing implementation details?"
    
    print(f"\nQuery: {query}")
    print(f"Anchor: {anchor}")
    
    # Retrieve with path information
    pcr_results = pcr.retrieve_with_paths(anchor, query, k=5, include_paths=True)
    
    print("\nPCR Results with Path Information:")
    for i, (node_id, score, metadata) in enumerate(pcr_results, 1):
        path_len = metadata.get('path_length', 'N/A')
        paths = metadata.get('paths', [])
        print(f"{i}. {node_id} (score: {score:.3f}, path length: {path_len})")
        if paths:
            print(f"   Example path: {' -> '.join(paths[0][:3])}...")
    
    return pcr_results

def run_all_case_studies():
    """Run all case studies."""
    print("\n" + "="*60)
    print("REAL-WORLD CASE STUDIES")
    print("="*60)
    
    case_study_1()
    case_study_2()
    case_study_3()
    
    print("\n" + "="*60)
    print("Case studies complete!")
    print("="*60)

if __name__ == "__main__":
    run_all_case_studies()

