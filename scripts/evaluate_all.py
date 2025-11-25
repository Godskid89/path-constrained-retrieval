#!/usr/bin/env python3
"""
Comprehensive evaluation script for Path-Constrained Retrieval.
Generates full results for ArXiv paper.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.embedder import Embedder
from src.graph_builder import GraphBuilder
from src.vector_index import VectorIndex
from src.path_retrieval import PathConstrainedRetrieval
from src.dataset_loader import DatasetLoader
from src.evaluation import Evaluator
from src.baselines import BM25Retriever, HybridRetriever
from src.statistics import aggregate_with_statistics, compare_methods_statistically
from src.ablation import AblationStudy
from src.benchmark import PerformanceBenchmark

def create_baseline_retrievers(graph, vector_index, embedder):
    """Create baseline retrieval methods."""
    baselines = {}
    
    # BM25 baseline
    bm25 = BM25Retriever()
    for node_id in graph.graph.nodes():
        node_data = graph.get_node(node_id)
        bm25.add_document(node_id, node_data.get('text', ''))
    bm25.build_index()
    baselines['BM25'] = lambda query, k: bm25.search(query, k=k)
    
    # Hybrid baseline
    hybrid = HybridRetriever(vector_index, bm25, embedder, alpha=0.7)
    baselines['Hybrid'] = lambda query, k: hybrid.search(query, k=k)
    
    # Vector search baseline
    baselines['Vector'] = lambda query, k: vector_index.search(
        embedder.embed(query), candidates=None, k=k
    )
    
    return baselines

def evaluate_domain(domain, data_dir, embedder, output_dir):
    """Evaluate a single domain."""
    print(f"\n{'='*60}")
    print(f"Evaluating domain: {domain.upper()}")
    print(f"{'='*60}")
    
    # Load domain
    loader = DatasetLoader(data_dir, embedder=embedder)
    graph, vector_index, metadata = loader.load_domain(domain)
    
    print(f"Loaded {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")
    
    # Initialize PCR
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    # Load queries
    queries_data = loader.load_queries()
    queries = queries_data.get(domain, [])
    
    if len(queries) == 0:
        print(f"No queries found for {domain}")
        return None
    
    print(f"Evaluating on {len(queries)} queries")
    
    # Prepare ground truth
    ground_truth = {}
    for q in queries:
        ground_truth[q['id']] = q.get('relevant_nodes', [])
    
    evaluator = Evaluator(graph, ground_truth=ground_truth)
    
    # Create baselines
    baselines = create_baseline_retrievers(graph, vector_index, embedder)
    
    # Evaluate PCR
    print("\nEvaluating PCR...")
    pcr_results = []
    for query_data in queries:
        query_id = query_data.get('id', '')
        anchor = query_data.get('anchor', '')
        query_text = query_data.get('query', '')
        
        pcr_retrieved = pcr.retrieve(anchor, query_text, k=10)
        retrieved_ids = [node_id for node_id, _, _ in pcr_retrieved]
        
        metrics = evaluator.evaluate_query(query_id, anchor, retrieved_ids, k_values=[1, 5, 10])
        metrics['method'] = 'PCR'
        metrics['query_id'] = query_id
        metrics['domain'] = domain
        pcr_results.append(metrics)
    
    # Evaluate baselines
    baseline_results = []
    for baseline_name, baseline_func in baselines.items():
        print(f"Evaluating {baseline_name}...")
        for query_data in queries:
            query_id = query_data.get('id', '')
            anchor = query_data.get('anchor', '')
            query_text = query_data.get('query', '')
            
            try:
                baseline_retrieved = baseline_func(query_text, k=10)
                retrieved_ids = [node_id for node_id, _ in baseline_retrieved]
            except Exception as e:
                print(f"Error in {baseline_name}: {e}")
                retrieved_ids = []
            
            metrics = evaluator.evaluate_query(query_id, anchor, retrieved_ids, k_values=[1, 5, 10])
            metrics['method'] = baseline_name
            metrics['query_id'] = query_id
            metrics['domain'] = domain
            baseline_results.append(metrics)
    
    # Combine results
    all_results = pcr_results + baseline_results
    results_df = pd.DataFrame(all_results)
    
    # Aggregate results
    aggregated = evaluator.aggregate_results(results_df)
    
    # Statistical comparison
    stats_comparison = compare_methods_statistically(results_df, metric='relevance@10')
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / f"{domain}_detailed_results.csv", index=False)
    aggregated.to_csv(output_dir / f"{domain}_aggregated.csv")
    stats_comparison.to_csv(output_dir / f"{domain}_statistics.csv", index=False)
    
    print(f"\n{domain.upper()} Results:")
    print(aggregated)
    print(f"\nStatistical Comparison (relevance@10):")
    print(stats_comparison)
    
    return results_df, aggregated, stats_comparison

def run_ablation_studies(domain, data_dir, embedder, output_dir):
    """Run ablation studies."""
    print(f"\n{'='*60}")
    print(f"Ablation Studies: {domain.upper()}")
    print(f"{'='*60}")
    
    # Load domain
    loader = DatasetLoader(data_dir, embedder=embedder)
    graph, vector_index, metadata = loader.load_domain(domain)
    
    # Initialize PCR
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    # Load queries
    queries_data = loader.load_queries()
    queries = queries_data.get(domain, [])[:10]  # Use subset for ablation
    
    if len(queries) == 0:
        return None
    
    # Prepare ground truth
    ground_truth = {}
    for q in queries:
        ground_truth[q['id']] = q.get('relevant_nodes', [])
    
    evaluator = Evaluator(graph, ground_truth=ground_truth)
    ablation = AblationStudy(pcr, evaluator, queries)
    
    # Run ablation studies
    print("Running max_depth ablation...")
    depth_results = ablation.study_max_depth(depth_values=[None, 1, 2, 3, 5])
    depth_results.to_csv(output_dir / f"{domain}_ablation_depth.csv", index=False)
    
    print("Running hybrid search ablation...")
    hybrid_results = ablation.study_hybrid_search()
    hybrid_results.to_csv(output_dir / f"{domain}_ablation_hybrid.csv", index=False)
    
    print("Running comprehensive ablation...")
    all_results = ablation.study_all_components()
    all_results.to_csv(output_dir / f"{domain}_ablation_all.csv", index=False)
    
    return depth_results, hybrid_results, all_results

def run_benchmarks(domain, data_dir, embedder, output_dir):
    """Run performance benchmarks."""
    print(f"\n{'='*60}")
    print(f"Performance Benchmark: {domain.upper()}")
    print(f"{'='*60}")
    
    # Load domain
    loader = DatasetLoader(data_dir, embedder=embedder)
    graph, vector_index, metadata = loader.load_domain(domain)
    
    # Initialize PCR
    pcr = PathConstrainedRetrieval(graph, vector_index, embedder)
    
    # Load queries
    queries_data = loader.load_queries()
    queries = queries_data.get(domain, [])[:10]  # Use subset for benchmarking
    
    if len(queries) == 0:
        return None
    
    benchmark = PerformanceBenchmark(pcr)
    
    print("Benchmarking retrieval latency...")
    retrieval_bench = benchmark.benchmark_retrieval(queries, k=10, num_runs=3)
    retrieval_bench.to_csv(output_dir / f"{domain}_benchmark_retrieval.csv", index=False)
    
    print("Benchmarking reachability computation...")
    anchors = [q.get('anchor') for q in queries if q.get('anchor')]
    if anchors:
        reachability_bench = benchmark.benchmark_reachability(anchors[:5], num_runs=5)
        reachability_bench.to_csv(output_dir / f"{domain}_benchmark_reachability.csv", index=False)
    
    print(f"\nRetrieval Latency (ms):")
    print(retrieval_bench[['query_id', 'mean_latency_ms', 'std_latency_ms']])
    
    return retrieval_bench

def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("Path-Constrained Retrieval: Comprehensive Evaluation")
    print("="*60)
    
    # Setup
    data_dir = Path("data")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize embedder
    print("\nInitializing embedder...")
    embedder = Embedder(cache_dir=Path("cache"))
    
    # Domains to evaluate
    domains = ['tech', 'legal', 'bio', 'microservices', 'citations', 'medical']
    
    # Collect all results
    all_domain_results = []
    all_aggregated = []
    all_statistics = []
    
    # Evaluate each domain
    for domain in domains:
        try:
            results_df, aggregated, stats = evaluate_domain(domain, data_dir, embedder, output_dir)
            if results_df is not None:
                all_domain_results.append(results_df)
                all_aggregated.append(aggregated)
                all_statistics.append(stats)
        except Exception as e:
            print(f"Error evaluating {domain}: {e}")
            import traceback
            traceback.print_exception(*sys.exc_info())
    
    # Combine all domain results
    if all_domain_results:
        combined_results = pd.concat(all_domain_results, ignore_index=True)
        combined_results.to_csv(output_dir / "all_domains_results.csv", index=False)
        
        # Overall aggregated results
        overall_aggregated = Evaluator.aggregate_results_static(combined_results)
        overall_aggregated.to_csv(output_dir / "overall_aggregated.csv")
        
        print("\n" + "="*60)
        print("OVERALL RESULTS (All Domains)")
        print("="*60)
        print(overall_aggregated)
        
        # Overall statistical comparison
        overall_stats = compare_methods_statistically(combined_results, metric='relevance@10')
        overall_stats.to_csv(output_dir / "overall_statistics.csv", index=False)
        print("\nOverall Statistical Comparison (relevance@10):")
        print(overall_stats)
    
    # Run ablation studies on tech domain
    print("\n" + "="*60)
    print("ABLATION STUDIES")
    print("="*60)
    try:
        run_ablation_studies('tech', data_dir, embedder, output_dir)
    except Exception as e:
        print(f"Error in ablation studies: {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())
    
    # Run benchmarks on tech domain
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    try:
        run_benchmarks('tech', data_dir, embedder, output_dir)
    except Exception as e:
        print(f"Error in benchmarks: {e}")
        import traceback
        traceback.print_exception(*sys.exc_info())
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

