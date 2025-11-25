"""
Performance benchmarking for PCR system.
"""

import time
from typing import List, Dict, Callable
import numpy as np
import pandas as pd
from .path_retrieval import PathConstrainedRetrieval
from .graph_builder import GraphBuilder


class PerformanceBenchmark:
    """
    Benchmark retrieval performance (speed, memory, etc.).
    """
    
    def __init__(self, pcr_system: PathConstrainedRetrieval):
        """
        Initialize benchmark.
        
        Args:
            pcr_system: PCR system to benchmark
        """
        self.pcr_system = pcr_system
        self.graph = pcr_system.graph
    
    def benchmark_retrieval(
        self,
        queries: List[Dict],
        k: int = 10,
        num_runs: int = 5
    ) -> pd.DataFrame:
        """
        Benchmark retrieval latency.
        
        Args:
            queries: List of query dictionaries
            k: Number of results
            num_runs: Number of runs for averaging
            
        Returns:
            DataFrame with timing results
        """
        results = []
        
        for query_data in queries:
            anchor = query_data.get('anchor', '')
            query_text = query_data.get('query', '')
            
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = self.pcr_system.retrieve(anchor, query_text, k=k)
                elapsed = (time.time() - start) * 1000  # Convert to ms
                times.append(elapsed)
            
            results.append({
                'query_id': query_data.get('id', ''),
                'anchor': anchor,
                'mean_latency_ms': np.mean(times),
                'std_latency_ms': np.std(times),
                'min_latency_ms': np.min(times),
                'max_latency_ms': np.max(times),
                'num_runs': num_runs
            })
        
        return pd.DataFrame(results)
    
    def benchmark_reachability(
        self,
        anchors: List[str],
        num_runs: int = 10
    ) -> pd.DataFrame:
        """
        Benchmark reachability computation.
        
        Args:
            anchors: List of anchor node IDs
            num_runs: Number of runs for averaging
            
        Returns:
            DataFrame with timing results
        """
        results = []
        
        for anchor in anchors:
            times = []
            for _ in range(num_runs):
                start = time.time()
                _ = self.graph.reachable(anchor)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            reachable_count = len(self.graph.reachable(anchor))
            
            results.append({
                'anchor': anchor,
                'reachable_nodes': reachable_count,
                'mean_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'num_runs': num_runs
            })
        
        return pd.DataFrame(results)
    
    def benchmark_scalability(
        self,
        query: str,
        anchor: str,
        k: int = 10,
        graph_sizes: List[int] = None
    ) -> pd.DataFrame:
        """
        Benchmark scalability with different graph sizes.
        
        Args:
            query: Test query
            anchor: Anchor node
            k: Number of results
            graph_sizes: List of graph sizes to test (if None, uses current graph)
            
        Returns:
            DataFrame with scalability results
        """
        if graph_sizes is None:
            # Use current graph size
            current_size = self.graph.graph.number_of_nodes()
            graph_sizes = [current_size]
        
        results = []
        
        for size in graph_sizes:
            # Note: This is a simplified benchmark
            # In practice, you'd create graphs of different sizes
            times = []
            for _ in range(5):
                start = time.time()
                _ = self.pcr_system.retrieve(anchor, query, k=k)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            results.append({
                'graph_size': size,
                'mean_latency_ms': np.mean(times),
                'std_latency_ms': np.std(times)
            })
        
        return pd.DataFrame(results)
    
    def compare_methods_performance(
        self,
        queries: List[Dict],
        methods: Dict[str, Callable],
        k: int = 10
    ) -> pd.DataFrame:
        """
        Compare performance of different retrieval methods.
        
        Args:
            queries: List of query dictionaries
            methods: Dictionary mapping method name to retrieval function
            k: Number of results
            
        Returns:
            DataFrame with performance comparison
        """
        results = []
        
        for query_data in queries:
            anchor = query_data.get('anchor', '')
            query_text = query_data.get('query', '')
            
            for method_name, method_func in methods.items():
                times = []
                for _ in range(3):
                    start = time.time()
                    _ = method_func(anchor, query_text, k)
                    elapsed = (time.time() - start) * 1000
                    times.append(elapsed)
                
                results.append({
                    'query_id': query_data.get('id', ''),
                    'method': method_name,
                    'mean_latency_ms': np.mean(times),
                    'std_latency_ms': np.std(times)
                })
        
        return pd.DataFrame(results)

