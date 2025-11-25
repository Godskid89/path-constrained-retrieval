"""
Evaluation metrics for Path-Constrained Retrieval.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from .graph_builder import GraphBuilder
from .path_retrieval import PathConstrainedRetrieval


class Evaluator:
    """
    Evaluation metrics for PCR system.
    """
    
    def __init__(
        self,
        graph: GraphBuilder,
        ground_truth: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            graph: GraphBuilder instance
            ground_truth: Optional ground truth mapping query_id -> relevant node IDs
        """
        self.graph = graph
        self.ground_truth = ground_truth or {}
    
    def relevance_at_k(
        self,
        retrieved: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Compute relevance@k metric.
        
        Args:
            retrieved: List of retrieved node IDs
            relevant: List of relevant node IDs
            k: Cutoff for top-k
        
        Returns:
            Relevance@k score (0-1)
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        if len(relevant_set) == 0:
            return 0.0
        
        num_relevant = sum(1 for node_id in top_k if node_id in relevant_set)
        return num_relevant / min(k, len(relevant_set))
    
    def hallucination_score(
        self,
        retrieved: List[str],
        anchor: str,
        max_depth: Optional[int] = None
    ) -> float:
        """
        Compute hallucination score (fraction of retrieved nodes that are unreachable).
        
        Args:
            retrieved: List of retrieved node IDs
            anchor: Anchor node ID
            max_depth: Maximum depth for reachability
            
        Returns:
            Hallucination score (0-1, where 0 is no hallucinations)
        """
        if len(retrieved) == 0:
            return 0.0
        
        try:
            reachable = self.graph.reachable(anchor, max_depth)
        except ValueError:
            return 1.0  # All are hallucinations if anchor doesn't exist
        
        unreachable = [node_id for node_id in retrieved if node_id not in reachable]
        return len(unreachable) / len(retrieved)
    
    def multi_hop_consistency(
        self,
        retrieved: List[str],
        anchor: str
    ) -> float:
        """
        Compute multi-hop consistency score based on path lengths.
        
        Args:
            retrieved: List of retrieved node IDs
            anchor: Anchor node ID
            
        Returns:
            Consistency score (0-1, higher is better)
        """
        if len(retrieved) == 0:
            return 0.0
        
        path_lengths = []
        for node_id in retrieved:
            path_len = self.graph.get_shortest_path_length(anchor, node_id)
            if path_len is not None:
                path_lengths.append(path_len)
        
        if len(path_lengths) == 0:
            return 0.0
        
        # Consistency: lower variance in path lengths is better
        # Normalize by average path length
        avg_length = np.mean(path_lengths)
        if avg_length == 0:
            return 1.0
        
        std_length = np.std(path_lengths)
        consistency = 1.0 / (1.0 + std_length / avg_length)
        return consistency
    
    def graph_distance_penalty(
        self,
        retrieved: List[str],
        anchor: str,
        penalty_weight: float = 0.1
    ) -> float:
        """
        Compute graph distance penalty (penalizes distant nodes).
        
        Args:
            retrieved: List of retrieved node IDs
            anchor: Anchor node ID
            penalty_weight: Weight for penalty
            
        Returns:
            Penalty score (0-1, lower is better)
        """
        if len(retrieved) == 0:
            return 0.0
        
        total_penalty = 0.0
        for node_id in retrieved:
            path_len = self.graph.get_shortest_path_length(anchor, node_id)
            if path_len is None:
                # Unreachable nodes get maximum penalty
                total_penalty += 1.0
            else:
                # Penalty increases with distance
                total_penalty += penalty_weight * path_len
        
        return total_penalty / len(retrieved)
    
    def evaluate_query(
        self,
        query_id: str,
        anchor: str,
        retrieved: List[str],
        k_values: List[int] = [1, 5, 10],
        max_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single query.
        
        Args:
            query_id: Query identifier
            anchor: Anchor node ID
            retrieved: List of retrieved node IDs
            k_values: List of k values for relevance@k
            max_depth: Maximum depth for reachability
            
        Returns:
            Dictionary of metric scores
        """
        relevant = self.ground_truth.get(query_id, [])
        
        results = {}
        
        # Relevance metrics
        for k in k_values:
            results[f'relevance@{k}'] = self.relevance_at_k(retrieved, relevant, k)
        
        # Hallucination score
        results['hallucination'] = self.hallucination_score(retrieved, anchor, max_depth)
        
        # Multi-hop consistency
        results['consistency'] = self.multi_hop_consistency(retrieved, anchor)
        
        # Graph distance penalty
        results['distance_penalty'] = self.graph_distance_penalty(retrieved, anchor)
        
        return results
    
    def evaluate_domain(
        self,
        queries: List[Dict],
        pcr_system: PathConstrainedRetrieval,
        baseline_retrieval: Optional[callable] = None,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Evaluate PCR system on a domain.
        
        Args:
            queries: List of query dictionaries with 'id', 'anchor', 'query' fields
            pcr_system: PathConstrainedRetrieval instance
            baseline_retrieval: Optional baseline retrieval function
            k: Number of results to retrieve
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for query_data in queries:
            query_id = query_data.get('id', '')
            anchor = query_data.get('anchor', '')
            query_text = query_data.get('query', '')
            
            # PCR retrieval
            pcr_results = pcr_system.retrieve(anchor, query_text, k=k)
            pcr_retrieved = [node_id for node_id, _, _ in pcr_results]
            
            pcr_metrics = self.evaluate_query(query_id, anchor, pcr_retrieved, k_values=[1, 5, 10])
            pcr_metrics['method'] = 'PCR'
            pcr_metrics['query_id'] = query_id
            results.append(pcr_metrics)
            
            # Baseline retrieval (if provided)
            if baseline_retrieval:
                baseline_results = baseline_retrieval(query_text, k=k)
                baseline_retrieved = [node_id for node_id, _ in baseline_results]
                
                baseline_metrics = self.evaluate_query(query_id, anchor, baseline_retrieved, k_values=[1, 5, 10])
                baseline_metrics['method'] = 'Baseline'
                baseline_metrics['query_id'] = query_id
                results.append(baseline_metrics)
        
        df = pd.DataFrame(results)
        return df
    
    def aggregate_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate evaluation results across queries.
        
        Args:
            results_df: DataFrame with per-query results
            
        Returns:
            Aggregated statistics DataFrame
        """
        # Filter out non-numeric columns
        exclude_cols = ['method', 'query_id', 'domain', 'config', 'max_depth', 'hybrid', 'fallback']
        numeric_cols = [col for col in results_df.columns 
                       if col not in exclude_cols and pd.api.types.is_numeric_dtype(results_df[col])]
        
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        aggregated = results_df.groupby('method')[numeric_cols].agg(['mean', 'std'])
        return aggregated
    
    @staticmethod
    def aggregate_results_static(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Static method to aggregate results (for use without Evaluator instance).
        
        Args:
            results_df: DataFrame with per-query results
            
        Returns:
            Aggregated statistics DataFrame
        """
        if 'method' not in results_df.columns:
            return pd.DataFrame()
        
        exclude_cols = ['method', 'query_id', 'domain', 'config', 'max_depth', 'hybrid', 'fallback']
        numeric_cols = [col for col in results_df.columns 
                       if col not in exclude_cols]
        
        # Filter to only numeric columns
        numeric_cols = [col for col in numeric_cols 
                       if pd.api.types.is_numeric_dtype(results_df[col])]
        
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        aggregated = results_df.groupby('method')[numeric_cols].agg(['mean', 'std'])
        return aggregated

