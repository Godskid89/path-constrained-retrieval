"""
LLM agent simulator for comparing PCR vs baseline retrieval.
"""

from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from .path_retrieval import PathConstrainedRetrieval
from .vector_index import VectorIndex
from .embedder import Embedder
from .graph_builder import GraphBuilder


class AgentSimulator:
    """
    Simulate LLM agent behavior with different retrieval methods.
    """
    
    def __init__(
        self,
        graph: GraphBuilder,
        vector_index: VectorIndex,
        embedder: Embedder,
        llm_function: Optional[Callable] = None
    ):
        """
        Initialize agent simulator.
        
        Args:
            graph: GraphBuilder instance
            vector_index: VectorIndex instance
            embedder: Embedder instance
            llm_function: Optional LLM call function (defaults to placeholder)
        """
        self.graph = graph
        self.vector_index = vector_index
        self.embedder = embedder
        self.pcr_system = PathConstrainedRetrieval(graph, vector_index, embedder)
        self.llm_function = llm_function or self._default_llm
    
    def _default_llm(self, context: str, query: str) -> str:
        """
        Placeholder LLM function.
        
        Args:
            context: Retrieved context
            query: User query
            
        Returns:
            Generated response
        """
        # Simple template-based response (replace with actual LLM call)
        return f"Based on the context: {context[:200]}... The answer to '{query}' is derived from the retrieved information."
    
    def simulate_with_pcr(
        self,
        anchor: str,
        query: str,
        k: int = 5,
        max_depth: Optional[int] = None
    ) -> Dict:
        """
        Simulate agent using PCR retrieval.
        
        Args:
            anchor: Anchor node ID
            query: User query
            k: Number of retrieved nodes
            max_depth: Maximum depth for reachability
            
        Returns:
            Dictionary with retrieval results and agent response
        """
        # Retrieve context using PCR
        retrieved = self.pcr_system.retrieve(anchor, query, k=k, max_depth=max_depth)
        
        # Build context from retrieved nodes
        context_parts = []
        for node_id, score, metadata in retrieved:
            text = metadata.get('text', '')
            context_parts.append(f"[Node {node_id}]: {text}")
        
        context = "\n\n".join(context_parts)
        
        # Call LLM
        response = self.llm_function(context, query)
        
        return {
            'method': 'PCR',
            'anchor': anchor,
            'query': query,
            'retrieved_nodes': [node_id for node_id, _, _ in retrieved],
            'context': context,
            'response': response,
            'num_retrieved': len(retrieved)
        }
    
    def simulate_with_baseline(
        self,
        query: str,
        k: int = 5
    ) -> Dict:
        """
        Simulate agent using baseline vector search (no path constraints).
        
        Args:
            query: User query
            k: Number of retrieved nodes
            
        Returns:
            Dictionary with retrieval results and agent response
        """
        # Embed query
        query_emb = self.embedder.embed(query)
        
        # Baseline retrieval (no path constraints)
        retrieved = self.vector_index.search(query_emb, candidates=None, k=k)
        
        # Build context
        context_parts = []
        for node_id, score in retrieved:
            node_data = self.graph.get_node(node_id)
            text = node_data.get('text', '')
            context_parts.append(f"[Node {node_id}]: {text}")
        
        context = "\n\n".join(context_parts)
        
        # Call LLM
        response = self.llm_function(context, query)
        
        return {
            'method': 'Baseline',
            'query': query,
            'retrieved_nodes': [node_id for node_id, _ in retrieved],
            'context': context,
            'response': response,
            'num_retrieved': len(retrieved)
        }
    
    def compare_methods(
        self,
        anchor: str,
        query: str,
        ground_truth: Optional[str] = None,
        k: int = 5,
        max_depth: Optional[int] = None
    ) -> Dict:
        """
        Compare PCR vs baseline retrieval.
        
        Args:
            anchor: Anchor node ID
            query: User query
            ground_truth: Optional ground truth answer
            k: Number of retrieved nodes
            max_depth: Maximum depth for PCR
            
        Returns:
            Dictionary with comparison results
        """
        pcr_result = self.simulate_with_pcr(anchor, query, k, max_depth)
        baseline_result = self.simulate_with_baseline(query, k)
        
        comparison = {
            'query': query,
            'anchor': anchor,
            'ground_truth': ground_truth,
            'pcr': pcr_result,
            'baseline': baseline_result,
            'metrics': {
                'pcr_num_retrieved': pcr_result['num_retrieved'],
                'baseline_num_retrieved': baseline_result['num_retrieved'],
                'pcr_nodes': set(pcr_result['retrieved_nodes']),
                'baseline_nodes': set(baseline_result['retrieved_nodes']),
                'overlap': len(set(pcr_result['retrieved_nodes']) & set(baseline_result['retrieved_nodes']))
            }
        }
        
        return comparison
    
    def evaluate_on_queries(
        self,
        queries: List[Dict],
        ground_truth: Optional[Dict[str, str]] = None
    ) -> List[Dict]:
        """
        Evaluate on multiple queries.
        
        Args:
            queries: List of query dictionaries with 'anchor' and 'query' fields
            ground_truth: Optional mapping of query_id -> ground truth answer
            
        Returns:
            List of comparison results
        """
        results = []
        
        for query_data in queries:
            anchor = query_data.get('anchor', '')
            query = query_data.get('query', '')
            query_id = query_data.get('id', '')
            
            gt = ground_truth.get(query_id) if ground_truth else None
            
            comparison = self.compare_methods(anchor, query, gt)
            comparison['query_id'] = query_id
            results.append(comparison)
        
        return results

