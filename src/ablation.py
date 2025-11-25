"""
Ablation studies for Path-Constrained Retrieval.
"""

from typing import List, Dict, Optional
import pandas as pd
from .path_retrieval import PathConstrainedRetrieval
from .evaluation import Evaluator


class AblationStudy:
    """
    Conduct ablation studies on PCR components.
    """
    
    def __init__(
        self,
        pcr_system: PathConstrainedRetrieval,
        evaluator: Evaluator,
        queries: List[Dict]
    ):
        """
        Initialize ablation study.
        
        Args:
            pcr_system: PCR system instance
            evaluator: Evaluator instance
            queries: List of query dictionaries
        """
        self.pcr_system = pcr_system
        self.evaluator = evaluator
        self.queries = queries
    
    def study_max_depth(
        self,
        depth_values: List[Optional[int]] = [None, 1, 2, 3, 5, 10],
        k: int = 10
    ) -> pd.DataFrame:
        """
        Study effect of max_depth parameter.
        
        Args:
            depth_values: List of max_depth values to test
            k: Number of results to retrieve
            
        Returns:
            DataFrame with results for each depth value
        """
        results = []
        
        for max_depth in depth_values:
            for query_data in self.queries:
                query_id = query_data.get('id', '')
                anchor = query_data.get('anchor', '')
                query_text = query_data.get('query', '')
                
                # Retrieve with this max_depth
                pcr_results = self.pcr_system.retrieve(
                    anchor, query_text, k=k, max_depth=max_depth
                )
                retrieved = [node_id for node_id, _, _ in pcr_results]
                
                # Evaluate
                metrics = self.evaluator.evaluate_query(
                    query_id, anchor, retrieved, k_values=[1, 5, 10], max_depth=max_depth
                )
                metrics['max_depth'] = max_depth if max_depth is not None else 'unlimited'
                metrics['query_id'] = query_id
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def study_hybrid_search(
        self,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Study effect of hybrid search mode.
        
        Args:
            k: Number of results to retrieve
            
        Returns:
            DataFrame comparing vector-only vs hybrid
        """
        results = []
        
        for hybrid in [False, True]:
            for query_data in self.queries:
                query_id = query_data.get('id', '')
                anchor = query_data.get('anchor', '')
                query_text = query_data.get('query', '')
                
                # Retrieve with/without hybrid
                pcr_results = self.pcr_system.retrieve(
                    anchor, query_text, k=k, hybrid=hybrid
                )
                retrieved = [node_id for node_id, _, _ in pcr_results]
                
                # Evaluate
                metrics = self.evaluator.evaluate_query(
                    query_id, anchor, retrieved, k_values=[1, 5, 10]
                )
                metrics['hybrid'] = hybrid
                metrics['query_id'] = query_id
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def study_fallback_mode(
        self,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Study effect of fallback mode.
        
        Args:
            k: Number of results to retrieve
            
        Returns:
            DataFrame comparing with/without fallback
        """
        results = []
        
        for fallback in [False, True]:
            for query_data in self.queries:
                query_id = query_data.get('id', '')
                anchor = query_data.get('anchor', '')
                query_text = query_data.get('query', '')
                
                # Retrieve with/without fallback
                try:
                    pcr_results = self.pcr_system.retrieve(
                        anchor, query_text, k=k, fallback=fallback
                    )
                    retrieved = [node_id for node_id, _, _ in pcr_results]
                except:
                    retrieved = []
                
                # Evaluate
                metrics = self.evaluator.evaluate_query(
                    query_id, anchor, retrieved, k_values=[1, 5, 10]
                )
                metrics['fallback'] = fallback
                metrics['query_id'] = query_id
                results.append(metrics)
        
        return pd.DataFrame(results)
    
    def study_all_components(
        self,
        k: int = 10
    ) -> pd.DataFrame:
        """
        Comprehensive ablation study of all components.
        
        Args:
            k: Number of results to retrieve
            
        Returns:
            DataFrame with all ablation results
        """
        all_results = []
        
        # Test different configurations
        configs = [
            {'max_depth': None, 'hybrid': False, 'fallback': True, 'name': 'baseline_pcr'},
            {'max_depth': 2, 'hybrid': False, 'fallback': True, 'name': 'depth_2'},
            {'max_depth': 5, 'hybrid': False, 'fallback': True, 'name': 'depth_5'},
            {'max_depth': None, 'hybrid': True, 'fallback': True, 'name': 'hybrid'},
            {'max_depth': 2, 'hybrid': True, 'fallback': True, 'name': 'depth_2_hybrid'},
        ]
        
        for config in configs:
            for query_data in self.queries:
                query_id = query_data.get('id', '')
                anchor = query_data.get('anchor', '')
                query_text = query_data.get('query', '')
                
                try:
                    pcr_results = self.pcr_system.retrieve(
                        anchor, query_text, k=k,
                        max_depth=config['max_depth'],
                        hybrid=config['hybrid'],
                        fallback=config['fallback']
                    )
                    retrieved = [node_id for node_id, _, _ in pcr_results]
                except:
                    retrieved = []
                
                # Evaluate
                metrics = self.evaluator.evaluate_query(
                    query_id, anchor, retrieved, k_values=[1, 5, 10],
                    max_depth=config['max_depth']
                )
                metrics['config'] = config['name']
                metrics['query_id'] = query_id
                all_results.append(metrics)
        
        return pd.DataFrame(all_results)

