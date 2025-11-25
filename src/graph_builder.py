"""
Graph builder module using NetworkX for knowledge graph construction.
"""

from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import networkx as nx
import numpy as np
import pickle
import json
from .utils import ensure_dir


class GraphBuilder:
    """
    Build and manage directed graphs with node metadata and embeddings.
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self.graph = nx.DiGraph()
    
    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            text: Text content of the node
            embedding: Optional embedding vector
            metadata: Optional metadata dictionary
        """
        node_data = {
            'text': text,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        self.graph.add_node(node_id, **node_data)
    
    def add_edge(self, parent: str, child: str, weight: float = 1.0) -> None:
        """
        Add a directed edge from parent to child.
        
        Args:
            parent: Source node ID
            child: Target node ID
            weight: Edge weight (default 1.0)
        """
        if parent not in self.graph:
            raise ValueError(f"Parent node {parent} does not exist")
        if child not in self.graph:
            raise ValueError(f"Child node {child} does not exist")
        
        self.graph.add_edge(parent, child, weight=weight)
    
    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get node data.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node data dictionary
        """
        if node_id not in self.graph:
            raise ValueError(f"Node {node_id} does not exist")
        return self.graph.nodes[node_id]
    
    def reachable(
        self,
        anchor: str,
        max_depth: Optional[int] = None
    ) -> Set[str]:
        """
        Get all nodes reachable from an anchor node.
        
        Args:
            anchor: Starting node ID
            max_depth: Maximum depth to traverse (None for unlimited)
            
        Returns:
            Set of reachable node IDs
        """
        if anchor not in self.graph:
            raise ValueError(f"Anchor node {anchor} does not exist")
        
        reachable_nodes = {anchor}
        
        if max_depth is None:
            # Use BFS to find all reachable nodes
            queue = [anchor]
            visited = {anchor}
            
            while queue:
                current = queue.pop(0)
                for neighbor in self.graph.successors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        reachable_nodes.add(neighbor)
                        queue.append(neighbor)
        else:
            # BFS with depth limit
            queue = [(anchor, 0)]
            visited = {anchor}
            
            while queue:
                current, depth = queue.pop(0)
                if depth < max_depth:
                    for neighbor in self.graph.successors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            reachable_nodes.add(neighbor)
                            queue.append((neighbor, depth + 1))
        
        return reachable_nodes
    
    def induced_subgraph(self, anchor: str) -> nx.DiGraph:
        """
        Get the induced subgraph of all nodes reachable from anchor.
        
        Args:
            anchor: Starting node ID
            
        Returns:
            NetworkX DiGraph containing the subgraph
        """
        reachable_nodes = self.reachable(anchor)
        return self.graph.subgraph(reachable_nodes).copy()
    
    def get_shortest_path_length(self, source: str, target: str) -> Optional[int]:
        """
        Get shortest path length between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            Shortest path length, or None if no path exists
        """
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None
    
    def get_all_paths(self, source: str, target: str, max_length: int = 10) -> List[List[str]]:
        """
        Get all simple paths between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            max_length: Maximum path length to consider
            
        Returns:
            List of paths (each path is a list of node IDs)
        """
        try:
            return list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
        except nx.NetworkXNoPath:
            return []
    
    def save_graph(self, path: Path) -> None:
        """
        Save graph to disk.
        
        Args:
            path: File path to save to
        """
        ensure_dir(path.parent)
        
        # Save as pickle for full fidelity
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        # Also save a JSON representation (without embeddings) for inspection
        json_path = path.with_suffix('.json')
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        for node_id, data in self.graph.nodes(data=True):
            node_info = {
                'id': node_id,
                'text': data.get('text', '')[:100],  # Truncate for JSON
                'metadata': data.get('metadata', {})
            }
            graph_data['nodes'].append(node_info)
        
        for source, target, data in self.graph.edges(data=True):
            graph_data['edges'].append({
                'source': source,
                'target': target,
                'weight': data.get('weight', 1.0)
            })
        
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self, path: Path) -> None:
        """
        Load graph from disk.
        
        Args:
            path: File path to load from
        """
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes())
        }

