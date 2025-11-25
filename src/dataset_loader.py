"""
Dataset loader for PathRAG-6 benchmark.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from .graph_builder import GraphBuilder
from .vector_index import VectorIndex
from .embedder import Embedder


class DatasetLoader:
    """
    Load PathRAG-6 benchmark datasets.
    """
    
    def __init__(
        self,
        data_dir: Path,
        embedder: Optional[Embedder] = None,
        auto_embed: bool = True
    ):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Root directory containing pathrag6/ folder
            embedder: Optional embedder instance
            auto_embed: Whether to auto-generate embeddings if missing
        """
        self.data_dir = Path(data_dir)
        self.pathrag_dir = self.data_dir / "pathrag6"
        self.embedder = embedder
        self.auto_embed = auto_embed
    
    def load_domain(
        self,
        domain: str
    ) -> Tuple[GraphBuilder, VectorIndex, Dict]:
        """
        Load a specific domain from PathRAG-6.
        
        Args:
            domain: Domain name (tech, legal, bio, microservices, citations, medical)
            
        Returns:
            Tuple of (graph, vector_index, metadata)
        """
        domain_dir = self.pathrag_dir / domain
        
        if not domain_dir.exists():
            raise ValueError(f"Domain directory not found: {domain_dir}")
        
        # Load nodes
        nodes_file = domain_dir / "nodes.json"
        edges_file = domain_dir / "edges.json"
        
        if not nodes_file.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
        
        with open(nodes_file, 'r') as f:
            nodes_data = json.load(f)
        
        # Load edges if available
        edges_data = []
        if edges_file.exists():
            with open(edges_file, 'r') as f:
                edges_data = json.load(f)
        
        # Build graph
        graph = GraphBuilder()
        vector_index = None
        embedding_dim = None
        
        # Process nodes
        for node in nodes_data:
            node_id = node.get('id', str(node.get('node_id', '')))
            text = node.get('text', node.get('content', ''))
            embedding = node.get('embedding')
            metadata = node.get('metadata', {})
            
            # Convert embedding if it's a list
            if embedding and isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            # Auto-embed if needed
            if embedding is None and self.auto_embed and self.embedder:
                embedding = self.embedder.embed(text)
            
            if embedding is not None:
                embedding_dim = len(embedding)
                if vector_index is None:
                    vector_index = VectorIndex(embedding_dim)
                vector_index.add_vector(node_id, embedding)
            
            graph.add_node(node_id, text, embedding, metadata)
        
        # Process edges
        for edge in edges_data:
            source = edge.get('source', edge.get('from'))
            target = edge.get('target', edge.get('to'))
            weight = edge.get('weight', 1.0)
            
            if source and target:
                try:
                    graph.add_edge(source, target, weight)
                except ValueError:
                    # Skip if nodes don't exist
                    pass
        
        metadata = {
            'domain': domain,
            'num_nodes': graph.graph.number_of_nodes(),
            'num_edges': graph.graph.number_of_edges(),
            'embedding_dim': embedding_dim
        }
        
        return graph, vector_index, metadata
    
    def load_all_domains(self) -> Dict[str, Tuple[GraphBuilder, VectorIndex, Dict]]:
        """
        Load all domains from PathRAG-6.
        
        Returns:
            Dictionary mapping domain names to (graph, vector_index, metadata) tuples
        """
        domains = ['tech', 'legal', 'bio', 'microservices', 'citations', 'medical']
        results = {}
        
        for domain in domains:
            try:
                graph, index, metadata = self.load_domain(domain)
                results[domain] = (graph, index, metadata)
            except Exception as e:
                print(f"Warning: Could not load domain {domain}: {e}")
        
        return results
    
    def load_queries(self) -> Dict[str, List[Dict]]:
        """
        Load sample queries for evaluation.
        
        Returns:
            Dictionary mapping domain names to lists of query dictionaries
        """
        queries_file = self.data_dir / "sample_queries.json"
        
        if not queries_file.exists():
            return {}
        
        with open(queries_file, 'r') as f:
            queries_data = json.load(f)
        
        return queries_data
    
    @staticmethod
    def generate_synthetic_domain(
        domain: str,
        num_nodes: int = 30,
        num_edges_per_node: float = 2.0,
        output_dir: Path = None
    ) -> None:
        """
        Generate synthetic data for a domain.
        
        Args:
            domain: Domain name
            num_nodes: Number of nodes to generate
            num_edges_per_node: Average number of edges per node
            output_dir: Output directory (defaults to data/pathrag6/{domain})
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "pathrag6" / domain
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Domain-specific text templates
        templates = {
            'tech': [
                "Introduction to {topic}. This technology enables {capability}.",
                "Advanced concepts in {topic}. Key principles include {principle}.",
                "Implementation details for {topic}. Best practices involve {practice}.",
                "Architecture patterns for {topic}. Common approaches use {approach}.",
                "Performance optimization in {topic}. Critical factors are {factor}."
            ],
            'legal': [
                "Legal framework for {topic}. Key regulations include {regulation}.",
                "Case law analysis on {topic}. Precedents establish {precedent}.",
                "Compliance requirements for {topic}. Organizations must {requirement}.",
                "Contractual obligations regarding {topic}. Parties agree to {obligation}.",
                "Jurisdictional considerations for {topic}. Courts have ruled {ruling}."
            ],
            'bio': [
                "Biological mechanisms of {topic}. The process involves {process}.",
                "Research findings on {topic}. Studies show {finding}.",
                "Clinical applications of {topic}. Treatment protocols include {protocol}.",
                "Genetic factors in {topic}. Variations affect {effect}.",
                "Ecosystem interactions with {topic}. Species relationships show {relationship}."
            ],
            'microservices': [
                "Microservices architecture for {topic}. Services communicate via {method}.",
                "Service decomposition strategy for {topic}. Boundaries are defined by {boundary}.",
                "Inter-service communication in {topic}. Protocols include {protocol}.",
                "Deployment patterns for {topic}. Infrastructure uses {infrastructure}.",
                "Monitoring and observability for {topic}. Metrics track {metric}."
            ],
            'citations': [
                "Research paper on {topic}. Authors propose {proposal}.",
                "Citation network for {topic}. Key references include {reference}.",
                "Literature review of {topic}. Findings indicate {finding}.",
                "Methodology for {topic}. Approach uses {approach}.",
                "Experimental results for {topic}. Data shows {result}."
            ],
            'medical': [
                "Medical diagnosis of {topic}. Symptoms include {symptom}.",
                "Treatment protocols for {topic}. Therapies involve {therapy}.",
                "Drug interactions with {topic}. Medications affect {effect}.",
                "Patient care guidelines for {topic}. Procedures require {procedure}.",
                "Research evidence on {topic}. Studies demonstrate {evidence}."
            ]
        }
        
        topic_words = {
            'tech': ['cloud computing', 'machine learning', 'distributed systems', 'databases', 'networking'],
            'legal': ['intellectual property', 'contract law', 'employment law', 'taxation', 'corporate governance'],
            'bio': ['cell biology', 'genetics', 'ecology', 'biochemistry', 'neuroscience'],
            'microservices': ['API design', 'service mesh', 'container orchestration', 'event-driven architecture', 'distributed tracing'],
            'citations': ['deep learning', 'natural language processing', 'computer vision', 'reinforcement learning', 'graph neural networks'],
            'medical': ['cardiology', 'oncology', 'pediatrics', 'neurology', 'immunology']
        }
        
        import random
        template_list = templates.get(domain, templates['tech'])
        topics = topic_words.get(domain, topic_words['tech'])
        
        # Generate nodes
        nodes = []
        for i in range(num_nodes):
            template = random.choice(template_list)
            topic = random.choice(topics)
            fillers = {
                'capability': 'scalability and reliability',
                'principle': 'modularity and abstraction',
                'practice': 'testing and documentation',
                'approach': 'layered architecture',
                'factor': 'latency and throughput',
                'regulation': 'data protection laws',
                'precedent': 'fair use doctrine',
                'requirement': 'maintain accurate records',
                'obligation': 'deliver services on time',
                'ruling': 'jurisdiction applies',
                'process': 'cellular respiration',
                'finding': 'correlation exists',
                'protocol': 'standardized procedures',
                'effect': 'phenotypic expression',
                'relationship': 'symbiotic interactions',
                'method': 'REST APIs',
                'boundary': 'business capabilities',
                'protocol': 'gRPC and HTTP',
                'infrastructure': 'Kubernetes',
                'metric': 'response times',
                'proposal': 'novel framework',
                'reference': 'seminal works',
                'finding': 'significant improvements',
                'approach': 'experimental validation',
                'result': 'promising outcomes',
                'symptom': 'fever and fatigue',
                'therapy': 'medication and rest',
                'effect': 'metabolic pathways',
                'procedure': 'informed consent',
                'evidence': 'clinical efficacy'
            }
            
            import random
            text = template.format(topic=topic, **{k: random.choice([v]) for k, v in fillers.items() if k in template})
            if '{' in text:  # Fallback if formatting failed
                text = f"Content about {topic} in {domain} domain. This node covers important aspects of the topic."
            
            nodes.append({
                'id': f"{domain}_node_{i:03d}",
                'text': text,
                'metadata': {
                    'domain': domain,
                    'topic': topic,
                    'node_index': i
                }
            })
        
        # Generate edges (create a DAG-like structure)
        import random
        edges = []
        num_edges = int(num_nodes * num_edges_per_node)
        
        for _ in range(num_edges):
            source_idx = random.randint(0, num_nodes - 1)
            target_idx = random.randint(source_idx + 1, num_nodes) if source_idx < num_nodes - 1 else random.randint(0, num_nodes - 1)
            if target_idx >= num_nodes:
                target_idx = num_nodes - 1
            
            source_id = nodes[source_idx]['id']
            target_id = nodes[target_idx]['id']
            
            edges.append({
                'source': source_id,
                'target': target_id,
                'weight': 1.0
            })
        
        # Save nodes
        with open(output_dir / "nodes.json", 'w') as f:
            json.dump(nodes, f, indent=2)
        
        # Save edges
        with open(output_dir / "edges.json", 'w') as f:
            json.dump(edges, f, indent=2)
        
        print(f"Generated synthetic data for {domain}: {num_nodes} nodes, {len(edges)} edges")

