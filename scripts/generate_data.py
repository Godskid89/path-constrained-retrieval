#!/usr/bin/env python3
"""
Script to generate synthetic benchmark data for PathRAG-6.
"""

import json
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

def generate_synthetic_domain(domain: str, num_nodes: int = 30, num_edges_per_node: float = 2.0):
    """Generate synthetic data for a domain."""
    output_dir = Path(__file__).parent / "data" / "pathrag6" / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    
    templates = {
        'tech': [
            "Introduction to {topic}. This technology enables scalability and reliability.",
            "Advanced concepts in {topic}. Key principles include modularity and abstraction.",
            "Implementation details for {topic}. Best practices involve testing and documentation.",
            "Architecture patterns for {topic}. Common approaches use layered architecture.",
            "Performance optimization in {topic}. Critical factors are latency and throughput."
        ],
        'legal': [
            "Legal framework for {topic}. Key regulations include data protection laws.",
            "Case law analysis on {topic}. Precedents establish fair use doctrine.",
            "Compliance requirements for {topic}. Organizations must maintain accurate records.",
            "Contractual obligations regarding {topic}. Parties agree to deliver services on time.",
            "Jurisdictional considerations for {topic}. Courts have ruled jurisdiction applies."
        ],
        'bio': [
            "Biological mechanisms of {topic}. The process involves cellular respiration.",
            "Research findings on {topic}. Studies show correlation exists.",
            "Clinical applications of {topic}. Treatment protocols include standardized procedures.",
            "Genetic factors in {topic}. Variations affect phenotypic expression.",
            "Ecosystem interactions with {topic}. Species relationships show symbiotic interactions."
        ],
        'microservices': [
            "Microservices architecture for {topic}. Services communicate via REST APIs.",
            "Service decomposition strategy for {topic}. Boundaries are defined by business capabilities.",
            "Inter-service communication in {topic}. Protocols include gRPC and HTTP.",
            "Deployment patterns for {topic}. Infrastructure uses Kubernetes.",
            "Monitoring and observability for {topic}. Metrics track response times."
        ],
        'citations': [
            "Research paper on {topic}. Authors propose novel framework.",
            "Citation network for {topic}. Key references include seminal works.",
            "Literature review of {topic}. Findings indicate significant improvements.",
            "Methodology for {topic}. Approach uses experimental validation.",
            "Experimental results for {topic}. Data shows promising outcomes."
        ],
        'medical': [
            "Medical diagnosis of {topic}. Symptoms include fever and fatigue.",
            "Treatment protocols for {topic}. Therapies involve medication and rest.",
            "Drug interactions with {topic}. Medications affect metabolic pathways.",
            "Patient care guidelines for {topic}. Procedures require informed consent.",
            "Research evidence on {topic}. Studies demonstrate clinical efficacy."
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
    
    template_list = templates.get(domain, templates['tech'])
    topics = topic_words.get(domain, topic_words['tech'])
    
    # Generate nodes
    nodes = []
    for i in range(num_nodes):
        template = random.choice(template_list)
        topic = random.choice(topics)
        text = template.format(topic=topic)
        if '{' in text:  # Fallback
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

if __name__ == "__main__":
    domains = ['tech', 'legal', 'bio', 'microservices', 'citations', 'medical']
    for domain in domains:
        generate_synthetic_domain(domain, num_nodes=30, num_edges_per_node=2.0)

