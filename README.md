# Path-Constrained Retrieval

**A Structural Approach to Reliable LLM Agent Reasoning Through Graph-Scoped Semantic Search**

📄 **Paper**: [arXiv:2511.18313](https://arxiv.org/abs/2511.18313) | [PDF](https://arxiv.org/pdf/2511.18313)

## Overview

Path-Constrained Retrieval (PCR) is a novel retrieval method that combines structural graph constraints with semantic search to improve the reliability and consistency of LLM agent reasoning. Unlike traditional vector search that retrieves globally similar content, PCR restricts the search space to nodes reachable from an anchor node in a knowledge graph, ensuring retrieved information maintains logical and structural relationships.

## Key Features

- **Graph-Scoped Search**: Restricts retrieval to structurally connected nodes
- **Semantic Similarity**: Uses state-of-the-art embeddings for relevance
- **Multi-hop Reasoning**: Supports reasoning across multiple graph hops
- **Structural Consistency**: Ensures all retrieved information is structurally reachable from the anchor
- **Hybrid Search**: Combines vector and keyword matching for improved recall

## System Architecture

```
┌─────────────────┐
│   User Query    │
│   + Anchor      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Path Constraint │
│  (Reachability)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Candidate Set  │◄─────┤  Knowledge   │
│  (Reachable)    │      │    Graph     │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Vector Search  │◄─────┤  Embeddings  │
│  (Semantic)     │      │    Index     │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  Ranked Results │
│  (Top-K)        │
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (for embeddings)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Godskid89/path-constrained-retrieval.git
cd path-constrained-retrieval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

4. Install the package (optional):
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from src.embedder import Embedder
from src.graph_builder import GraphBuilder
from src.vector_index import VectorIndex
from src.path_retrieval import PathConstrainedRetrieval
from pathlib import Path

# Initialize components
embedder = Embedder(cache_dir=Path("cache"))
graph = GraphBuilder()
vector_index = VectorIndex(embedding_dim=1536)

# Build graph
graph.add_node("node1", "Text about machine learning", 
               embedding=embedder.embed("Text about machine learning"))
graph.add_node("node2", "Text about neural networks",
               embedding=embedder.embed("Text about neural networks"))
graph.add_edge("node1", "node2")

# Add to vector index
vector_index.add_vector("node1", embedder.embed("Text about machine learning"))
vector_index.add_vector("node2", embedder.embed("Text about neural networks"))

# Initialize PCR system
pcr = PathConstrainedRetrieval(graph, vector_index, embedder)

# Retrieve
results = pcr.retrieve(
    anchor="node1",
    query="What are neural networks?",
    k=5
)

for node_id, score, metadata in results:
    print(f"Node: {node_id}, Score: {score:.3f}")
    print(f"Text: {metadata['text'][:100]}...")
```

### Loading Benchmark Data

```python
from src.dataset_loader import DatasetLoader
from pathlib import Path

# Load a domain
loader = DatasetLoader(Path("data"), embedder=embedder)
graph, vector_index, metadata = loader.load_domain("tech")

print(f"Loaded {metadata['num_nodes']} nodes, {metadata['num_edges']} edges")
```

### Running Evaluation

```python
from src.evaluation import Evaluator
import json

# Load queries
with open("data/sample_queries.json") as f:
    queries_data = json.load(f)

# Initialize evaluator
evaluator = Evaluator(graph, ground_truth={
    "tech_001": ["tech_node_001", "tech_node_005"]
})

# Evaluate
results = evaluator.evaluate_domain(
    queries=queries_data["tech"],
    pcr_system=pcr,
    k=10
)

print(results)
```

## Benchmark: PathRAG-6

The repository includes the PathRAG-6 benchmark with six domains:

- **Tech**: Technology and software engineering
- **Legal**: Legal frameworks and regulations
- **Bio**: Biological and life sciences
- **Microservices**: Microservices architecture
- **Citations**: Academic citations and research
- **Medical**: Medical diagnosis and treatment

Each domain contains:
- Nodes with text content and embeddings
- Directed edges representing relationships
- Sample queries with ground truth

### Running Comprehensive Evaluation

```bash
# Run full evaluation pipeline
python scripts/evaluate_all.py

# Generate results report
python scripts/generate_results_report.py

# Run case studies
python scripts/case_studies.py
```

Or use the evaluation script:
```bash
bash scripts/run_evaluation.sh
```

## Evaluation Metrics

The system implements several evaluation metrics:

1. **Relevance@K**: Fraction of relevant nodes in top-K results
2. **Structural Inconsistency Score**: Fraction of retrieved nodes that are unreachable from anchor
3. **Multi-hop Consistency**: Consistency of path lengths in retrieved nodes
4. **Graph Distance Penalty**: Penalty for retrieving distant nodes

## Project Structure

```
path-constrained-retrieval/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── scripts/                   # Utility scripts
│   ├── evaluate_all.py        # Comprehensive evaluation
│   ├── generate_data.py        # Data generation
│   ├── generate_results_report.py  # Results reporting
│   ├── case_studies.py         # Case study examples
│   └── run_evaluation.sh       # Evaluation script
│
├── data/
│   ├── pathrag6/            # PathRAG-6 benchmark data
│   │   ├── tech/
│   │   ├── legal/
│   │   ├── bio/
│   │   ├── microservices/
│   │   ├── citations/
│   │   └── medical/
│   └── sample_queries.json  # Sample queries
│
├── src/
│   ├── __init__.py
│   ├── embedder.py          # Text embedding with caching
│   ├── graph_builder.py      # Graph construction and operations
│   ├── vector_index.py       # Vector similarity search
│   ├── path_retrieval.py     # Core PCR algorithm
│   ├── dataset_loader.py     # Benchmark data loading
│   ├── evaluation.py         # Evaluation metrics
│   ├── agent_simulator.py    # LLM agent simulation
│   └── utils.py              # Utility functions
│
├── notebooks/
│   └── demo.ipynb            # Interactive demo
│
└── tests/
    ├── test_graph.py
    ├── test_vector_index.py
    ├── test_path_retrieval.py
    └── test_eval.py
```

## Core Components

### Embedder (`embedder.py`)
- OpenAI embeddings with caching
- Batch processing for efficiency
- Automatic cache management

### GraphBuilder (`graph_builder.py`)
- NetworkX-based graph construction
- Node metadata and embeddings
- Reachability and path queries
- Graph serialization

### VectorIndex (`vector_index.py`)
- FAISS or sklearn-based similarity search
- Efficient nearest neighbor search
- Candidate set filtering

### PathConstrainedRetrieval (`path_retrieval.py`)
- Core PCR algorithm
- Path-constrained candidate filtering
- Hybrid search support
- Path information extraction

## Advanced Features

### Hybrid Search

Combine vector and keyword search:

```python
results = pcr.retrieve(
    anchor="node1",
    query="machine learning algorithms",
    k=10,
    hybrid=True
)
```

### Path-Aware Retrieval

Include path information in results:

```python
results = pcr.retrieve_with_paths(
    anchor="node1",
    query="neural networks",
    k=5,
    include_paths=True
)

for node_id, score, metadata in results:
    print(f"Paths from anchor: {metadata['paths']}")
```

### Custom LLM Integration

```python
from src.agent_simulator import AgentSimulator

def my_llm(context: str, query: str) -> str:
    # Your LLM call here
    return llm_client.generate(context, query)

simulator = AgentSimulator(graph, vector_index, embedder, llm_function=my_llm)
result = simulator.simulate_with_pcr("node1", "What is machine learning?")
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Or run individual test files:

```bash
pytest tests/test_graph.py
pytest tests/test_path_retrieval.py
```

## Performance Considerations

- **Caching**: Embeddings are cached to reduce API costs
- **Batch Processing**: Embeddings are processed in batches
- **FAISS**: Uses FAISS for fast similarity search when available
- **Graph Traversal**: Efficient BFS for reachability queries

## Citation

If you use this code in your research, please cite:

```bibtex
@article{oladokun2025path,
  title={Path-Constrained Retrieval: A Structural Approach to Reliable LLM Agent Reasoning Through Graph-Scoped Semantic Search},
  author={Oladokun, Joseph},
  journal={arXiv preprint arXiv:2511.18313},
  year={2025},
  url={https://arxiv.org/abs/2511.18313}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open a GitHub issue.

