# Quick Start Guide

## Installation

```bash
# Clone repository
git clone https://github.com/Godskid89/path-constrained-retrieval.git
cd path-constrained-retrieval

# Install dependencies
pip install -r requirements.txt

# Set up API key
export OPENAI_API_KEY="your-key-here"
# Or create .env file with: OPENAI_API_KEY=your-key-here
```

## Basic Usage

```python
from pathlib import Path
from src.embedder import Embedder
from src.dataset_loader import DatasetLoader
from src.path_retrieval import PathConstrainedRetrieval

# Initialize
embedder = Embedder(cache_dir=Path("cache"))
loader = DatasetLoader(Path("data"), embedder=embedder)

# Load domain
graph, vector_index, metadata = loader.load_domain("tech")

# Create PCR system
pcr = PathConstrainedRetrieval(graph, vector_index, embedder)

# Retrieve
results = pcr.retrieve(
    anchor="tech_node_000",
    query="What are the key principles of cloud computing?",
    k=5
)

for node_id, score, metadata in results:
    print(f"{node_id}: {score:.3f}")
```

## Run Evaluation

```bash
# Full evaluation pipeline
python scripts/evaluate_all.py

# Generate report
python scripts/generate_results_report.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Documentation

- **README.md**: Full documentation
- **notebooks/demo.ipynb**: Interactive walkthrough
- **paper.tex**: Research paper

