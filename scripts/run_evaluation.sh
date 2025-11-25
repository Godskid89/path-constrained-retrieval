#!/bin/bash
# Comprehensive evaluation script for ArXiv paper

echo "=========================================="
echo "Path-Constrained Retrieval Evaluation"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run comprehensive evaluation
echo "Running comprehensive evaluation..."
python3 evaluate_all.py

# Generate report
echo "Generating results report..."
python3 generate_results_report.py

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved in: results/"
echo "=========================================="

