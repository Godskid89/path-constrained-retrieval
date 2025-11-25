#!/usr/bin/env python3
"""
Generate comprehensive results report for ArXiv paper.
"""

import pandas as pd
from pathlib import Path
import json

def generate_report(results_dir: Path = Path("results")):
    """Generate markdown report from results."""
    
    report = []
    report.append("# Path-Constrained Retrieval: Experimental Results\n")
    report.append("This report contains comprehensive evaluation results for the Path-Constrained Retrieval system.\n")
    
    # Overall results
    overall_file = results_dir / "overall_aggregated.csv"
    if overall_file.exists():
        report.append("## Overall Results (All Domains)\n")
        df = pd.read_csv(overall_file, index_col=[0, 1])
        report.append(df.to_markdown())
        report.append("\n")
    
    # Statistical significance
    stats_file = results_dir / "overall_statistics.csv"
    if stats_file.exists():
        report.append("## Statistical Significance Tests\n")
        df = pd.read_csv(stats_file)
        report.append(df.to_markdown())
        report.append("\n")
    
    # Per-domain results
    for domain in ['tech', 'legal', 'bio', 'microservices', 'citations', 'medical']:
        domain_file = results_dir / f"{domain}_aggregated.csv"
        if domain_file.exists():
            report.append(f"## {domain.upper()} Domain Results\n")
            df = pd.read_csv(domain_file, index_col=[0, 1])
            report.append(df.to_markdown())
            report.append("\n")
    
    # Ablation studies
    ablation_file = results_dir / "tech_ablation_all.csv"
    if ablation_file.exists():
        report.append("## Ablation Studies\n")
        df = pd.read_csv(ablation_file)
        # Summarize ablation results
        summary = df.groupby('config')[['relevance@10', 'hallucination']].mean()
        report.append(summary.to_markdown())
        report.append("\n")
    
    # Performance benchmarks
    bench_file = results_dir / "tech_benchmark_retrieval.csv"
    if bench_file.exists():
        report.append("## Performance Benchmarks\n")
        df = pd.read_csv(bench_file)
        report.append(f"Average retrieval latency: {df['mean_latency_ms'].mean():.2f} ms (±{df['std_latency_ms'].mean():.2f})\n")
        report.append("\n")
    
    # Save report
    report_text = "\n".join(report)
    with open(results_dir / "RESULTS_REPORT.md", 'w') as f:
        f.write(report_text)
    
    print("Report generated: results/RESULTS_REPORT.md")
    return report_text

if __name__ == "__main__":
    generate_report()

