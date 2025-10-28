"""
Pareto frontier analysis and dominance comparisons.

Analyzes the trade-off between latency (minimize) and accuracy (maximize) across
all configurations and methods.

Metrics:
1. Hypervolume: Area dominated by the Pareto frontier (larger is better)
2. Dominance ratio: Fraction of baseline points dominated by planner

Reference point for hypervolume:
- latency_ref = worst (highest) latency across all configs
- accuracy_ref = worst (lowest) accuracy across all configs

Output: results/pareto_analysis.csv
Schema: method,num_configs,hypervolume,dominance_ratio,num_pareto_points
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.pareto_analysis")


def load_all_results():
    """Load all evaluation results."""
    LOGGER.info("Loading all evaluation results...")
    
    baseline_path = Path("results/baseline_results.csv")
    planner_path = Path("results/planner_results.csv")
    
    if not baseline_path.exists():
        raise FileNotFoundError("Baseline results not found. Run 03_run_baselines.py first.")
    if not planner_path.exists():
        raise FileNotFoundError("Planner results not found. Run 04_run_planner.py first.")
    
    baseline_df = pd.read_csv(baseline_path)
    planner_df = pd.read_csv(planner_path)
    
    # Combine all results
    all_results = pd.concat([baseline_df, planner_df], ignore_index=True)
    
    LOGGER.info(f"Loaded {len(all_results)} total results")
    
    return all_results, baseline_df, planner_df


def is_dominated(point, other_points):
    """
    Check if a point is dominated by any other point.
    
    For latency (minimize) and accuracy (maximize):
    Point A dominates point B if:
    - A.latency <= B.latency AND A.accuracy >= B.accuracy
    - At least one inequality is strict
    """
    lat, acc = point
    
    for other_lat, other_acc in other_points:
        # Check if other point dominates this point
        if other_lat <= lat and other_acc >= acc:
            if other_lat < lat or other_acc > acc:
                return True
    
    return False


def compute_pareto_frontier(df):
    """Compute Pareto frontier points."""
    points = df[['lat_p95_ms', 'accuracy']].values
    
    pareto_mask = []
    for i, point in enumerate(points):
        # Check if this point is dominated by any other point
        other_points = np.delete(points, i, axis=0)
        dominated = is_dominated(point, other_points)
        pareto_mask.append(not dominated)
    
    pareto_df = df[pareto_mask].copy()
    
    LOGGER.info(f"  Pareto frontier: {len(pareto_df)}/{len(df)} points")
    
    return pareto_df


def compute_hypervolume(pareto_df, ref_latency, ref_accuracy):
    """
    Compute hypervolume indicator.
    
    Reference point: (ref_latency, ref_accuracy) = (worst latency, worst accuracy)
    
    For each Pareto point (lat, acc), the dominated area is:
    (ref_latency - lat) * (acc - ref_accuracy)
    
    Note: This is a simplified 2D hypervolume calculation.
    """
    if len(pareto_df) == 0:
        return 0.0
    
    # Sort by latency
    sorted_df = pareto_df.sort_values('lat_p95_ms')
    
    total_volume = 0.0
    prev_lat = ref_latency
    
    for _, row in sorted_df.iterrows():
        lat = row['lat_p95_ms']
        acc = row['accuracy']
        
        # Width: from current latency to previous (or reference)
        width = prev_lat - lat
        # Height: from current accuracy to reference
        height = acc - ref_accuracy
        
        # Area contribution (only if positive)
        if width > 0 and height > 0:
            total_volume += width * height
        
        prev_lat = lat
    
    return total_volume


def compute_dominance_ratio(planner_df, baseline_df):
    """
    Compute fraction of baseline points dominated by planner points.
    """
    planner_points = planner_df[['lat_p95_ms', 'accuracy']].values
    baseline_points = baseline_df[['lat_p95_ms', 'accuracy']].values
    
    dominated_count = 0
    
    for baseline_point in baseline_points:
        if is_dominated(baseline_point, planner_points):
            dominated_count += 1
    
    dominance_ratio = dominated_count / len(baseline_points) if len(baseline_points) > 0 else 0.0
    
    return dominance_ratio


def main():
    LOGGER.info("Starting Pareto frontier analysis...")
    
    # Load results
    all_results, baseline_df, planner_df = load_all_results()
    
    # Compute reference point (worst latency, worst accuracy)
    ref_latency = all_results['lat_p95_ms'].max()
    ref_accuracy = all_results['accuracy'].min()
    
    LOGGER.info(f"Reference point: latency={ref_latency:.2f}ms, accuracy={ref_accuracy:.4f}")
    
    results = []
    
    # Analyze each method
    for method in all_results['method'].unique():
        LOGGER.info(f"\nAnalyzing {method}...")
        
        method_df = all_results[all_results['method'] == method]
        
        # Compute Pareto frontier
        pareto_df = compute_pareto_frontier(method_df)
        
        # Compute hypervolume
        hypervolume = compute_hypervolume(pareto_df, ref_latency, ref_accuracy)
        
        LOGGER.info(f"  Hypervolume: {hypervolume:.2f}")
        
        results.append({
            'method': method,
            'num_configs': len(method_df),
            'hypervolume': hypervolume,
            'num_pareto_points': len(pareto_df)
        })
    
    # Compute dominance ratio for planner vs. baselines
    LOGGER.info("\nComputing dominance ratios...")
    
    baseline_methods = ['StaticSmall', 'StaticLarge', 'ThroughputAutotuner']
    
    for baseline_method in baseline_methods:
        baseline_subset = baseline_df[baseline_df['method'] == baseline_method]
        
        if len(baseline_subset) > 0 and len(planner_df) > 0:
            dominance_ratio = compute_dominance_ratio(planner_df, baseline_subset)
            
            LOGGER.info(f"  CascadePlanner dominates {dominance_ratio:.1%} of {baseline_method} points")
            
            # Update results
            for result in results:
                if result['method'] == 'CascadePlanner':
                    result[f'dominance_vs_{baseline_method}'] = dominance_ratio
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("results/pareto_analysis.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    LOGGER.info(f"\nPareto analysis complete. Results saved to {output_path}")
    
    # Summary
    LOGGER.info("\nSummary:")
    for _, row in df.iterrows():
        LOGGER.info(f"  {row['method']}:")
        LOGGER.info(f"    Hypervolume: {row['hypervolume']:.2f}")
        LOGGER.info(f"    Pareto points: {row['num_pareto_points']}/{row['num_configs']}")


if __name__ == "__main__":
    main()

