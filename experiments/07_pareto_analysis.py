"""Pareto frontier analysis and dominance comparisons.

Analyzes the Pareto frontier of configurations in the latency-accuracy space.
Does NOT claim formal proof of optimality - the planner is greedy over profiled
configurations.

Metrics:
- Hypervolume: Area under Pareto frontier (larger is better)
- Dominance ratio: Fraction of baseline points dominated by planner
- Pareto efficiency: Fraction of planner points on Pareto frontier

Reference point for hypervolume:
- Latency: worst (highest) latency
- Accuracy: worst (lowest) accuracy

Output: results/pareto_analysis.csv
Schema: method,task,hypervolume,num_pareto_points,dominance_ratio,pareto_efficiency
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.theory.pareto import (
    compute_pareto_frontier,
    compute_hypervolume,
    dominance_ratio
)
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.pareto")


def load_results(results_dir: Path) -> tuple:
    """Load baseline and planner results."""
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    return baseline_df, planner_df


def analyze_pareto(baseline_df, planner_df, task):
    """Analyze Pareto frontier for a task.
    
    Objectives:
    - Minimize latency (lat_p95_ms)
    - Maximize accuracy
    """
    LOGGER.info(f"\nAnalyzing {task.upper()}...")
    
    # Filter by task
    baseline_task = baseline_df[baseline_df['task'] == task]
    planner_task = planner_df[planner_df['task'] == task]
    
    # Aggregate over seeds (use mean)
    baseline_agg = baseline_task.groupby(['method', 'deadline_ms']).agg({
        'lat_p95_ms': 'mean',
        'accuracy': 'mean'
    }).reset_index()
    
    planner_agg = planner_task.groupby(['threshold', 'deadline_ms']).agg({
        'lat_p95_ms': 'mean',
        'accuracy': 'mean'
    }).reset_index()
    
    # Extract points for each method
    results = []
    
    # Reference point (worst latency, worst accuracy)
    all_points = pd.concat([
        baseline_agg[['lat_p95_ms', 'accuracy']],
        planner_agg[['lat_p95_ms', 'accuracy']]
    ])
    ref_latency = all_points['lat_p95_ms'].max()
    ref_accuracy = all_points['accuracy'].min()
    
    LOGGER.info(f"  Reference point: latency={ref_latency:.1f}ms, accuracy={ref_accuracy:.3f}")
    
    # Analyze each baseline method
    for method in baseline_agg['method'].unique():
        method_points = baseline_agg[baseline_agg['method'] == method][['lat_p95_ms', 'accuracy']].values
        
        # Compute Pareto frontier
        pareto_points = compute_pareto_frontier(
            method_points,
            objectives=['minimize', 'maximize']
        )
        
        # Compute hypervolume
        hv = compute_hypervolume(
            pareto_points,
            reference_point=np.array([ref_latency, ref_accuracy]),
            objectives=['minimize', 'maximize']
        )
        
        # Pareto efficiency
        pareto_eff = len(pareto_points) / len(method_points) if len(method_points) > 0 else 0.0
        
        LOGGER.info(f"  {method}: HV={hv:.1f}, Pareto points={len(pareto_points)}/{len(method_points)}")
        
        results.append({
            'method': method,
            'task': task,
            'hypervolume': hv,
            'num_points': len(method_points),
            'num_pareto_points': len(pareto_points),
            'pareto_efficiency': pareto_eff,
            'dominance_ratio': 0.0  # Computed later
        })
    
    # Analyze planner
    planner_points = planner_agg[['lat_p95_ms', 'accuracy']].values
    
    pareto_points_planner = compute_pareto_frontier(
        planner_points,
        objectives=['minimize', 'maximize']
    )
    
    hv_planner = compute_hypervolume(
        pareto_points_planner,
        reference_point=np.array([ref_latency, ref_accuracy]),
        objectives=['minimize', 'maximize']
    )
    
    pareto_eff_planner = len(pareto_points_planner) / len(planner_points) if len(planner_points) > 0 else 0.0
    
    LOGGER.info(f"  CascadePlanner: HV={hv_planner:.1f}, Pareto points={len(pareto_points_planner)}/{len(planner_points)}")
    
    # Compute dominance ratio: fraction of baseline points dominated by planner
    all_baseline_points = baseline_agg[['lat_p95_ms', 'accuracy']].values
    dom_ratio = dominance_ratio(
        planner_points,
        all_baseline_points,
        objectives=['minimize', 'maximize']
    )
    
    LOGGER.info(f"  Dominance ratio: {dom_ratio:.2%} of baseline points dominated by planner")
    
    results.append({
        'method': 'CascadePlanner',
        'task': task,
        'hypervolume': hv_planner,
        'num_points': len(planner_points),
        'num_pareto_points': len(pareto_points_planner),
        'pareto_efficiency': pareto_eff_planner,
        'dominance_ratio': dom_ratio
    })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    results_dir = Path("results")
    
    LOGGER.info("Loading results...")
    baseline_df, planner_df = load_results(results_dir)
    
    all_results = []
    
    for task in ['text', 'vision']:
        results = analyze_pareto(baseline_df, planner_df, task)
        all_results.extend(results)
    
    # Save
    results_df = pd.DataFrame(all_results)
    output_path = results_dir / "pareto_analysis.csv"
    save_csv(results_df, output_path)
    
    LOGGER.info(f"\nComplete. Results: {output_path}")
    
    # Summary
    LOGGER.info("\nSummary:")
    for _, row in results_df.iterrows():
        LOGGER.info(f"  {row['method']} ({row['task']}): HV={row['hypervolume']:.1f}, Dom={row['dominance_ratio']:.2%}")


if __name__ == "__main__":
    main()

