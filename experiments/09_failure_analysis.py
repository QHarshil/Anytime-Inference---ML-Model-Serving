"""Failure analysis and graceful degradation strategies.

Analyzes what happens when all models miss deadlines and evaluates
graceful degradation strategies:

1. Deadline miss analysis: Characterize when and why deadlines are missed
2. Degradation strategies:
   - Fast fallback: Switch to fastest model when deadline is tight
   - Accuracy fallback: Switch to most accurate model when deadline is relaxed
   - Adaptive threshold: Adjust cascade threshold based on load

Outputs:
- results/failure_miss_analysis.csv
- results/failure_degradation_strategies.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.failure")


def analyze_deadline_misses(baseline_df, planner_df):
    """Analyze when and why deadlines are missed."""
    LOGGER.info("\nDeadline miss analysis:")
    
    results = []
    
    for task in ['text', 'vision']:
        task_baseline = baseline_df[baseline_df['task'] == task]
        task_planner = planner_df[planner_df['task'] == task]
        
        for method in task_baseline['method'].unique():
            method_data = task_baseline[task_baseline['method'] == method]
            
            miss_rate = 1.0 - method_data['deadline_hit_rate'].mean()
            avg_latency = method_data['lat_p95_ms'].mean()
            avg_deadline = method_data['deadline_ms'].mean()
            latency_overhead = ((avg_latency - avg_deadline) / avg_deadline * 100) if avg_deadline > 0 else 0.0
            
            LOGGER.info(f"  {method} ({task}): miss_rate={miss_rate:.2%}, "
                       f"avg_latency={avg_latency:.1f}ms, avg_deadline={avg_deadline:.1f}ms, "
                       f"overhead={latency_overhead:+.1f}%")
            
            results.append({
                'method': method,
                'task': task,
                'miss_rate': miss_rate,
                'hit_rate': 1.0 - miss_rate,
                'avg_latency_p95_ms': avg_latency,
                'avg_deadline_ms': avg_deadline,
                'latency_overhead_pct': latency_overhead,
                'num_evaluations': len(method_data)
            })
        
        # Planner
        planner_miss_rate = 1.0 - task_planner['deadline_hit_rate'].mean()
        planner_latency = task_planner['lat_p95_ms'].mean()
        planner_deadline = task_planner['deadline_ms'].mean()
        planner_overhead = ((planner_latency - planner_deadline) / planner_deadline * 100) if planner_deadline > 0 else 0.0
        
        LOGGER.info(f"  CascadePlanner ({task}): miss_rate={planner_miss_rate:.2%}, "
                   f"avg_latency={planner_latency:.1f}ms, avg_deadline={planner_deadline:.1f}ms, "
                   f"overhead={planner_overhead:+.1f}%")
        
        results.append({
            'method': 'CascadePlanner',
            'task': task,
            'miss_rate': planner_miss_rate,
            'hit_rate': 1.0 - planner_miss_rate,
            'avg_latency_p95_ms': planner_latency,
            'avg_deadline_ms': planner_deadline,
            'latency_overhead_pct': planner_overhead,
            'num_evaluations': len(task_planner)
        })
    
    return results


def evaluate_degradation_strategies(baseline_df, planner_df):
    """Evaluate graceful degradation strategies."""
    LOGGER.info("\nGraceful degradation strategies:")
    
    results = []
    
    for task in ['text', 'vision']:
        task_baseline = baseline_df[baseline_df['task'] == task]
        task_planner = planner_df[planner_df['task'] == task]
        
        # Strategy 1: Fast fallback (use StaticSmall when deadline is tight)
        static_small = task_baseline[task_baseline['method'] == 'StaticSmall']
        if len(static_small) > 0:
            hit_rate = static_small['deadline_hit_rate'].mean()
            accuracy = static_small['accuracy'].mean()
            latency = static_small['lat_p95_ms'].mean()
            
            LOGGER.info(f"  {task} - Fast fallback: hit_rate={hit_rate:.3f}, accuracy={accuracy:.3f}, latency={latency:.1f}ms")
            
            results.append({
                'strategy': 'fast_fallback',
                'task': task,
                'deadline_hit_rate': hit_rate,
                'accuracy': accuracy,
                'latency_p95_ms': latency,
                'description': 'Switch to fastest model when deadline is tight'
            })
        
        # Strategy 2: Accuracy fallback (use StaticLarge when deadline is relaxed)
        static_large = task_baseline[task_baseline['method'] == 'StaticLarge']
        if len(static_large) > 0:
            hit_rate = static_large['deadline_hit_rate'].mean()
            accuracy = static_large['accuracy'].mean()
            latency = static_large['lat_p95_ms'].mean()
            
            LOGGER.info(f"  {task} - Accuracy fallback: hit_rate={hit_rate:.3f}, accuracy={accuracy:.3f}, latency={latency:.1f}ms")
            
            results.append({
                'strategy': 'accuracy_fallback',
                'task': task,
                'deadline_hit_rate': hit_rate,
                'accuracy': accuracy,
                'latency_p95_ms': latency,
                'description': 'Switch to most accurate model when deadline is relaxed'
            })
        
        # Strategy 3: Adaptive threshold (adjust cascade threshold based on deadline)
        # For tight deadlines, use lower threshold (more fast model)
        # For relaxed deadlines, use higher threshold (more accurate model)
        if len(task_planner) > 0:
            # Group by deadline and find best threshold for each
            for deadline_ms in task_planner['deadline_ms'].unique():
                deadline_data = task_planner[task_planner['deadline_ms'] == deadline_ms]
                best = deadline_data.loc[deadline_data['deadline_hit_rate'].idxmax()]
                
                LOGGER.info(f"  {task} - Adaptive threshold (deadline={deadline_ms}ms): "
                           f"threshold={best['threshold']:.2f}, hit_rate={best['deadline_hit_rate']:.3f}, "
                           f"accuracy={best['accuracy']:.3f}")
                
                results.append({
                    'strategy': 'adaptive_threshold',
                    'task': task,
                    'deadline_ms': deadline_ms,
                    'optimal_threshold': best['threshold'],
                    'deadline_hit_rate': best['deadline_hit_rate'],
                    'accuracy': best['accuracy'],
                    'latency_p95_ms': best['lat_p95_ms'],
                    'description': f'Adaptive cascade threshold for deadline={deadline_ms}ms'
                })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    results_dir = Path("results")
    
    LOGGER.info("Loading results...")
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    
    # 1. Deadline miss analysis
    miss_results = analyze_deadline_misses(baseline_df, planner_df)
    miss_df = pd.DataFrame(miss_results)
    miss_output = results_dir / "failure_miss_analysis.csv"
    save_csv(miss_df, miss_output)
    LOGGER.info(f"\nMiss analysis saved: {miss_output}")
    
    # 2. Degradation strategies
    degradation_results = evaluate_degradation_strategies(baseline_df, planner_df)
    degradation_df = pd.DataFrame(degradation_results)
    degradation_output = results_dir / "failure_degradation_strategies.csv"
    save_csv(degradation_df, degradation_output)
    LOGGER.info(f"Degradation strategies saved: {degradation_output}")
    
    LOGGER.info("\nComplete.")


if __name__ == "__main__":
    main()

