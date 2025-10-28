"""Failure analysis and graceful degradation strategies.

Analyzes what happens when all available models miss the deadline constraint.

Strategies evaluated:
1. Best-effort: Use fastest model even if it misses deadline
2. Timeout: Return error after deadline expires
3. Approximate: Use cached/approximate results

Metrics:
- Frequency of deadline misses by severity
- Degradation strategy effectiveness
- User-perceived quality under failures

Outputs:
- results/failure_miss_analysis.csv: Deadline miss frequency and severity
- results/failure_degradation_strategies.csv: Strategy effectiveness comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.failure_analysis")

TEXT_DEADLINES = [50, 100, 150]
VISION_DEADLINES = [50, 100, 150]


def load_profiles():
    """Load latency and accuracy profiles."""
    latency_path = Path("results/latency_profiles.csv")
    accuracy_path = Path("results/accuracy_profiles.csv")
    
    if not latency_path.exists():
        raise FileNotFoundError("Latency profiles not found. Run 01_profile_latency.py first.")
    if not accuracy_path.exists():
        raise FileNotFoundError("Accuracy profiles not found. Run 02_profile_accuracy.py first.")
    
    latency_df = pd.read_csv(latency_path)
    accuracy_df = pd.read_csv(accuracy_path)
    
    # Merge profiles
    merged = latency_df.merge(
        accuracy_df[['task', 'model', 'variant', 'device', 'accuracy']],
        on=['task', 'model', 'variant', 'device'],
        how='left'
    )
    
    return merged


def analyze_deadline_misses(profiles_df, task, deadline_ms):
    """Analyze configurations that miss the deadline."""
    task_profiles = profiles_df[profiles_df['task'] == task].copy()
    
    # Identify configs that miss deadline (p95 latency > deadline)
    task_profiles['misses_deadline'] = task_profiles['lat_p95_ms'] > deadline_ms
    
    miss_rate = task_profiles['misses_deadline'].mean()
    
    # Find fastest config (even if it misses deadline)
    fastest_config = task_profiles.loc[task_profiles['lat_p95_ms'].idxmin()]
    
    # Find most accurate config that meets deadline
    meets_deadline = task_profiles[~task_profiles['misses_deadline']]
    if len(meets_deadline) > 0:
        best_within_deadline = meets_deadline.loc[meets_deadline['accuracy'].idxmax()]
        has_solution = True
    else:
        best_within_deadline = None
        has_solution = False
    
    return {
        'task': task,
        'deadline_ms': deadline_ms,
        'total_configs': len(task_profiles),
        'configs_miss_deadline': int(task_profiles['misses_deadline'].sum()),
        'miss_rate': miss_rate,
        'has_solution': has_solution,
        'fastest_config_latency_ms': fastest_config['lat_p95_ms'],
        'fastest_config_accuracy': fastest_config['accuracy'],
        'best_within_deadline_latency_ms': best_within_deadline['lat_p95_ms'] if has_solution else None,
        'best_within_deadline_accuracy': best_within_deadline['accuracy'] if has_solution else None
    }


def evaluate_degradation_strategy(profiles_df, task, deadline_ms, strategy='best_effort'):
    """
    Evaluate a degradation strategy when no config meets deadline.
    
    Strategies:
    - best_effort: Use fastest model even if it misses deadline
    - timeout: Return error (no result)
    - approximate: Use cached result (simulated as 80% of best accuracy)
    """
    task_profiles = profiles_df[profiles_df['task'] == task].copy()
    
    # Check if any config meets deadline
    meets_deadline = task_profiles[task_profiles['lat_p95_ms'] <= deadline_ms]
    
    if len(meets_deadline) > 0:
        # Normal case: use best config within deadline
        best_config = meets_deadline.loc[meets_deadline['accuracy'].idxmax()]
        return {
            'strategy': strategy,
            'degraded': False,
            'latency_ms': best_config['lat_p95_ms'],
            'accuracy': best_config['accuracy'],
            'user_satisfaction': 1.0
        }
    else:
        # Failure case: apply degradation strategy
        if strategy == 'best_effort':
            # Use fastest model
            fastest = task_profiles.loc[task_profiles['lat_p95_ms'].idxmin()]
            # User satisfaction decreases with latency overshoot
            overshoot_ratio = fastest['lat_p95_ms'] / deadline_ms
            user_satisfaction = max(0.0, 1.0 - 0.3 * (overshoot_ratio - 1.0))
            
            return {
                'strategy': strategy,
                'degraded': True,
                'latency_ms': fastest['lat_p95_ms'],
                'accuracy': fastest['accuracy'],
                'user_satisfaction': user_satisfaction
            }
        
        elif strategy == 'timeout':
            # Return error
            return {
                'strategy': strategy,
                'degraded': True,
                'latency_ms': deadline_ms,
                'accuracy': 0.0,
                'user_satisfaction': 0.0
            }
        
        elif strategy == 'approximate':
            # Use cached/approximate result
            # Simulate as 80% of best possible accuracy, instant response
            best_accuracy = task_profiles['accuracy'].max()
            approx_accuracy = best_accuracy * 0.8
            
            return {
                'strategy': strategy,
                'degraded': True,
                'latency_ms': 0.0,  # Instant from cache
                'accuracy': approx_accuracy,
                'user_satisfaction': 0.7  # Reduced due to approximation
            }


def main():
    LOGGER.info("Starting failure analysis...")
    
    # Load profiles
    profiles_df = load_profiles()
    
    miss_analysis_results = []
    degradation_results = []
    
    # Analyze deadline misses for each task and deadline
    LOGGER.info("\nAnalyzing deadline miss rates...")
    
    for task in ['text', 'vision']:
        deadlines = TEXT_DEADLINES if task == 'text' else VISION_DEADLINES
        
        for deadline in deadlines:
            LOGGER.info(f"\n{task} @ {deadline}ms:")
            
            # Analyze misses
            miss_analysis = analyze_deadline_misses(profiles_df, task, deadline)
            miss_analysis_results.append(miss_analysis)
            
            LOGGER.info(f"  Miss rate: {miss_analysis['miss_rate']:.1%}")
            LOGGER.info(f"  Has solution: {miss_analysis['has_solution']}")
            LOGGER.info(f"  Fastest config: {miss_analysis['fastest_config_latency_ms']:.1f}ms, acc={miss_analysis['fastest_config_accuracy']:.3f}")
            
            # Evaluate degradation strategies
            for strategy in ['best_effort', 'timeout', 'approximate']:
                result = evaluate_degradation_strategy(profiles_df, task, deadline, strategy)
                result['task'] = task
                result['deadline_ms'] = deadline
                degradation_results.append(result)
                
                if result['degraded']:
                    LOGGER.info(f"  {strategy}: lat={result['latency_ms']:.1f}ms, acc={result['accuracy']:.3f}, satisfaction={result['user_satisfaction']:.2f}")
    
    # Save results
    miss_df = pd.DataFrame(miss_analysis_results)
    miss_output_path = Path("results/failure_miss_analysis.csv")
    miss_output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(miss_df, miss_output_path)
    
    degrad_df = pd.DataFrame(degradation_results)
    degrad_output_path = Path("results/failure_degradation_strategies.csv")
    save_csv(degrad_df, degrad_output_path)
    
    LOGGER.info(f"\nFailure analysis complete.")
    LOGGER.info(f"Miss analysis saved to {miss_output_path}")
    LOGGER.info(f"Degradation strategies saved to {degrad_output_path}")
    
    # Summary
    LOGGER.info("\nSummary:")
    LOGGER.info(f"  Overall miss rate: {miss_df['miss_rate'].mean():.1%}")
    LOGGER.info(f"  Configs with solutions: {miss_df['has_solution'].sum()}/{len(miss_df)}")
    
    degraded_only = degrad_df[degrad_df['degraded']]
    if len(degraded_only) > 0:
        LOGGER.info(f"\nDegradation strategy comparison (when deadline missed):")
        for strategy in degraded_only['strategy'].unique():
            subset = degraded_only[degraded_only['strategy'] == strategy]
            LOGGER.info(f"  {strategy}:")
            LOGGER.info(f"    Mean satisfaction: {subset['user_satisfaction'].mean():.2f}")
            LOGGER.info(f"    Mean accuracy: {subset['accuracy'].mean():.3f}")


if __name__ == "__main__":
    main()

