"""Evaluate baseline planners.

Baselines:
1. StaticSmall: Always use fastest model (lowest latency)
2. StaticLarge: Always use most accurate model (highest accuracy)
3. ThroughputAutotuner: Select model with best throughput under deadline
4. INFaaS-style: Variant-aware selection (adapted from Romero et al., ATC'20)

Evaluation:
- Simulates request workloads with log-normal latency sampling
- Measures deadline hit rate, accuracy, and latency statistics

Output: results/baseline_results.csv
Schema: method,task,deadline_ms,config_selected,deadline_hit_rate,accuracy,lat_p95_ms,num_requests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils.io import save_csv
from src.utils.logger import get_logger
from src.planner.infaas_style_baseline import INFaaSStyleBaseline

LOGGER = get_logger("experiments.run_baselines")

# Deadlines for evaluation (milliseconds)
TEXT_DEADLINES = [50, 100, 150]
VISION_DEADLINES = [100, 200, 300]
NUM_REQUESTS = 1000  # Simulate 1000 requests per deadline


def load_profiles():
    """Load latency and accuracy profiles."""
    LOGGER.info("Loading profiling results...")
    
    latency_path = Path("results/latency_profiles.csv")
    accuracy_path = Path("results/accuracy_profiles.csv")
    
    if not latency_path.exists():
        raise FileNotFoundError(f"Latency profiles not found: {latency_path}. Run 01_profile_latency.py first.")
    if not accuracy_path.exists():
        raise FileNotFoundError(f"Accuracy profiles not found: {accuracy_path}. Run 02_profile_accuracy.py first.")
    
    latency_df = pd.read_csv(latency_path)
    accuracy_df = pd.read_csv(accuracy_path)
    
    LOGGER.info(f"Loaded {len(latency_df)} latency profiles")
    LOGGER.info(f"Loaded {len(accuracy_df)} accuracy profiles")
    
    return latency_df, accuracy_df


def merge_profiles(latency_df, accuracy_df):
    """Merge latency and accuracy profiles on common keys."""
    # For non-cascade configs, merge on task, model, variant, device
    non_cascade_latency = latency_df[latency_df['batch_size'] == 1].copy()
    non_cascade_accuracy = accuracy_df[accuracy_df['exit_policy'] == 'none'].copy()
    
    merged = pd.merge(
        non_cascade_latency,
        non_cascade_accuracy[['task', 'model', 'variant', 'device', 'accuracy']],
        on=['task', 'model', 'variant', 'device'],
        how='inner'
    )
    
    LOGGER.info(f"Merged {len(merged)} configurations with both latency and accuracy data")
    
    return merged


def select_static_small(profiles, task):
    """Select fastest configuration (lowest p95 latency)."""
    task_profiles = profiles[profiles['task'] == task]
    if len(task_profiles) == 0:
        return None
    
    fastest = task_profiles.loc[task_profiles['lat_p95_ms'].idxmin()]
    return fastest.to_dict()


def select_static_large(profiles, task):
    """Select most accurate configuration (highest accuracy)."""
    task_profiles = profiles[profiles['task'] == task]
    if len(task_profiles) == 0:
        return None
    
    most_accurate = task_profiles.loc[task_profiles['accuracy'].idxmax()]
    return most_accurate.to_dict()


def select_throughput_autotuner(profiles, task, deadline_ms):
    """Select configuration with best throughput under deadline constraint."""
    task_profiles = profiles[profiles['task'] == task]
    
    # Filter configs that can meet deadline (p95 latency < deadline)
    feasible = task_profiles[task_profiles['lat_p95_ms'] <= deadline_ms]
    
    if len(feasible) == 0:
        # No config meets deadline, select fastest
        LOGGER.warning(f"No configuration meets {deadline_ms}ms deadline for {task}, selecting fastest")
        return select_static_small(profiles, task)
    
    # Among feasible, select highest throughput
    best = feasible.loc[feasible['throughput_items_per_sec'].idxmax()]
    return best.to_dict()


def simulate_requests(config, num_requests):
    """Simulate requests using latency distribution from profiling."""
    # Use p50 and p95 to approximate a log-normal distribution
    p50 = config['lat_p50_ms']
    p95 = config['lat_p95_ms']
    
    # Approximate log-normal parameters
    # For log-normal: p95 ≈ median * exp(1.645 * sigma)
    sigma = np.log(p95 / p50) / 1.645 if p95 > p50 else 0.1
    mu = np.log(p50)
    
    # Generate latencies
    latencies = np.random.lognormal(mu, sigma, num_requests)
    
    return latencies


def evaluate_baseline(method_name, config, deadline_ms, num_requests):
    """Evaluate a baseline configuration on simulated requests."""
    if config is None:
        return {
            'method': method_name,
            'task': 'unknown',
            'deadline_ms': deadline_ms,
            'config_selected': 'none',
            'deadline_hit_rate': 0.0,
            'accuracy': 0.0,
            'lat_p95_ms': float('inf'),
            'num_requests': num_requests
        }
    
    # Simulate request latencies
    latencies = simulate_requests(config, num_requests)
    
    # Compute metrics
    deadline_hit_rate = np.mean(latencies <= deadline_ms)
    accuracy = config['accuracy']
    lat_p95_ms = np.percentile(latencies, 95)
    
    return {
        'method': method_name,
        'task': config['task'],
        'deadline_ms': deadline_ms,
        'config_selected': f"{config['model']}-{config['variant']}-{config['device']}",
        'deadline_hit_rate': deadline_hit_rate,
        'accuracy': accuracy,
        'lat_p95_ms': lat_p95_ms,
        'num_requests': num_requests
    }


def main():
    LOGGER.info("Starting baseline evaluation...")
    
    # Load profiles
    latency_df, accuracy_df = load_profiles()
    profiles = merge_profiles(latency_df, accuracy_df)
    
    results = []
    
    # Evaluate on text task
    LOGGER.info("Evaluating baselines on text task...")
    for deadline in TEXT_DEADLINES:
        LOGGER.info(f"  Deadline: {deadline}ms")
        
        # StaticSmall
        config = select_static_small(profiles, 'text')
        result = evaluate_baseline('StaticSmall', config, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    StaticSmall: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
        
        # StaticLarge
        config = select_static_large(profiles, 'text')
        result = evaluate_baseline('StaticLarge', config, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    StaticLarge: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
        
        # ThroughputAutotuner
        config = select_throughput_autotuner(profiles, 'text', deadline)
        result = evaluate_baseline('ThroughputAutotuner', config, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    ThroughputAutotuner: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
        
        # INFaaS-style
        infaas = INFaaSStyleBaseline(profiles)
        config_dict = infaas.select('text', deadline)
        result = evaluate_baseline('INFaaS-style', config_dict, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    INFaaS-style: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
    
    # Evaluate on vision task
    LOGGER.info("Evaluating baselines on vision task...")
    for deadline in VISION_DEADLINES:
        LOGGER.info(f"  Deadline: {deadline}ms")
        
        # StaticSmall
        config = select_static_small(profiles, 'vision')
        result = evaluate_baseline('StaticSmall', config, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    StaticSmall: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
        
        # StaticLarge
        config = select_static_large(profiles, 'vision')
        result = evaluate_baseline('StaticLarge', config, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    StaticLarge: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
        
        # ThroughputAutotuner
        config = select_throughput_autotuner(profiles, 'vision', deadline)
        result = evaluate_baseline('ThroughputAutotuner', config, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    ThroughputAutotuner: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
        
        # INFaaS-style
        infaas = INFaaSStyleBaseline(profiles)
        config_dict = infaas.select('vision', deadline)
        result = evaluate_baseline('INFaaS-style', config_dict, deadline, NUM_REQUESTS)
        results.append(result)
        LOGGER.info(f"    INFaaS-style: hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("results/baseline_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    LOGGER.info(f"Baseline evaluation complete. Results saved to {output_path}")
    LOGGER.info(f"Total evaluations: {len(results)}")
    
    # Summary
    LOGGER.info("\nSummary by method:")
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        LOGGER.info(f"  {method}:")
        LOGGER.info(f"    Mean hit rate: {method_df['deadline_hit_rate'].mean():.3f}")
        LOGGER.info(f"    Mean accuracy: {method_df['accuracy'].mean():.3f}")


if __name__ == "__main__":
    main()

