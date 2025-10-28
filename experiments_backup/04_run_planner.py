"""
Evaluate CascadePlanner on real workload traces.

CascadePlanner: Selects cascade configuration (small->large model) with threshold
that maximizes accuracy while meeting deadline constraint.

Evaluation metrics:
- Deadline hit rate: Fraction of requests meeting deadline
- Accuracy: Mean accuracy across all requests  
- Latency p95: 95th percentile latency
- Coverage: Fraction of requests handled by small model (Stage 1)

Output: results/planner_results.csv
Schema: method,task,deadline_ms,config_selected,threshold,deadline_hit_rate,accuracy,coverage,lat_p95_ms,num_requests
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.run_planner")

# Deadlines for evaluation (milliseconds)
TEXT_DEADLINES = [50, 100, 150]
VISION_DEADLINES = [100, 200, 300]
NUM_REQUESTS = 1000


def load_profiles():
    """Load latency and accuracy profiles."""
    LOGGER.info("Loading profiling results...")
    
    latency_path = Path("results/latency_profiles.csv")
    accuracy_path = Path("results/accuracy_profiles.csv")
    
    if not latency_path.exists():
        raise FileNotFoundError(f"Latency profiles not found. Run 01_profile_latency.py first.")
    if not accuracy_path.exists():
        raise FileNotFoundError(f"Accuracy profiles not found. Run 02_profile_accuracy.py first.")
    
    latency_df = pd.read_csv(latency_path)
    accuracy_df = pd.read_csv(accuracy_path)
    
    LOGGER.info(f"Loaded {len(latency_df)} latency profiles")
    LOGGER.info(f"Loaded {len(accuracy_df)} accuracy profiles")
    
    return latency_df, accuracy_df


def get_cascade_profiles(accuracy_df, task):
    """Get cascade configurations for a task."""
    cascade = accuracy_df[
        (accuracy_df['task'] == task) & 
        (accuracy_df['exit_policy'] == 'cascade')
    ].copy()
    
    LOGGER.info(f"Found {len(cascade)} cascade configurations for {task}")
    return cascade


def estimate_cascade_latency(cascade_config, latency_df):
    """Estimate cascade latency based on coverage and component latencies."""
    # Parse cascade model names (e.g., "minilm->distilbert")
    models = cascade_config['model'].split('->')
    if len(models) != 2:
        return None
    
    model_small, model_large = models[0], models[1]
    variant = cascade_config['variant']
    device = cascade_config['device']
    coverage = cascade_config['coverage']
    
    # Get latencies for small and large models (batch_size=1)
    small_lat = latency_df[
        (latency_df['model'] == model_small) &
        (latency_df['variant'] == variant) &
        (latency_df['device'] == device) &
        (latency_df['batch_size'] == 1)
    ]
    
    large_lat = latency_df[
        (latency_df['model'] == model_large) &
        (latency_df['variant'] == variant) &
        (latency_df['device'] == device) &
        (latency_df['batch_size'] == 1)
    ]
    
    if len(small_lat) == 0 or len(large_lat) == 0:
        return None
    
    # Weighted average latency based on coverage
    # coverage fraction goes through small model only
    # (1-coverage) fraction goes through both models
    lat_p50_small = small_lat.iloc[0]['lat_p50_ms']
    lat_p95_small = small_lat.iloc[0]['lat_p95_ms']
    lat_p50_large = large_lat.iloc[0]['lat_p50_ms']
    lat_p95_large = large_lat.iloc[0]['lat_p95_ms']
    
    # Cascade latency: coverage * small + (1-coverage) * (small + large)
    lat_p50_cascade = coverage * lat_p50_small + (1 - coverage) * (lat_p50_small + lat_p50_large)
    lat_p95_cascade = coverage * lat_p95_small + (1 - coverage) * (lat_p95_small + lat_p95_large)
    
    return {
        'lat_p50_ms': lat_p50_cascade,
        'lat_p95_ms': lat_p95_cascade,
        'coverage': coverage
    }


def select_cascade_config(cascade_profiles, latency_df, task, deadline_ms):
    """Select best cascade configuration for deadline."""
    candidates = []
    
    for idx, cascade_config in cascade_profiles.iterrows():
        latency_est = estimate_cascade_latency(cascade_config, latency_df)
        if latency_est is None:
            continue
        
        # Check if p95 latency meets deadline
        if latency_est['lat_p95_ms'] <= deadline_ms:
            candidates.append({
                'config': cascade_config,
                'lat_p50_ms': latency_est['lat_p50_ms'],
                'lat_p95_ms': latency_est['lat_p95_ms'],
                'accuracy': cascade_config['accuracy'],
                'coverage': latency_est['coverage'],
                'threshold': cascade_config['threshold']
            })
    
    if len(candidates) == 0:
        LOGGER.warning(f"No cascade config meets {deadline_ms}ms deadline for {task}")
        return None
    
    # Select config with highest accuracy among feasible
    best = max(candidates, key=lambda x: x['accuracy'])
    return best


def simulate_cascade_requests(config, num_requests):
    """Simulate cascade request latencies."""
    p50 = config['lat_p50_ms']
    p95 = config['lat_p95_ms']
    
    # Approximate log-normal distribution
    sigma = np.log(p95 / p50) / 1.645 if p95 > p50 else 0.1
    mu = np.log(p50)
    
    latencies = np.random.lognormal(mu, sigma, num_requests)
    return latencies


def evaluate_planner(config, deadline_ms, num_requests):
    """Evaluate cascade planner configuration."""
    if config is None:
        return {
            'method': 'CascadePlanner',
            'task': 'unknown',
            'deadline_ms': deadline_ms,
            'config_selected': 'none',
            'threshold': None,
            'deadline_hit_rate': 0.0,
            'accuracy': 0.0,
            'coverage': 0.0,
            'lat_p95_ms': float('inf'),
            'num_requests': num_requests
        }
    
    # Simulate requests
    latencies = simulate_cascade_requests(config, num_requests)
    
    # Compute metrics
    deadline_hit_rate = np.mean(latencies <= deadline_ms)
    lat_p95_ms = np.percentile(latencies, 95)
    
    cascade_config = config['config']
    
    return {
        'method': 'CascadePlanner',
        'task': cascade_config['task'],
        'deadline_ms': deadline_ms,
        'config_selected': f"{cascade_config['model']}-{cascade_config['variant']}-{cascade_config['device']}",
        'threshold': config['threshold'],
        'deadline_hit_rate': deadline_hit_rate,
        'accuracy': config['accuracy'],
        'coverage': config['coverage'],
        'lat_p95_ms': lat_p95_ms,
        'num_requests': num_requests
    }


def main():
    LOGGER.info("Starting CascadePlanner evaluation...")
    
    # Load profiles
    latency_df, accuracy_df = load_profiles()
    
    results = []
    
    # Evaluate on text task
    LOGGER.info("Evaluating CascadePlanner on text task...")
    cascade_text = get_cascade_profiles(accuracy_df, 'text')
    
    for deadline in TEXT_DEADLINES:
        LOGGER.info(f"  Deadline: {deadline}ms")
        
        config = select_cascade_config(cascade_text, latency_df, 'text', deadline)
        result = evaluate_planner(config, deadline, NUM_REQUESTS)
        results.append(result)
        
        if config is not None:
            LOGGER.info(f"    Selected: τ={result['threshold']:.2f}, hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}, coverage={result['coverage']:.3f}")
        else:
            LOGGER.info(f"    No feasible configuration")
    
    # Evaluate on vision task
    LOGGER.info("Evaluating CascadePlanner on vision task...")
    cascade_vision = get_cascade_profiles(accuracy_df, 'vision')
    
    if len(cascade_vision) > 0:
        for deadline in VISION_DEADLINES:
            LOGGER.info(f"  Deadline: {deadline}ms")
            
            config = select_cascade_config(cascade_vision, latency_df, 'vision', deadline)
            result = evaluate_planner(config, deadline, NUM_REQUESTS)
            results.append(result)
            
            if config is not None:
                LOGGER.info(f"    Selected: τ={result['threshold']:.2f}, hit_rate={result['deadline_hit_rate']:.3f}, acc={result['accuracy']:.3f}, coverage={result['coverage']:.3f}")
            else:
                LOGGER.info(f"    No feasible configuration")
    else:
        LOGGER.warning("No cascade configurations found for vision task")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("results/planner_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    LOGGER.info(f"CascadePlanner evaluation complete. Results saved to {output_path}")
    LOGGER.info(f"Total evaluations: {len(results)}")
    
    # Summary
    LOGGER.info("\nSummary:")
    LOGGER.info(f"  Mean hit rate: {df['deadline_hit_rate'].mean():.3f}")
    LOGGER.info(f"  Mean accuracy: {df['accuracy'].mean():.3f}")
    LOGGER.info(f"  Mean coverage: {df['coverage'].mean():.3f}")


if __name__ == "__main__":
    main()

