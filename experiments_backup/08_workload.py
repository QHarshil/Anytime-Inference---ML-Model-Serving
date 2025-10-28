"""
Workload sensitivity analysis.

Evaluates how CascadePlanner and baselines perform under different traffic patterns:
1. Steady workload: Constant request rate (Poisson arrivals)
2. Bursty workload: Alternating between low and high request rates

Metrics:
- Deadline hit rate under different traffic patterns
- Accuracy under different traffic patterns
- Queue length statistics (mean, p95)
- Response time statistics (mean, p95)

Output: results/workload_results.csv
Schema: method,workload_type,deadline_ms,deadline_hit_rate,accuracy,mean_queue_length,p95_queue_length,mean_response_time_ms,p95_response_time_ms
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.workload")

# Simulation parameters
SIMULATION_DURATION_SEC = 300  # 5 minutes
STEADY_RATE_RPS = 10  # requests per second
BURSTY_LOW_RPS = 5
BURSTY_HIGH_RPS = 20
BURSTY_PERIOD_SEC = 30  # Switch every 30 seconds

TEXT_DEADLINES = [50, 100, 150]


def generate_steady_arrivals(duration_sec, rate_rps):
    """Generate Poisson arrival times for steady workload."""
    num_requests = int(duration_sec * rate_rps * 1.5)  # Overestimate
    inter_arrival_times = np.random.exponential(1.0 / rate_rps, num_requests)
    arrival_times = np.cumsum(inter_arrival_times)
    
    # Filter to duration
    arrival_times = arrival_times[arrival_times <= duration_sec]
    
    LOGGER.info(f"Generated {len(arrival_times)} steady arrivals over {duration_sec}s (rate={rate_rps} rps)")
    
    return arrival_times


def generate_bursty_arrivals(duration_sec, low_rps, high_rps, period_sec):
    """Generate arrival times for bursty workload."""
    arrivals = []
    current_time = 0.0
    
    while current_time < duration_sec:
        # Determine current rate based on period
        period_position = current_time % (2 * period_sec)
        if period_position < period_sec:
            current_rate = low_rps
        else:
            current_rate = high_rps
        
        # Generate next arrival
        inter_arrival = np.random.exponential(1.0 / current_rate)
        current_time += inter_arrival
        
        if current_time <= duration_sec:
            arrivals.append(current_time)
    
    LOGGER.info(f"Generated {len(arrivals)} bursty arrivals over {duration_sec}s (low={low_rps}, high={high_rps} rps)")
    
    return np.array(arrivals)


def simulate_queue(arrival_times, service_times, deadline_ms):
    """
    Simulate M/G/1 queue with deadline constraints.
    
    Returns:
    - deadline_hit_rate: Fraction of requests meeting deadline
    - queue_lengths: Queue length at each arrival
    - response_times: Response time for each request (ms)
    """
    num_requests = len(arrival_times)
    completion_times = np.zeros(num_requests)
    queue_lengths = np.zeros(num_requests)
    response_times = np.zeros(num_requests)
    
    # Simulate queue
    for i in range(num_requests):
        arrival = arrival_times[i]
        service = service_times[i]
        
        # Determine when this request can start service
        if i == 0:
            start_time = arrival
            queue_length = 0
        else:
            # Start after previous request completes
            start_time = max(arrival, completion_times[i-1])
            # Count requests still in queue
            queue_length = np.sum(completion_times[:i] > arrival)
        
        completion_times[i] = start_time + service
        queue_lengths[i] = queue_length
        response_times[i] = (completion_times[i] - arrival) * 1000  # Convert to ms
    
    # Compute deadline hit rate
    deadline_hit_rate = np.mean(response_times <= deadline_ms)
    
    return {
        'deadline_hit_rate': deadline_hit_rate,
        'mean_queue_length': np.mean(queue_lengths),
        'p95_queue_length': np.percentile(queue_lengths, 95),
        'mean_response_time_ms': np.mean(response_times),
        'p95_response_time_ms': np.percentile(response_times, 95)
    }


def load_baseline_results():
    """Load baseline evaluation results."""
    baseline_path = Path("results/baseline_results.csv")
    planner_path = Path("results/planner_results.csv")
    
    if not baseline_path.exists():
        raise FileNotFoundError("Baseline results not found. Run 03_run_baselines.py first.")
    if not planner_path.exists():
        raise FileNotFoundError("Planner results not found. Run 04_run_planner.py first.")
    
    baseline_df = pd.read_csv(baseline_path)
    planner_df = pd.read_csv(planner_path)
    
    return baseline_df, planner_df


def evaluate_workload(method_name, config_latency_ms, config_accuracy, deadline_ms, workload_type, arrival_times):
    """Evaluate a configuration under a workload."""
    # Generate service times from latency distribution
    # Approximate as log-normal
    p50 = config_latency_ms
    p95 = config_latency_ms * 1.5  # Approximate
    
    sigma = np.log(p95 / p50) / 1.645 if p95 > p50 else 0.1
    mu = np.log(p50)
    
    service_times_ms = np.random.lognormal(mu, sigma, len(arrival_times))
    service_times_sec = service_times_ms / 1000.0
    
    # Simulate queue
    metrics = simulate_queue(arrival_times, service_times_sec, deadline_ms)
    
    return {
        'method': method_name,
        'workload_type': workload_type,
        'deadline_ms': deadline_ms,
        'config_latency_ms': config_latency_ms,
        'accuracy': config_accuracy,
        **metrics
    }


def main():
    LOGGER.info("Starting workload sensitivity analysis...")
    LOGGER.info(f"Simulation duration: {SIMULATION_DURATION_SEC}s")
    LOGGER.info(f"Steady rate: {STEADY_RATE_RPS} rps")
    LOGGER.info(f"Bursty rates: {BURSTY_LOW_RPS}/{BURSTY_HIGH_RPS} rps")
    
    # Generate workload traces
    LOGGER.info("\nGenerating workload traces...")
    steady_arrivals = generate_steady_arrivals(SIMULATION_DURATION_SEC, STEADY_RATE_RPS)
    bursty_arrivals = generate_bursty_arrivals(SIMULATION_DURATION_SEC, BURSTY_LOW_RPS, BURSTY_HIGH_RPS, BURSTY_PERIOD_SEC)
    
    # Load baseline and planner results
    baseline_df, planner_df = load_baseline_results()
    
    results = []
    
    # Evaluate each method under each workload
    for deadline in TEXT_DEADLINES:
        LOGGER.info(f"\nEvaluating deadline: {deadline}ms")
        
        # StaticSmall
        static_small = baseline_df[(baseline_df['method'] == 'StaticSmall') & (baseline_df['deadline_ms'] == deadline) & (baseline_df['task'] == 'text')]
        if len(static_small) > 0:
            config = static_small.iloc[0]
            
            # Steady
            result = evaluate_workload('StaticSmall', config['lat_p95_ms'], config['accuracy'], deadline, 'steady', steady_arrivals)
            results.append(result)
            LOGGER.info(f"  StaticSmall/steady: hit_rate={result['deadline_hit_rate']:.3f}, queue_p95={result['p95_queue_length']:.1f}")
            
            # Bursty
            result = evaluate_workload('StaticSmall', config['lat_p95_ms'], config['accuracy'], deadline, 'bursty', bursty_arrivals)
            results.append(result)
            LOGGER.info(f"  StaticSmall/bursty: hit_rate={result['deadline_hit_rate']:.3f}, queue_p95={result['p95_queue_length']:.1f}")
        
        # StaticLarge
        static_large = baseline_df[(baseline_df['method'] == 'StaticLarge') & (baseline_df['deadline_ms'] == deadline) & (baseline_df['task'] == 'text')]
        if len(static_large) > 0:
            config = static_large.iloc[0]
            
            # Steady
            result = evaluate_workload('StaticLarge', config['lat_p95_ms'], config['accuracy'], deadline, 'steady', steady_arrivals)
            results.append(result)
            LOGGER.info(f"  StaticLarge/steady: hit_rate={result['deadline_hit_rate']:.3f}, queue_p95={result['p95_queue_length']:.1f}")
            
            # Bursty
            result = evaluate_workload('StaticLarge', config['lat_p95_ms'], config['accuracy'], deadline, 'bursty', bursty_arrivals)
            results.append(result)
            LOGGER.info(f"  StaticLarge/bursty: hit_rate={result['deadline_hit_rate']:.3f}, queue_p95={result['p95_queue_length']:.1f}")
        
        # CascadePlanner
        planner = planner_df[(planner_df['method'] == 'CascadePlanner') & (planner_df['deadline_ms'] == deadline) & (planner_df['task'] == 'text')]
        if len(planner) > 0:
            config = planner.iloc[0]
            
            # Steady
            result = evaluate_workload('CascadePlanner', config['lat_p95_ms'], config['accuracy'], deadline, 'steady', steady_arrivals)
            results.append(result)
            LOGGER.info(f"  CascadePlanner/steady: hit_rate={result['deadline_hit_rate']:.3f}, queue_p95={result['p95_queue_length']:.1f}")
            
            # Bursty
            result = evaluate_workload('CascadePlanner', config['lat_p95_ms'], config['accuracy'], deadline, 'bursty', bursty_arrivals)
            results.append(result)
            LOGGER.info(f"  CascadePlanner/bursty: hit_rate={result['deadline_hit_rate']:.3f}, queue_p95={result['p95_queue_length']:.1f}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("results/workload_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    LOGGER.info(f"\nWorkload sensitivity analysis complete. Results saved to {output_path}")
    LOGGER.info(f"Total evaluations: {len(results)}")
    
    # Summary
    LOGGER.info("\nSummary by workload type:")
    for workload_type in df['workload_type'].unique():
        subset = df[df['workload_type'] == workload_type]
        LOGGER.info(f"  {workload_type}:")
        LOGGER.info(f"    Mean hit rate: {subset['deadline_hit_rate'].mean():.3f}")
        LOGGER.info(f"    Mean queue p95: {subset['p95_queue_length'].mean():.1f}")


if __name__ == "__main__":
    main()

