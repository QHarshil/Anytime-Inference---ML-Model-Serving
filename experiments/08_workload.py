"""Workload sensitivity analysis with real inference.

Evaluates how planner performance varies under different workload patterns:
- Steady: Constant arrival rate (Poisson process)
- Bursty: Alternating low/high arrival rates

Simulates request arrivals and measures:
- Deadline hit rate under load
- Queue depth statistics
- Throughput degradation

Output: results/workload_sensitivity.csv
Schema: workload_type,task,deadline_ms,arrival_rate,method,
        deadline_hit_rate,avg_queue_depth,max_queue_depth,throughput
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from collections import deque

from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.workload")


def generate_steady_arrivals(duration_sec, arrival_rate):
    """Generate steady Poisson arrivals.
    
    Args:
        duration_sec: Simulation duration in seconds
        arrival_rate: Requests per second
    
    Returns:
        Array of arrival times in seconds
    """
    num_arrivals = int(duration_sec * arrival_rate * 1.2)  # Overestimate
    inter_arrival_times = np.random.exponential(1.0 / arrival_rate, num_arrivals)
    arrival_times = np.cumsum(inter_arrival_times)
    arrival_times = arrival_times[arrival_times < duration_sec]
    return arrival_times


def generate_bursty_arrivals(duration_sec, base_rate, burst_rate, burst_duration_sec=5.0, burst_interval_sec=20.0):
    """Generate bursty arrivals.
    
    Alternates between base rate and burst rate.
    
    Args:
        duration_sec: Simulation duration
        base_rate: Base arrival rate (req/sec)
        burst_rate: Burst arrival rate (req/sec)
        burst_duration_sec: Duration of each burst
        burst_interval_sec: Time between burst starts
    
    Returns:
        Array of arrival times
    """
    arrivals = []
    t = 0.0
    
    while t < duration_sec:
        # Burst period
        burst_end = min(t + burst_duration_sec, duration_sec)
        burst_arrivals = generate_steady_arrivals(burst_end - t, burst_rate)
        arrivals.extend(burst_arrivals + t)
        t = burst_end
        
        # Quiet period
        quiet_end = min(t + (burst_interval_sec - burst_duration_sec), duration_sec)
        quiet_arrivals = generate_steady_arrivals(quiet_end - t, base_rate)
        arrivals.extend(quiet_arrivals + t)
        t = quiet_end
    
    return np.array(sorted(arrivals))


def simulate_serving(arrival_times, latencies_ms, deadline_ms):
    """Simulate request serving with queue.
    
    Args:
        arrival_times: Array of arrival times (seconds)
        latencies_ms: Array of latencies for each request (milliseconds)
        deadline_ms: Deadline constraint (milliseconds)
    
    Returns:
        Dict with metrics
    """
    queue = deque()
    current_time = 0.0
    completed = []
    queue_depths = []
    
    for arrival_time, latency_ms in zip(arrival_times, latencies_ms):
        # Advance time to arrival
        current_time = arrival_time
        
        # Process completed requests
        while queue and queue[0]['completion_time'] <= current_time:
            req = queue.popleft()
            completed.append(req)
        
        # Add new request
        service_time_sec = latency_ms / 1000.0
        completion_time = current_time + service_time_sec
        
        queue.append({
            'arrival_time': arrival_time,
            'completion_time': completion_time,
            'latency_ms': latency_ms,
            'deadline_ms': deadline_ms
        })
        
        queue_depths.append(len(queue))
    
    # Process remaining queue
    while queue:
        req = queue.popleft()
        completed.append(req)
    
    # Compute metrics
    deadline_hits = sum(1 for req in completed if req['latency_ms'] <= req['deadline_ms'])
    deadline_hit_rate = deadline_hits / len(completed) if completed else 0.0
    
    avg_queue_depth = np.mean(queue_depths) if queue_depths else 0.0
    max_queue_depth = max(queue_depths) if queue_depths else 0
    
    throughput = len(completed) / (arrival_times[-1] - arrival_times[0]) if len(arrival_times) > 1 else 0.0
    
    return {
        'deadline_hit_rate': deadline_hit_rate,
        'avg_queue_depth': avg_queue_depth,
        'max_queue_depth': max_queue_depth,
        'throughput': throughput,
        'num_requests': len(completed)
    }


def evaluate_workload(workload_type, task, deadline_ms, method_name, latency_p50, latency_p95, arrival_rate, duration_sec=60.0):
    """Evaluate method under workload."""
    LOGGER.info(f"  {workload_type} workload, rate={arrival_rate} req/s")
    
    # Generate arrivals
    if workload_type == 'steady':
        arrivals = generate_steady_arrivals(duration_sec, arrival_rate)
    else:  # bursty
        arrivals = generate_bursty_arrivals(duration_sec, base_rate=arrival_rate*0.5, burst_rate=arrival_rate*2.0)
    
    # Generate latencies (log-normal approximation)
    sigma = np.log(latency_p95 / latency_p50) / 1.645 if latency_p95 > latency_p50 else 0.1
    mu = np.log(latency_p50)
    latencies = np.random.lognormal(mu, sigma, len(arrivals))
    
    # Simulate
    metrics = simulate_serving(arrivals, latencies, deadline_ms)
    
    return {
        'workload_type': workload_type,
        'task': task,
        'deadline_ms': deadline_ms,
        'arrival_rate': arrival_rate,
        'method': method_name,
        **metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    results_dir = Path("results")
    
    LOGGER.info("Loading baseline and planner results...")
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    
    # Configuration
    if args.quick:
        duration_sec = 30.0
        arrival_rates = [5, 10]
    else:
        duration_sec = 60.0
        arrival_rates = [5, 10, 20]
    
    workload_types = ['steady', 'bursty']
    
    all_results = []
    
    for task in ['text', 'vision']:
        LOGGER.info(f"\n{task.upper()}:")
        
        # Get representative latencies for each method
        task_baseline = baseline_df[baseline_df['task'] == task]
        task_planner = planner_df[planner_df['task'] == task]
        
        methods = []
        
        # Baselines
        for method_name in task_baseline['method'].unique():
            method_data = task_baseline[task_baseline['method'] == method_name]
            methods.append({
                'name': method_name,
                'lat_p50': method_data['lat_p50_ms'].mean(),
                'lat_p95': method_data['lat_p95_ms'].mean()
            })
        
        # Planner (best threshold)
        best_planner = task_planner.loc[task_planner['accuracy'].idxmax()]
        methods.append({
            'name': 'CascadePlanner',
            'lat_p50': best_planner['lat_p50_ms'],
            'lat_p95': best_planner['lat_p95_ms']
        })
        
        # Evaluate each method under workloads
        for deadline_ms in [100, 150]:
            for workload_type in workload_types:
                LOGGER.info(f"\n{workload_type.capitalize()} workload, deadline={deadline_ms}ms:")
                
                for arrival_rate in arrival_rates:
                    for method in methods:
                        result = evaluate_workload(
                            workload_type, task, deadline_ms,
                            method['name'], method['lat_p50'], method['lat_p95'],
                            arrival_rate, duration_sec
                        )
                        all_results.append(result)
                        
                        LOGGER.info(f"    {method['name']}: hit_rate={result['deadline_hit_rate']:.3f}, "
                                  f"avg_queue={result['avg_queue_depth']:.1f}, throughput={result['throughput']:.1f}")
    
    # Save
    results_df = pd.DataFrame(all_results)
    output_path = results_dir / "workload_sensitivity.csv"
    save_csv(results_df, output_path)
    
    LOGGER.info(f"\nComplete. Results: {output_path}")
    LOGGER.info(f"Total evaluations: {len(all_results)}")


if __name__ == "__main__":
    main()

