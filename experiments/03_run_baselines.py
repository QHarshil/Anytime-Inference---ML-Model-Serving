"""Evaluate baseline planners with real inference measurements.

Baselines:
1. StaticSmall: Always use fastest model (lowest latency)
2. StaticLarge: Always use most accurate model (highest accuracy)
3. ThroughputAutotuner: Select model with best throughput under deadline
4. INFaaS-style: Variant-aware selection (adapted from Romero et al., ATC'20)

Evaluation:
- Real batched inference runs (not simulated)
- Multiple independent runs with different seeds
- Proper variance reporting and confidence intervals

Output: results/baseline_results.csv
Schema: method,task,deadline_ms,seed,deadline_hit_rate,accuracy,accuracy_std,lat_p95_ms,lat_std_ms,num_runs,num_samples
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.evaluation.real_inference import RealInferenceEvaluator
from src.models.model_zoo import ModelZoo
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.baselines")


def load_profiles(results_dir: Path) -> tuple:
    """Load latency and accuracy profiles."""
    latency_df = pd.read_csv(results_dir / "latency_profiles.csv")
    accuracy_df = pd.read_csv(results_dir / "accuracy_profiles.csv")
    return latency_df, accuracy_df


def select_static_small(latency_df: pd.DataFrame, task: str) -> dict:
    """Select fastest model for task."""
    task_df = latency_df[latency_df['task'] == task]
    fastest = task_df.loc[task_df['lat_p50_ms'].idxmin()]
    return {
        'model': fastest['model'],
        'variant': fastest['variant'],
        'device': fastest['device'],
        'batch_size': int(fastest['batch_size'])
    }


def select_static_large(accuracy_df: pd.DataFrame, task: str) -> dict:
    """Select most accurate model for task."""
    task_df = accuracy_df[accuracy_df['task'] == task]
    best = task_df.loc[task_df['accuracy'].idxmax()]
    return {
        'model': best['model'],
        'variant': best['variant'],
        'device': best['device'],
        'batch_size': 1
    }


def select_throughput_autotuner(
    latency_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    task: str,
    deadline_ms: float
) -> dict:
    """Select model with best throughput under deadline."""
    task_lat = latency_df[latency_df['task'] == task].copy()
    
    # Filter configs that meet deadline
    feasible = task_lat[task_lat['lat_p95_ms'] < deadline_ms]
    
    if len(feasible) == 0:
        return select_static_small(latency_df, task)
    
    # Select highest throughput
    feasible['throughput'] = feasible.get('items_per_sec', feasible.get('throughput_items_per_sec', 1000.0 / feasible['lat_p50_ms']))
    best = feasible.loc[feasible['throughput'].idxmax()]
    
    return {
        'model': best['model'],
        'variant': best['variant'],
        'device': best['device'],
        'batch_size': int(best['batch_size'])
    }


def select_infaas_style(
    latency_df: pd.DataFrame,
    accuracy_df: pd.DataFrame,
    task: str,
    deadline_ms: float,
    accuracy_target: float = 0.85
) -> dict:
    """INFaaS-style variant-aware selection (adapted from Romero et al., ATC'20)."""
    task_lat = latency_df[latency_df['task'] == task].copy()
    task_acc = accuracy_df[accuracy_df['task'] == task].copy()
    
    # Merge
    merged = task_lat.merge(
        task_acc[['model', 'variant', 'accuracy']],
        on=['model', 'variant'],
        how='inner'
    )
    
    # Filter by constraints
    feasible = merged[
        (merged['lat_p95_ms'] < deadline_ms) &
        (merged['accuracy'] >= accuracy_target)
    ]
    
    if len(feasible) == 0:
        feasible = merged[merged['lat_p95_ms'] < deadline_ms]
        if len(feasible) == 0:
            return select_static_small(latency_df, task)
    
    # Select cheapest (lowest latency)
    best = feasible.loc[feasible['lat_p50_ms'].idxmin()]
    
    return {
        'model': best['model'],
        'variant': best['variant'],
        'device': best['device'],
        'batch_size': int(best['batch_size'])
    }


def evaluate_baseline(
    method_name: str,
    config: dict,
    task: str,
    deadline_ms: float,
    evaluator: RealInferenceEvaluator,
    model_zoo: ModelZoo,
    inputs: list,
    labels: list,
    num_runs: int,
    num_samples: int,
    seed: int
) -> dict:
    """Evaluate baseline with real inference."""
    LOGGER.info(f"  {method_name}: {config['model']}/{config['variant']} (seed={seed})")
    
    # Load model
    model, tokenizer_or_transform = model_zoo.load_model(
        task=task,
        model_name=config['model'],
        variant=config['variant'],
        device=config['device']
    )
    
    # Run real inference
    result = evaluator.evaluate_config(
        model=model,
        tokenizer_or_transform=tokenizer_or_transform,
        inputs=inputs,
        labels=labels,
        config=config,
        task=task,
        num_runs=num_runs,
        num_samples=num_samples,
        seed=seed,
        use_cache=True
    )
    
    sample_latencies = np.array(result.latency_samples_ms, dtype=float)
    if sample_latencies.size > 0:
        deadline_hit_rate = float(np.mean(sample_latencies <= deadline_ms))
    else:
        deadline_hit_rate = 1.0 if result.latency_p95 < deadline_ms else 0.0
    
    return {
        'method': method_name,
        'task': task,
        'deadline_ms': deadline_ms,
        'seed': seed,
        'model': config['model'],
        'variant': config['variant'],
        'device': config['device'],
        'batch_size': config['batch_size'],
        'config_selected': f"{config['model']}_{config['variant']}_{config['device']}_b{config['batch_size']}",
        'deadline_hit_rate': deadline_hit_rate,
        'accuracy': result.accuracy_mean,
        'accuracy_std': result.accuracy_std,
        'lat_p50_ms': result.latency_p50,
        'lat_p95_ms': result.latency_p95,
        'lat_std_ms': result.latency_std,
        'num_runs': result.num_runs,
        'num_samples': result.num_samples_per_run
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Configuration
    if args.quick:
        num_runs = 3
        num_samples = 50
        seeds = [42, 43, 44]
        deadlines = {'text': [100], 'vision': [150]}
    else:
        num_runs = 5
        num_samples = 100
        seeds = [42, 43, 44, 45, 46]
        deadlines = {'text': [50, 100, 150], 'vision': [100, 150, 200]}
    
    LOGGER.info("Loading profiles...")
    latency_df, accuracy_df = load_profiles(results_dir)
    
    LOGGER.info("Initializing...")
    evaluator = RealInferenceEvaluator()
    model_zoo = ModelZoo()
    
    # Load datasets
    LOGGER.info("Loading datasets...")
    from datasets import load_dataset
    from torchvision import datasets, transforms
    
    sst2 = load_dataset("glue", "sst2", split="validation")
    text_inputs = sst2['sentence']
    text_labels = sst2['label']
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    cifar10 = datasets.CIFAR10(root='./data', train=False, download=True)
    vision_inputs = [img for img, _ in cifar10]
    vision_labels = [label for _, label in cifar10]
    
    # Evaluate
    all_results = []
    
    for task in ['text', 'vision']:
        inputs = text_inputs if task == 'text' else vision_inputs
        labels = text_labels if task == 'text' else vision_labels
        
        for deadline_ms in deadlines[task]:
            LOGGER.info(f"\n{task.upper()} - Deadline: {deadline_ms}ms")
            
            # Select configs
            config_small = select_static_small(latency_df, task)
            config_large = select_static_large(accuracy_df, task)
            config_throughput = select_throughput_autotuner(latency_df, accuracy_df, task, deadline_ms)
            config_infaas = select_infaas_style(latency_df, accuracy_df, task, deadline_ms)
            
            baselines = [
                ('StaticSmall', config_small),
                ('StaticLarge', config_large),
                ('ThroughputAutotuner', config_throughput),
                ('INFaaS-style', config_infaas)
            ]
            
            for seed in seeds:
                for method_name, config in baselines:
                    result = evaluate_baseline(
                        method_name, config, task, deadline_ms,
                        evaluator, model_zoo, inputs, labels,
                        num_runs, num_samples, seed
                    )
                    all_results.append(result)
    
    # Save
    results_df = pd.DataFrame(all_results)
    output_path = results_dir / "baseline_results.csv"
    save_csv(results_df, output_path)
    
    LOGGER.info(f"\nComplete. Results: {output_path}")
    LOGGER.info(f"Total evaluations: {len(all_results)}")
    
    # Summary
    summary = results_df.groupby(['method', 'task']).agg({
        'deadline_hit_rate': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(3)
    LOGGER.info(f"\nSummary:\n{summary}")


if __name__ == "__main__":
    main()
