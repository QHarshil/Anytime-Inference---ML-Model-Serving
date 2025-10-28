"""Evaluate cascade planner with real end-to-end inference.

Implements two-stage cascade:
- Stage 1: Fast model with confidence threshold
- Stage 2: Accurate model (if stage 1 confidence < threshold)

Measures real latencies for both stages and end-to-end performance.

Output: results/planner_results.csv
Schema: task,deadline_ms,threshold,seed,deadline_hit_rate,accuracy,accuracy_std,
        coverage,lat_p95_ms,lat_std_ms,num_runs,num_samples
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.evaluation.real_inference import CascadeInferenceEvaluator
from src.models.model_zoo import ModelZoo
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.planner")


def load_profiles(results_dir: Path) -> tuple:
    """Load latency and accuracy profiles."""
    latency_df = pd.read_csv(results_dir / "latency_profiles.csv")
    accuracy_df = pd.read_csv(results_dir / "accuracy_profiles.csv")
    return latency_df, accuracy_df


def select_cascade_models(latency_df: pd.DataFrame, accuracy_df: pd.DataFrame, task: str) -> tuple:
    """Select small and large models for cascade.
    
    Returns:
        Tuple of (small_config, large_config)
    """
    task_lat = latency_df[latency_df['task'] == task]
    task_acc = accuracy_df[accuracy_df['task'] == task]
    
    # Small model: fastest
    small = task_lat.loc[task_lat['lat_p50_ms'].idxmin()]
    
    # Large model: most accurate
    large = task_acc.loc[task_acc['accuracy'].idxmax()]
    
    small_config = {
        'model': small['model'],
        'variant': small['variant'],
        'device': small['device'],
        'batch_size': int(small['batch_size'])
    }

    large_config = {
        'model': large['model'],
        'variant': large['variant'],
        'device': large['device'],
        'batch_size': int(large.get('batch_size', 1))
    }
    
    return small_config, large_config


def evaluate_cascade(
    small_config: dict,
    large_config: dict,
    threshold: float,
    task: str,
    deadline_ms: float,
    evaluator: CascadeInferenceEvaluator,
    model_zoo: ModelZoo,
    inputs: list,
    labels: list,
    num_runs: int,
    num_samples: int,
    seed: int
) -> dict:
    """Evaluate cascade configuration with real end-to-end inference."""
    LOGGER.info(f"  Threshold {threshold:.2f} (seed={seed})")
    
    # Load models
    small_model, tokenizer_or_transform = model_zoo.load_model(
        task=task,
        model_name=small_config['model'],
        variant=small_config['variant'],
        device=small_config['device']
    )

    large_model, _ = model_zoo.load_model(
        task=task,
        model_name=large_config['model'],
        variant=large_config['variant'],
        device=large_config['device']
    )
    
    # Run real cascade inference
    result = evaluator.evaluate_cascade(
        small_model=small_model,
        large_model=large_model,
        tokenizer_or_transform=tokenizer_or_transform,
        inputs=inputs,
        labels=labels,
        threshold=threshold,
        task=task,
        device=small_config['device'],
        num_runs=num_runs,
        num_samples=num_samples,
        seed=seed,
        use_cache=True
    )
    
    # Deadline hit rate
    deadline_hit_rate = 1.0 if result['latency_p95'] < deadline_ms else 0.0
    
    return {
        'task': task,
        'deadline_ms': deadline_ms,
        'threshold': threshold,
        'seed': seed,
        'small_model': small_config['model'],
        'small_variant': small_config['variant'],
        'small_device': small_config['device'],
        'large_model': large_config['model'],
        'large_variant': large_config['variant'],
        'large_device': large_config['device'],
        'deadline_hit_rate': deadline_hit_rate,
        'accuracy': result['accuracy_mean'],
        'accuracy_std': result['accuracy_std'],
        'coverage': result['coverage_mean'],
        'coverage_std': result['coverage_std'],
        'lat_p50_ms': result['latency_p50'],
        'lat_p95_ms': result['latency_p95'],
        'lat_std_ms': result['latency_std'],
        'stage1_lat_mean_ms': result['latency_mean'],  # Approximation
        'num_runs': result['num_runs'],
        'num_samples': result['num_samples_per_run'],
        'cache_key': result['cache_key'],
        'latency_samples_count': len(result['latency_samples_ms'])
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
        thresholds = [0.8, 0.9]
    else:
        num_runs = 5
        num_samples = 100
        seeds = [42, 43, 44, 45, 46]
        deadlines = {'text': [50, 100, 150], 'vision': [100, 150, 200]}
        thresholds = [0.7, 0.8, 0.9, 0.95]
    
    LOGGER.info("Loading profiles...")
    latency_df, accuracy_df = load_profiles(results_dir)
    
    LOGGER.info("Initializing...")
    evaluator = CascadeInferenceEvaluator()
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
        
        # Select cascade models
        small_config, large_config = select_cascade_models(latency_df, accuracy_df, task)
        LOGGER.info(f"\n{task.upper()} Cascade:")
        LOGGER.info(f"  Small: {small_config['model']}/{small_config['variant']}")
        LOGGER.info(f"  Large: {large_config['model']}/{large_config['variant']}")
        
        for deadline_ms in deadlines[task]:
            LOGGER.info(f"\nDeadline: {deadline_ms}ms")
            
            for threshold in thresholds:
                for seed in seeds:
                    result = evaluate_cascade(
                        small_config, large_config, threshold,
                        task, deadline_ms,
                        evaluator, model_zoo, inputs, labels,
                        num_runs, num_samples, seed
                    )
                    all_results.append(result)
    
    # Save
    results_df = pd.DataFrame(all_results)
    output_path = results_dir / "planner_results.csv"
    save_csv(results_df, output_path)
    
    LOGGER.info(f"\nComplete. Results: {output_path}")
    LOGGER.info(f"Total evaluations: {len(all_results)}")
    
    # Summary
    summary = results_df.groupby(['task', 'threshold']).agg({
        'deadline_hit_rate': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'coverage': ['mean', 'std']
    }).round(3)
    LOGGER.info(f"\nSummary:\n{summary}")


if __name__ == "__main__":
    main()
