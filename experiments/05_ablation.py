"""Ablation studies with real inference measurements.

Analyzes the impact of individual design choices by comparing configurations
that differ in only one factor.

Implemented studies:
1. Model size: Compare small vs. large models (text and vision)
2. Quantization: Compare FP32 vs. FP16/INT8 quantization
3. Batch size: Compare batch sizes 1, 4, 8
4. Device: Compare CPU vs. GPU (NEW - real measurements)
5. Cascade: Compare single-model vs. two-stage cascade (NEW - real measurements)

Each ablation measures the impact on:
- Latency change (percentage)
- Accuracy change (percentage)
- Throughput change (percentage)

Output: results/ablation_results.csv
Schema: ablation_type,factor,baseline_value,comparison_value,
        latency_change_pct,accuracy_change_pct,throughput_change_pct,
        latency_baseline,latency_comparison,accuracy_baseline,accuracy_comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.evaluation.real_inference import RealInferenceEvaluator, CascadeInferenceEvaluator
from src.models.model_zoo import ModelZoo
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.ablation")


def ablation_model_size(latency_df, accuracy_df, task):
    """Ablation: Model size impact."""
    results = []
    
    task_lat = latency_df[latency_df['task'] == task]
    task_acc = accuracy_df[accuracy_df['task'] == task]
    
    # Group by model family
    if task == 'text':
        pairs = [
            ('distilbert', 'MiniLM'),  # Small vs medium
        ]
    else:  # vision
        pairs = [
            ('MobileNetV2', 'ResNet18'),  # Small vs medium
        ]
    
    for small_model, large_model in pairs:
        small_lat = task_lat[task_lat['model'].str.contains(small_model, case=False)]
        large_lat = task_lat[task_lat['model'].str.contains(large_model, case=False)]
        
        small_acc = task_acc[task_acc['model'].str.contains(small_model, case=False)]
        large_acc = task_acc[task_acc['model'].str.contains(large_model, case=False)]
        
        if len(small_lat) > 0 and len(large_lat) > 0:
            small_lat_val = small_lat['lat_p50_ms'].mean()
            large_lat_val = large_lat['lat_p50_ms'].mean()
            
            small_acc_val = small_acc['accuracy'].mean() if len(small_acc) > 0 else 0.0
            large_acc_val = large_acc['accuracy'].mean() if len(large_acc) > 0 else 0.0
            
            lat_change = ((large_lat_val - small_lat_val) / small_lat_val) * 100
            acc_change = ((large_acc_val - small_acc_val) / small_acc_val) * 100 if small_acc_val > 0 else 0.0
            throughput_change = -lat_change  # Inverse of latency
            
            results.append({
                'ablation_type': 'model_size',
                'task': task,
                'factor': f'{small_model}_vs_{large_model}',
                'baseline_value': small_model,
                'comparison_value': large_model,
                'latency_change_pct': lat_change,
                'accuracy_change_pct': acc_change,
                'throughput_change_pct': throughput_change,
                'latency_baseline': small_lat_val,
                'latency_comparison': large_lat_val,
                'accuracy_baseline': small_acc_val,
                'accuracy_comparison': large_acc_val
            })
    
    return results


def ablation_quantization(latency_df, accuracy_df, task):
    """Ablation: Quantization impact."""
    results = []
    
    task_lat = latency_df[latency_df['task'] == task]
    task_acc = accuracy_df[accuracy_df['task'] == task]
    
    # Compare FP32 vs quantized variants
    variants = task_lat['variant'].unique()
    
    if 'fp32' in variants:
        fp32_lat = task_lat[task_lat['variant'] == 'fp32']
        fp32_acc = task_acc[task_acc['variant'] == 'fp32']
        
        for quant_variant in ['fp16', 'int8']:
            if quant_variant in variants:
                quant_lat = task_lat[task_lat['variant'] == quant_variant]
                quant_acc = task_acc[task_acc['variant'] == quant_variant]
                
                if len(fp32_lat) > 0 and len(quant_lat) > 0:
                    fp32_lat_val = fp32_lat['lat_p50_ms'].mean()
                    quant_lat_val = quant_lat['lat_p50_ms'].mean()
                    
                    fp32_acc_val = fp32_acc['accuracy'].mean() if len(fp32_acc) > 0 else 0.0
                    quant_acc_val = quant_acc['accuracy'].mean() if len(quant_acc) > 0 else 0.0
                    
                    lat_change = ((quant_lat_val - fp32_lat_val) / fp32_lat_val) * 100
                    acc_change = ((quant_acc_val - fp32_acc_val) / fp32_acc_val) * 100 if fp32_acc_val > 0 else 0.0
                    throughput_change = -lat_change
                    
                    results.append({
                        'ablation_type': 'quantization',
                        'task': task,
                        'factor': f'fp32_vs_{quant_variant}',
                        'baseline_value': 'fp32',
                        'comparison_value': quant_variant,
                        'latency_change_pct': lat_change,
                        'accuracy_change_pct': acc_change,
                        'throughput_change_pct': throughput_change,
                        'latency_baseline': fp32_lat_val,
                        'latency_comparison': quant_lat_val,
                        'accuracy_baseline': fp32_acc_val,
                        'accuracy_comparison': quant_acc_val
                    })
    
    return results


def ablation_batch_size(latency_df, task):
    """Ablation: Batch size impact."""
    results = []
    
    task_lat = latency_df[latency_df['task'] == task]
    
    # Compare different batch sizes
    batch_sizes = sorted(task_lat['batch_size'].unique())
    
    if len(batch_sizes) >= 2:
        baseline_batch = batch_sizes[0]
        
        for comparison_batch in batch_sizes[1:]:
            baseline_lat = task_lat[task_lat['batch_size'] == baseline_batch]
            comparison_lat = task_lat[task_lat['batch_size'] == comparison_batch]
            
            if len(baseline_lat) > 0 and len(comparison_lat) > 0:
                baseline_val = baseline_lat['lat_p50_ms'].mean()
                comparison_val = comparison_lat['lat_p50_ms'].mean()
                
                lat_change = ((comparison_val - baseline_val) / baseline_val) * 100
                
                # Throughput scales with batch size (ideally)
                throughput_baseline = baseline_batch / baseline_val
                throughput_comparison = comparison_batch / comparison_val
                throughput_change = ((throughput_comparison - throughput_baseline) / throughput_baseline) * 100
                
                results.append({
                    'ablation_type': 'batch_size',
                    'task': task,
                    'factor': f'batch{baseline_batch}_vs_batch{comparison_batch}',
                    'baseline_value': str(baseline_batch),
                    'comparison_value': str(comparison_batch),
                    'latency_change_pct': lat_change,
                    'accuracy_change_pct': 0.0,  # Batch size doesn't affect accuracy
                    'throughput_change_pct': throughput_change,
                    'latency_baseline': baseline_val,
                    'latency_comparison': comparison_val,
                    'accuracy_baseline': 0.0,
                    'accuracy_comparison': 0.0
                })
    
    return results


def ablation_device(latency_df, accuracy_df, task):
    """Ablation: Device impact (CPU vs GPU) with real measurements."""
    results = []
    
    task_lat = latency_df[latency_df['task'] == task]
    task_acc = accuracy_df[accuracy_df['task'] == task]
    
    devices = task_lat['device'].unique()
    
    if 'cpu' in devices and 'cuda' in devices:
        cpu_lat = task_lat[task_lat['device'] == 'cpu']
        gpu_lat = task_lat[task_lat['device'] == 'cuda']
        
        cpu_acc = task_acc[task_acc['device'] == 'cpu']
        gpu_acc = task_acc[task_acc['device'] == 'cuda']
        
        if len(cpu_lat) > 0 and len(gpu_lat) > 0:
            cpu_lat_val = cpu_lat['lat_p50_ms'].mean()
            gpu_lat_val = gpu_lat['lat_p50_ms'].mean()
            
            cpu_acc_val = cpu_acc['accuracy'].mean() if len(cpu_acc) > 0 else 0.0
            gpu_acc_val = gpu_acc['accuracy'].mean() if len(gpu_acc) > 0 else 0.0
            
            lat_change = ((gpu_lat_val - cpu_lat_val) / cpu_lat_val) * 100
            acc_change = ((gpu_acc_val - cpu_acc_val) / cpu_acc_val) * 100 if cpu_acc_val > 0 else 0.0
            throughput_change = -lat_change
            
            results.append({
                'ablation_type': 'device',
                'task': task,
                'factor': 'cpu_vs_gpu',
                'baseline_value': 'cpu',
                'comparison_value': 'cuda',
                'latency_change_pct': lat_change,
                'accuracy_change_pct': acc_change,
                'throughput_change_pct': throughput_change,
                'latency_baseline': cpu_lat_val,
                'latency_comparison': gpu_lat_val,
                'accuracy_baseline': cpu_acc_val,
                'accuracy_comparison': gpu_acc_val
            })
    
    return results


def ablation_cascade(planner_results_df, baseline_results_df, task):
    """Ablation: Cascade vs single-model with real measurements."""
    results = []
    
    # Compare cascade (from planner) vs StaticLarge (single model)
    cascade = planner_results_df[planner_results_df['task'] == task]
    single = baseline_results_df[
        (baseline_results_df['task'] == task) &
        (baseline_results_df['method'] == 'StaticLarge')
    ]
    
    if len(cascade) > 0 and len(single) > 0:
        # Use best cascade threshold
        best_cascade = cascade.loc[cascade['accuracy'].idxmax()]
        
        cascade_lat = best_cascade['lat_p95_ms']
        cascade_acc = best_cascade['accuracy']
        
        single_lat = single['lat_p95_ms'].mean()
        single_acc = single['accuracy'].mean()
        
        lat_change = ((cascade_lat - single_lat) / single_lat) * 100
        acc_change = ((cascade_acc - single_acc) / single_acc) * 100 if single_acc > 0 else 0.0
        throughput_change = -lat_change
        
        results.append({
            'ablation_type': 'cascade',
            'task': task,
            'factor': 'single_vs_cascade',
            'baseline_value': 'single_model',
            'comparison_value': f'cascade_t{best_cascade["threshold"]:.2f}',
            'latency_change_pct': lat_change,
            'accuracy_change_pct': acc_change,
            'throughput_change_pct': throughput_change,
            'latency_baseline': single_lat,
            'latency_comparison': cascade_lat,
            'accuracy_baseline': single_acc,
            'accuracy_comparison': cascade_acc
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    results_dir = Path("results")
    
    LOGGER.info("Loading profiles and results...")
    latency_df = pd.read_csv(results_dir / "latency_profiles.csv")
    accuracy_df = pd.read_csv(results_dir / "accuracy_profiles.csv")
    
    # Load planner and baseline results for cascade ablation
    planner_results_df = pd.read_csv(results_dir / "planner_results.csv") if (results_dir / "planner_results.csv").exists() else pd.DataFrame()
    baseline_results_df = pd.read_csv(results_dir / "baseline_results.csv") if (results_dir / "baseline_results.csv").exists() else pd.DataFrame()
    
    all_results = []
    
    for task in ['text', 'vision']:
        LOGGER.info(f"\nAblations for {task.upper()}:")
        
        # 1. Model size
        LOGGER.info("  1. Model size...")
        results = ablation_model_size(latency_df, accuracy_df, task)
        all_results.extend(results)
        for r in results:
            LOGGER.info(f"    {r['factor']}: latency {r['latency_change_pct']:+.1f}%, accuracy {r['accuracy_change_pct']:+.1f}%")
        
        # 2. Quantization
        LOGGER.info("  2. Quantization...")
        results = ablation_quantization(latency_df, accuracy_df, task)
        all_results.extend(results)
        for r in results:
            LOGGER.info(f"    {r['factor']}: latency {r['latency_change_pct']:+.1f}%, accuracy {r['accuracy_change_pct']:+.1f}%")
        
        # 3. Batch size
        LOGGER.info("  3. Batch size...")
        results = ablation_batch_size(latency_df, task)
        all_results.extend(results)
        for r in results:
            LOGGER.info(f"    {r['factor']}: latency {r['latency_change_pct']:+.1f}%, throughput {r['throughput_change_pct']:+.1f}%")
        
        # 4. Device (NEW)
        LOGGER.info("  4. Device (CPU vs GPU)...")
        results = ablation_device(latency_df, accuracy_df, task)
        all_results.extend(results)
        for r in results:
            LOGGER.info(f"    {r['factor']}: latency {r['latency_change_pct']:+.1f}%, accuracy {r['accuracy_change_pct']:+.1f}%")
        
        # 5. Cascade (NEW)
        if len(planner_results_df) > 0 and len(baseline_results_df) > 0:
            LOGGER.info("  5. Cascade (single vs two-stage)...")
            results = ablation_cascade(planner_results_df, baseline_results_df, task)
            all_results.extend(results)
            for r in results:
                LOGGER.info(f"    {r['factor']}: latency {r['latency_change_pct']:+.1f}%, accuracy {r['accuracy_change_pct']:+.1f}%")
    
    # Save
    results_df = pd.DataFrame(all_results)
    output_path = results_dir / "ablation_results.csv"
    save_csv(results_df, output_path)
    
    LOGGER.info(f"\nComplete. Results: {output_path}")
    LOGGER.info(f"Total ablations: {len(all_results)}")


if __name__ == "__main__":
    main()

