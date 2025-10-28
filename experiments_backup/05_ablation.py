"""Ablation studies.

Analyzes the impact of individual design choices by comparing configurations
that differ in only one factor.

Implemented studies:
1. Model size: Compare small vs. large models (text and vision)
2. Quantization: Compare FP32 vs. FP16/INT8 quantization
3. Batch size: Compare batch sizes 1, 4, 8 (holding model/device constant)

Each ablation measures the impact on:
- Latency change (percentage)
- Accuracy change (percentage)
- Throughput change (percentage)

Output: results/ablation_results.csv
Schema: ablation_type,factor,baseline_value,comparison_value,latency_change_pct,accuracy_change_pct,throughput_change_pct
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.ablation")


def load_profiles():
    """Load latency and accuracy profiles."""
    LOGGER.info("Loading profiling results...")
    
    latency_path = Path("results/latency_profiles.csv")
    accuracy_path = Path("results/accuracy_profiles.csv")
    
    if not latency_path.exists():
        raise FileNotFoundError("Latency profiles not found. Run 01_profile_latency.py first.")
    if not accuracy_path.exists():
        raise FileNotFoundError("Accuracy profiles not found. Run 02_profile_accuracy.py first.")
    
    latency_df = pd.read_csv(latency_path)
    accuracy_df = pd.read_csv(accuracy_path)
    
    LOGGER.info(f"Loaded {len(latency_df)} latency profiles")
    LOGGER.info(f"Loaded {len(accuracy_df)} accuracy profiles")
    
    return latency_df, accuracy_df


def compute_change_pct(baseline, comparison):
    """Compute percentage change from baseline to comparison."""
    if baseline == 0:
        return 0.0
    return ((comparison - baseline) / baseline) * 100


def ablation_model_size(latency_df, accuracy_df):
    """Ablation study: Impact of model size."""
    LOGGER.info("\nAblation: Model Size")
    
    results = []
    
    # Text models: minilm (small) vs. distilbert (large)
    for variant in ['fp32', 'int8']:
        for device in ['cpu']:
            # Get small model (minilm)
            small_lat = latency_df[
                (latency_df['model'] == 'minilm') &
                (latency_df['variant'] == variant) &
                (latency_df['device'] == device) &
                (latency_df['batch_size'] == 1)
            ]
            
            small_acc = accuracy_df[
                (accuracy_df['model'] == 'minilm') &
                (accuracy_df['variant'] == variant) &
                (accuracy_df['device'] == device) &
                (accuracy_df['exit_policy'] == 'none')
            ]
            
            # Get large model (distilbert)
            large_lat = latency_df[
                (latency_df['model'] == 'distilbert') &
                (latency_df['variant'] == variant) &
                (latency_df['device'] == device) &
                (latency_df['batch_size'] == 1)
            ]
            
            large_acc = accuracy_df[
                (accuracy_df['model'] == 'distilbert') &
                (accuracy_df['variant'] == variant) &
                (accuracy_df['device'] == device) &
                (accuracy_df['exit_policy'] == 'none')
            ]
            
            if len(small_lat) > 0 and len(large_lat) > 0 and len(small_acc) > 0 and len(large_acc) > 0:
                small_lat_val = small_lat.iloc[0]['lat_p50_ms']
                large_lat_val = large_lat.iloc[0]['lat_p50_ms']
                small_acc_val = small_acc.iloc[0]['accuracy']
                large_acc_val = large_acc.iloc[0]['accuracy']
                small_thr_val = small_lat.iloc[0]['throughput_items_per_sec']
                large_thr_val = large_lat.iloc[0]['throughput_items_per_sec']
                
                lat_change = compute_change_pct(small_lat_val, large_lat_val)
                acc_change = compute_change_pct(small_acc_val, large_acc_val)
                thr_change = compute_change_pct(small_thr_val, large_thr_val)
                
                results.append({
                    'ablation_type': 'model_size',
                    'task': 'text',
                    'factor': f'{variant}_{device}',
                    'baseline': 'minilm',
                    'comparison': 'distilbert',
                    'baseline_latency_ms': small_lat_val,
                    'comparison_latency_ms': large_lat_val,
                    'latency_change_pct': lat_change,
                    'baseline_accuracy': small_acc_val,
                    'comparison_accuracy': large_acc_val,
                    'accuracy_change_pct': acc_change,
                    'throughput_change_pct': thr_change
                })
                
                LOGGER.info(f"  {variant}/{device}: latency +{lat_change:.1f}%, accuracy +{acc_change:.1f}%")
    
    # Vision models: mobilenetv2 (small) vs. resnet18 (large)
    for variant in ['fp32', 'int8']:
        for device in ['cpu']:
            small_lat = latency_df[
                (latency_df['model'] == 'mobilenetv2') &
                (latency_df['variant'] == variant) &
                (latency_df['device'] == device) &
                (latency_df['batch_size'] == 1)
            ]
            
            small_acc = accuracy_df[
                (accuracy_df['model'] == 'mobilenetv2') &
                (accuracy_df['variant'] == variant) &
                (accuracy_df['device'] == device) &
                (accuracy_df['exit_policy'] == 'none')
            ]
            
            large_lat = latency_df[
                (latency_df['model'] == 'resnet18') &
                (latency_df['variant'] == variant) &
                (latency_df['device'] == device) &
                (latency_df['batch_size'] == 1)
            ]
            
            large_acc = accuracy_df[
                (accuracy_df['model'] == 'resnet18') &
                (accuracy_df['variant'] == variant) &
                (accuracy_df['device'] == device) &
                (accuracy_df['exit_policy'] == 'none')
            ]
            
            if len(small_lat) > 0 and len(large_lat) > 0 and len(small_acc) > 0 and len(large_acc) > 0:
                small_lat_val = small_lat.iloc[0]['lat_p50_ms']
                large_lat_val = large_lat.iloc[0]['lat_p50_ms']
                small_acc_val = small_acc.iloc[0]['accuracy']
                large_acc_val = large_acc.iloc[0]['accuracy']
                small_thr_val = small_lat.iloc[0]['throughput_items_per_sec']
                large_thr_val = large_lat.iloc[0]['throughput_items_per_sec']
                
                lat_change = compute_change_pct(small_lat_val, large_lat_val)
                acc_change = compute_change_pct(small_acc_val, large_acc_val)
                thr_change = compute_change_pct(small_thr_val, large_thr_val)
                
                results.append({
                    'ablation_type': 'model_size',
                    'task': 'vision',
                    'factor': f'{variant}_{device}',
                    'baseline': 'mobilenetv2',
                    'comparison': 'resnet18',
                    'baseline_latency_ms': small_lat_val,
                    'comparison_latency_ms': large_lat_val,
                    'latency_change_pct': lat_change,
                    'baseline_accuracy': small_acc_val,
                    'comparison_accuracy': large_acc_val,
                    'accuracy_change_pct': acc_change,
                    'throughput_change_pct': thr_change
                })
                
                LOGGER.info(f"  {variant}/{device}: latency +{lat_change:.1f}%, accuracy +{acc_change:.1f}%")
    
    return results


def ablation_quantization(latency_df, accuracy_df):
    """Ablation study: Impact of quantization."""
    LOGGER.info("\nAblation: Quantization")
    
    results = []
    
    # Compare FP32 vs. INT8 on CPU for each model
    for model in ['minilm', 'distilbert', 'mobilenetv2', 'resnet18']:
        task = 'text' if model in ['minilm', 'distilbert'] else 'vision'
        
        fp32_lat = latency_df[
            (latency_df['model'] == model) &
            (latency_df['variant'] == 'fp32') &
            (latency_df['device'] == 'cpu') &
            (latency_df['batch_size'] == 1)
        ]
        
        int8_lat = latency_df[
            (latency_df['model'] == model) &
            (latency_df['variant'] == 'int8') &
            (latency_df['device'] == 'cpu') &
            (latency_df['batch_size'] == 1)
        ]
        
        fp32_acc = accuracy_df[
            (accuracy_df['model'] == model) &
            (accuracy_df['variant'] == 'fp32') &
            (accuracy_df['device'] == 'cpu') &
            (accuracy_df['exit_policy'] == 'none')
        ]
        
        int8_acc = accuracy_df[
            (accuracy_df['model'] == model) &
            (accuracy_df['variant'] == 'int8') &
            (accuracy_df['device'] == 'cpu') &
            (accuracy_df['exit_policy'] == 'none')
        ]
        
        if len(fp32_lat) > 0 and len(int8_lat) > 0 and len(fp32_acc) > 0 and len(int8_acc) > 0:
            fp32_lat_val = fp32_lat.iloc[0]['lat_p50_ms']
            int8_lat_val = int8_lat.iloc[0]['lat_p50_ms']
            fp32_acc_val = fp32_acc.iloc[0]['accuracy']
            int8_acc_val = int8_acc.iloc[0]['accuracy']
            fp32_thr_val = fp32_lat.iloc[0]['throughput_items_per_sec']
            int8_thr_val = int8_lat.iloc[0]['throughput_items_per_sec']
            
            lat_change = compute_change_pct(fp32_lat_val, int8_lat_val)
            acc_change = compute_change_pct(fp32_acc_val, int8_acc_val)
            thr_change = compute_change_pct(fp32_thr_val, int8_thr_val)
            
            results.append({
                'ablation_type': 'quantization',
                'task': task,
                'factor': model,
                'baseline': 'fp32',
                'comparison': 'int8',
                'baseline_latency_ms': fp32_lat_val,
                'comparison_latency_ms': int8_lat_val,
                'latency_change_pct': lat_change,
                'baseline_accuracy': fp32_acc_val,
                'comparison_accuracy': int8_acc_val,
                'accuracy_change_pct': acc_change,
                'throughput_change_pct': thr_change
            })
            
            LOGGER.info(f"  {model}: latency {lat_change:.1f}%, accuracy {acc_change:.1f}%")
    
    return results


def ablation_batch_size(latency_df):
    """Ablation study: Impact of batch size."""
    LOGGER.info("\nAblation: Batch Size")
    
    results = []
    
    # Compare batch sizes for each model
    for model in ['minilm', 'distilbert']:
        task = 'text'
        
        for variant in ['fp32', 'int8']:
            for device in ['cpu']:
                batch_1 = latency_df[
                    (latency_df['model'] == model) &
                    (latency_df['variant'] == variant) &
                    (latency_df['device'] == device) &
                    (latency_df['batch_size'] == 1)
                ]
                
                batch_8 = latency_df[
                    (latency_df['model'] == model) &
                    (latency_df['variant'] == variant) &
                    (latency_df['device'] == device) &
                    (latency_df['batch_size'] == 8)
                ]
                
                if len(batch_1) > 0 and len(batch_8) > 0:
                    b1_lat = batch_1.iloc[0]['lat_p50_ms']
                    b8_lat = batch_8.iloc[0]['lat_p50_ms']
                    b1_thr = batch_1.iloc[0]['throughput_items_per_sec']
                    b8_thr = batch_8.iloc[0]['throughput_items_per_sec']
                    
                    lat_change = compute_change_pct(b1_lat, b8_lat)
                    thr_change = compute_change_pct(b1_thr, b8_thr)
                    
                    results.append({
                        'ablation_type': 'batch_size',
                        'task': task,
                        'factor': f'{model}_{variant}_{device}',
                        'baseline': 'batch_1',
                        'comparison': 'batch_8',
                        'baseline_latency_ms': b1_lat,
                        'comparison_latency_ms': b8_lat,
                        'latency_change_pct': lat_change,
                        'baseline_accuracy': None,
                        'comparison_accuracy': None,
                        'accuracy_change_pct': 0.0,
                        'throughput_change_pct': thr_change
                    })
                    
                    LOGGER.info(f"  {model}/{variant}/{device}: latency +{lat_change:.1f}%, throughput +{thr_change:.1f}%")
    
    return results


def main():
    LOGGER.info("Starting ablation studies...")
    
    # Load profiles
    latency_df, accuracy_df = load_profiles()
    
    results = []
    
    # Run ablation studies
    results.extend(ablation_model_size(latency_df, accuracy_df))
    results.extend(ablation_quantization(latency_df, accuracy_df))
    results.extend(ablation_batch_size(latency_df))
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("results/ablation_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(df, output_path)
    
    LOGGER.info(f"\nAblation studies complete. Results saved to {output_path}")
    LOGGER.info(f"Total ablations: {len(results)}")
    
    # Summary
    LOGGER.info("\nSummary by ablation type:")
    for ablation_type in df['ablation_type'].unique():
        subset = df[df['ablation_type'] == ablation_type]
        LOGGER.info(f"  {ablation_type}: {len(subset)} comparisons")
        LOGGER.info(f"    Mean latency change: {subset['latency_change_pct'].mean():.1f}%")
        if 'accuracy_change_pct' in subset.columns:
            acc_mean = subset['accuracy_change_pct'].mean()
            if not pd.isna(acc_mean):
                LOGGER.info(f"    Mean accuracy change: {acc_mean:.1f}%")


if __name__ == "__main__":
    main()

