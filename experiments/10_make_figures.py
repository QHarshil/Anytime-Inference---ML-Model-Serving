"""Generate all figures for the paper.

Creates publication-quality visualizations:
1. Pareto frontier (latency vs accuracy)
2. Baseline comparison (hit rate and accuracy bars)
3. Cascade threshold sensitivity
4. Ablation studies (bar charts)
5. Workload sensitivity (line plots)
6. Statistical significance (effect sizes)

Outputs (in results/figures/):
- 01_pareto_frontier.png
- 02_baseline_comparison.png
- 03_cascade_threshold.png
- 04_ablation_studies.png
- 05_workload_sensitivity.png
- 06_statistical_significance.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

LOGGER = get_logger("experiments.figures")

# Set style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def plot_pareto_frontier(pareto_df, baseline_df, planner_df, output_dir):
    """Plot Pareto frontier for each task."""
    LOGGER.info("Generating Pareto frontier plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, task in enumerate(['text', 'vision']):
        ax = axes[idx]
        
        # Baseline points
        task_baseline = baseline_df[baseline_df['task'] == task]
        for method in task_baseline['method'].unique():
            method_data = task_baseline[task_baseline['method'] == method]
            ax.scatter(
                method_data['lat_p95_ms'],
                method_data['accuracy'],
                label=method,
                alpha=0.6,
                s=50
            )
        
        # Planner points
        task_planner = planner_df[planner_df['task'] == task]
        ax.scatter(
            task_planner['lat_p95_ms'],
            task_planner['accuracy'],
            label='CascadePlanner',
            marker='s',
            s=80,
            alpha=0.8,
            color='red',
            edgecolors='black',
            linewidths=1
        )
        
        ax.set_xlabel('Latency (ms, p95)')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task.capitalize()} Task')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "01_pareto_frontier.png"
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"  Saved: {output_path}")


def plot_baseline_comparison(baseline_df, planner_df, output_dir):
    """Plot baseline comparison (hit rate and accuracy)."""
    LOGGER.info("Generating baseline comparison plot...")
    
    # Aggregate over seeds
    baseline_agg = baseline_df.groupby(['method', 'task']).agg({
        'deadline_hit_rate': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).reset_index()
    
    planner_agg = planner_df.groupby('task').agg({
        'deadline_hit_rate': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).reset_index()
    planner_agg['method'] = 'CascadePlanner'
    
    # Combine
    combined = pd.concat([
        baseline_agg,
        planner_agg
    ])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, task in enumerate(['text', 'vision']):
        task_data = combined[combined['task'] == task]
        
        # Hit rate
        ax = axes[idx, 0]
        x = np.arange(len(task_data))
        ax.bar(x, task_data[('deadline_hit_rate', 'mean')], yerr=task_data[('deadline_hit_rate', 'std')], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(task_data['method'], rotation=45, ha='right')
        ax.set_ylabel('Deadline Hit Rate')
        ax.set_title(f'{task.capitalize()} - Hit Rate')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Accuracy
        ax = axes[idx, 1]
        ax.bar(x, task_data[('accuracy', 'mean')], yerr=task_data[('accuracy', 'std')], capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(task_data['method'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{task.capitalize()} - Accuracy')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "02_baseline_comparison.png"
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"  Saved: {output_path}")


def plot_cascade_threshold(planner_df, output_dir):
    """Plot cascade threshold sensitivity."""
    LOGGER.info("Generating cascade threshold plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, task in enumerate(['text', 'vision']):
        ax = axes[idx]
        
        task_data = planner_df[planner_df['task'] == task]
        
        # Group by threshold
        threshold_agg = task_data.groupby('threshold').agg({
            'deadline_hit_rate': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'coverage': ['mean', 'std']
        }).reset_index()
        
        # Plot
        ax2 = ax.twinx()
        
        l1 = ax.errorbar(
            threshold_agg['threshold'],
            threshold_agg[('deadline_hit_rate', 'mean')],
            yerr=threshold_agg[('deadline_hit_rate', 'std')],
            label='Hit Rate',
            marker='o',
            capsize=5
        )
        
        l2 = ax.errorbar(
            threshold_agg['threshold'],
            threshold_agg[('accuracy', 'mean')],
            yerr=threshold_agg[('accuracy', 'std')],
            label='Accuracy',
            marker='s',
            capsize=5
        )
        
        l3 = ax2.errorbar(
            threshold_agg['threshold'],
            threshold_agg[('coverage', 'mean')],
            yerr=threshold_agg[('coverage', 'std')],
            label='Coverage',
            marker='^',
            capsize=5,
            color='green'
        )
        
        ax.set_xlabel('Cascade Threshold')
        ax.set_ylabel('Hit Rate / Accuracy')
        ax2.set_ylabel('Coverage (Stage 1)')
        ax.set_title(f'{task.capitalize()} Task')
        ax.set_ylim([0, 1.05])
        ax2.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # Combined legend
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    output_path = output_dir / "03_cascade_threshold.png"
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"  Saved: {output_path}")


def plot_ablation_studies(ablation_df, output_dir):
    """Plot ablation studies."""
    LOGGER.info("Generating ablation studies plot...")
    
    ablation_types = ablation_df['ablation_type'].unique()
    n_types = len(ablation_types)
    
    fig, axes = plt.subplots(1, n_types, figsize=(4*n_types, 5))
    if n_types == 1:
        axes = [axes]
    
    for idx, ablation_type in enumerate(ablation_types):
        ax = axes[idx]
        
        type_data = ablation_df[ablation_df['ablation_type'] == ablation_type]
        
        x = np.arange(len(type_data))
        width = 0.35
        
        ax.bar(x - width/2, type_data['latency_change_pct'], width, label='Latency Change', alpha=0.8)
        ax.bar(x + width/2, type_data['accuracy_change_pct'], width, label='Accuracy Change', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(type_data['factor'], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Change (%)')
        ax.set_title(f'{ablation_type.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    output_path = output_dir / "04_ablation_studies.png"
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"  Saved: {output_path}")


def plot_workload_sensitivity(workload_df, output_dir):
    """Plot workload sensitivity."""
    LOGGER.info("Generating workload sensitivity plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, task in enumerate(['text', 'vision']):
        task_data = workload_df[workload_df['task'] == task]
        
        for jdx, workload_type in enumerate(['steady', 'bursty']):
            ax = axes[idx, jdx]
            
            workload_data = task_data[task_data['workload_type'] == workload_type]
            
            for method in workload_data['method'].unique():
                method_data = workload_data[workload_data['method'] == method]
                ax.plot(
                    method_data['arrival_rate'],
                    method_data['deadline_hit_rate'],
                    marker='o',
                    label=method
                )
            
            ax.set_xlabel('Arrival Rate (req/s)')
            ax.set_ylabel('Deadline Hit Rate')
            ax.set_title(f'{task.capitalize()} - {workload_type.capitalize()}')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = output_dir / "05_workload_sensitivity.png"
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"  Saved: {output_path}")


def plot_statistical_significance(stats_df, output_dir):
    """Plot statistical significance (effect sizes)."""
    LOGGER.info("Generating statistical significance plot...")
    
    # Filter for primary metric
    hit_rate_stats = stats_df[stats_df['metric'] == 'deadline_hit_rate']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(hit_rate_stats))
    
    # Plot effect sizes with significance markers
    colors = ['green' if sig else 'gray' for sig in hit_rate_stats['significant_at_0.05']]
    
    ax.barh(x, hit_rate_stats['cohens_d'], color=colors, alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(hit_rate_stats['baseline_method'])
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title("CascadePlanner vs Baselines - Effect Sizes")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Small')
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Medium')
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.5, label='Large')
    ax.legend(title='Effect Size', loc='best')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add significance annotations
    for i, (_, row) in enumerate(hit_rate_stats.iterrows()):
        if row['significant_at_0.05']:
            ax.text(row['cohens_d'] + 0.05, i, f"p={row['p_value']:.3f}", va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "06_statistical_significance.png"
    plt.savefig(output_path)
    plt.close()
    LOGGER.info(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    LOGGER.info("Loading results...")
    
    # Load all results
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    pareto_df = pd.read_csv(results_dir / "pareto_analysis.csv") if (results_dir / "pareto_analysis.csv").exists() else pd.DataFrame()
    ablation_df = pd.read_csv(results_dir / "ablation_results.csv") if (results_dir / "ablation_results.csv").exists() else pd.DataFrame()
    workload_df = pd.read_csv(results_dir / "workload_sensitivity.csv") if (results_dir / "workload_sensitivity.csv").exists() else pd.DataFrame()
    stats_df = pd.read_csv(results_dir / "statistical_tests.csv") if (results_dir / "statistical_tests.csv").exists() else pd.DataFrame()
    
    # Generate figures
    plot_pareto_frontier(pareto_df, baseline_df, planner_df, figures_dir)
    plot_baseline_comparison(baseline_df, planner_df, figures_dir)
    plot_cascade_threshold(planner_df, figures_dir)
    
    if len(ablation_df) > 0:
        plot_ablation_studies(ablation_df, figures_dir)
    
    if len(workload_df) > 0:
        plot_workload_sensitivity(workload_df, figures_dir)
    
    if len(stats_df) > 0:
        plot_statistical_significance(stats_df, figures_dir)
    
    LOGGER.info(f"\nComplete. Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()

