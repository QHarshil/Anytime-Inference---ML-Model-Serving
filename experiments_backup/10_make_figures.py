"""
Generate publication-quality figures from experimental results.

Figures generated:
1. Pareto frontier (latency vs. accuracy)
2. Deadline hit rate comparison (bar chart)
3. Accuracy comparison (bar chart)
4. Ablation studies (grouped bar charts)
5. Workload sensitivity (line plots)
6. Statistical significance (effect size visualization)

Output: results/plots/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.visualization")

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def load_all_results():
    """Load all experimental results."""
    results = {}
    
    files = {
        'baseline': 'results/baseline_results.csv',
        'planner': 'results/planner_results.csv',
        'statistical': 'results/statistical_tests.csv',
        'pareto': 'results/pareto_analysis.csv',
        'ablation': 'results/ablation_results.csv',
        'workload': 'results/workload_results.csv'
    }
    
    for name, path in files.items():
        if Path(path).exists():
            results[name] = pd.read_csv(path)
            LOGGER.info(f"Loaded {name}: {len(results[name])} rows")
        else:
            LOGGER.warning(f"File not found: {path}")
            results[name] = None
    
    return results


def plot_pareto_frontier(baseline_df, planner_df, output_path):
    """Plot Pareto frontier for all methods."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot baseline methods
    for method in baseline_df['method'].unique():
        subset = baseline_df[baseline_df['method'] == method]
        ax.scatter(subset['lat_p95_ms'], subset['accuracy'], 
                  label=method, alpha=0.6, s=100)
    
    # Plot planner
    ax.scatter(planner_df['lat_p95_ms'], planner_df['accuracy'], 
              label='CascadePlanner', alpha=0.8, s=150, marker='*', 
              edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('Latency p95 (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Pareto Frontier: Latency vs. Accuracy Trade-off')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved Pareto frontier plot to {output_path}")


def plot_deadline_hit_rate(baseline_df, planner_df, output_path):
    """Plot deadline hit rate comparison."""
    # Combine data
    all_data = pd.concat([baseline_df, planner_df])
    
    # Group by method and deadline
    grouped = all_data.groupby(['method', 'deadline_ms'])['deadline_hit_rate'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bar chart
    methods = grouped['method'].unique()
    deadlines = sorted(grouped['deadline_ms'].unique())
    x = np.arange(len(deadlines))
    width = 0.2
    
    for i, method in enumerate(methods):
        method_data = grouped[grouped['method'] == method]
        values = [method_data[method_data['deadline_ms'] == d]['deadline_hit_rate'].values[0] 
                 if len(method_data[method_data['deadline_ms'] == d]) > 0 else 0 
                 for d in deadlines]
        ax.bar(x + i * width, values, width, label=method)
    
    ax.set_xlabel('Deadline (ms)')
    ax.set_ylabel('Deadline Hit Rate')
    ax.set_title('Deadline Hit Rate Comparison')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(deadlines)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved deadline hit rate plot to {output_path}")


def plot_accuracy_comparison(baseline_df, planner_df, output_path):
    """Plot accuracy comparison."""
    # Combine data
    all_data = pd.concat([baseline_df, planner_df])
    
    # Group by method and deadline
    grouped = all_data.groupby(['method', 'deadline_ms'])['accuracy'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bar chart
    methods = grouped['method'].unique()
    deadlines = sorted(grouped['deadline_ms'].unique())
    x = np.arange(len(deadlines))
    width = 0.2
    
    for i, method in enumerate(methods):
        method_data = grouped[grouped['method'] == method]
        values = [method_data[method_data['deadline_ms'] == d]['accuracy'].values[0] 
                 if len(method_data[method_data['deadline_ms'] == d]) > 0 else 0 
                 for d in deadlines]
        ax.bar(x + i * width, values, width, label=method)
    
    ax.set_xlabel('Deadline (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Comparison')
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(deadlines)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved accuracy comparison plot to {output_path}")


def plot_ablation_studies(ablation_df, output_path):
    """Plot ablation study results."""
    if ablation_df is None or len(ablation_df) == 0:
        LOGGER.warning("No ablation data to plot")
        return
    
    # Create subplots for each ablation type
    ablation_types = ablation_df['ablation_type'].unique()
    n_types = len(ablation_types)
    
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
    if n_types == 1:
        axes = [axes]
    
    for i, ablation_type in enumerate(ablation_types):
        subset = ablation_df[ablation_df['ablation_type'] == ablation_type]
        
        ax = axes[i]
        
        # Plot latency and accuracy changes
        x = np.arange(len(subset))
        width = 0.35
        
        ax.bar(x - width/2, subset['latency_change_pct'], width, label='Latency Change %', alpha=0.7)
        ax.bar(x + width/2, subset['accuracy_change_pct'], width, label='Accuracy Change %', alpha=0.7)
        
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Change (%)')
        ax.set_title(f'Ablation: {ablation_type}')
        ax.set_xticks(x)
        ax.set_xticklabels(subset['factor'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved ablation studies plot to {output_path}")


def plot_workload_sensitivity(workload_df, output_path):
    """Plot workload sensitivity results."""
    if workload_df is None or len(workload_df) == 0:
        LOGGER.warning("No workload data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Deadline hit rate by workload
    for method in workload_df['method'].unique():
        for workload in workload_df['workload_type'].unique():
            subset = workload_df[(workload_df['method'] == method) & (workload_df['workload_type'] == workload)]
            marker = 'o' if workload == 'steady' else 's'
            linestyle = '-' if workload == 'steady' else '--'
            ax1.plot(subset['deadline_ms'], subset['deadline_hit_rate'], 
                    marker=marker, linestyle=linestyle, label=f'{method}/{workload}')
    
    ax1.set_xlabel('Deadline (ms)')
    ax1.set_ylabel('Deadline Hit Rate')
    ax1.set_title('Workload Sensitivity: Hit Rate')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Queue length by workload
    for method in workload_df['method'].unique():
        for workload in workload_df['workload_type'].unique():
            subset = workload_df[(workload_df['method'] == method) & (workload_df['workload_type'] == workload)]
            marker = 'o' if workload == 'steady' else 's'
            linestyle = '-' if workload == 'steady' else '--'
            ax2.plot(subset['deadline_ms'], subset['p95_queue_length'], 
                    marker=marker, linestyle=linestyle, label=f'{method}/{workload}')
    
    ax2.set_xlabel('Deadline (ms)')
    ax2.set_ylabel('Queue Length (p95)')
    ax2.set_title('Workload Sensitivity: Queue Length')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved workload sensitivity plot to {output_path}")


def plot_statistical_significance(stats_df, output_path):
    """Plot statistical significance results."""
    if stats_df is None or len(stats_df) == 0:
        LOGGER.warning("No statistical data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Cohen's d effect sizes
    x = np.arange(len(stats_df))
    colors = ['green' if p < 0.05 else 'gray' for p in stats_df['p_value']]
    
    ax.barh(x, stats_df['cohens_d'], color=colors, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{row['baseline_method']}\nvs Planner\n({row['metric']})" 
                        for _, row in stats_df.iterrows()])
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title('Statistical Significance of Improvements')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.2, color='blue', linestyle='--', linewidth=0.5, alpha=0.5, label='Small effect')
    ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=0.5, alpha=0.5, label='Medium effect')
    ax.axvline(x=0.8, color='red', linestyle='--', linewidth=0.5, alpha=0.5, label='Large effect')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved statistical significance plot to {output_path}")


def main():
    LOGGER.info("Generating publication-quality figures...")
    
    # Load all results
    results = load_all_results()
    
    # Create output directory
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    if results['baseline'] is not None and results['planner'] is not None:
        plot_pareto_frontier(results['baseline'], results['planner'], 
                           output_dir / "01_pareto_frontier.png")
        plot_deadline_hit_rate(results['baseline'], results['planner'], 
                             output_dir / "02_deadline_hit_rate.png")
        plot_accuracy_comparison(results['baseline'], results['planner'], 
                               output_dir / "03_accuracy_comparison.png")
    
    if results['ablation'] is not None:
        plot_ablation_studies(results['ablation'], 
                            output_dir / "04_ablation_studies.png")
    
    if results['workload'] is not None:
        plot_workload_sensitivity(results['workload'], 
                                output_dir / "05_workload_sensitivity.png")
    
    if results['statistical'] is not None:
        plot_statistical_significance(results['statistical'], 
                                     output_dir / "06_statistical_significance.png")
    
    LOGGER.info(f"\nAll figures saved to {output_dir}")
    LOGGER.info("Figure generation complete.")


if __name__ == "__main__":
    main()

