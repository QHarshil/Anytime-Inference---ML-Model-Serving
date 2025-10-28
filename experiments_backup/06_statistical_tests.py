"""
Statistical significance testing for baseline vs. planner comparisons.

Tests:
1. Paired t-test: Tests if mean difference is significantly different from zero
2. Wilcoxon signed-rank test: Non-parametric alternative to paired t-test
3. Cohen's d: Effect size measure

Comparisons:
- CascadePlanner vs. StaticSmall (on deadline_hit_rate and accuracy)
- CascadePlanner vs. StaticLarge (on deadline_hit_rate and accuracy)
- CascadePlanner vs. ThroughputAutotuner (on deadline_hit_rate and accuracy)

Output: results/statistical_tests.csv
Schema: comparison,metric,mean_diff,t_statistic,p_value_ttest,p_value_wilcoxon,cohens_d,significant,interpretation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.statistical_tests")

ALPHA = 0.05  # Significance level


def load_results():
    """Load baseline and planner results."""
    LOGGER.info("Loading evaluation results...")
    
    baseline_path = Path("results/baseline_results.csv")
    planner_path = Path("results/planner_results.csv")
    
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results not found. Run 03_run_baselines.py first.")
    if not planner_path.exists():
        raise FileNotFoundError(f"Planner results not found. Run 04_run_planner.py first.")
    
    baseline_df = pd.read_csv(baseline_path)
    planner_df = pd.read_csv(planner_path)
    
    LOGGER.info(f"Loaded {len(baseline_df)} baseline results")
    LOGGER.info(f"Loaded {len(planner_df)} planner results")
    
    return baseline_df, planner_df


def cohens_d(x, y):
    """Compute Cohen's d effect size."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0


def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_comparison(planner_values, baseline_values, comparison_name, metric_name):
    """Perform paired statistical tests."""
    if len(planner_values) != len(baseline_values):
        LOGGER.warning(f"Mismatched lengths for {comparison_name} on {metric_name}: {len(planner_values)} vs {len(baseline_values)}")
        return None
    
    if len(planner_values) < 2:
        LOGGER.warning(f"Insufficient data for {comparison_name} on {metric_name}: {len(planner_values)} samples")
        return None
    
    # Compute differences
    differences = np.array(planner_values) - np.array(baseline_values)
    mean_diff = np.mean(differences)
    
    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(planner_values, baseline_values)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(planner_values, baseline_values)
    except ValueError:
        # All differences are zero
        p_wilcoxon = 1.0
    
    # Cohen's d
    d = cohens_d(planner_values, baseline_values)
    interpretation = interpret_cohens_d(d)
    
    # Significance
    significant = p_ttest < ALPHA
    
    LOGGER.info(f"  {comparison_name} - {metric_name}:")
    LOGGER.info(f"    Mean difference: {mean_diff:.4f}")
    LOGGER.info(f"    t-statistic: {t_stat:.4f}, p-value: {p_ttest:.4f}")
    LOGGER.info(f"    Wilcoxon p-value: {p_wilcoxon:.4f}")
    LOGGER.info(f"    Cohen's d: {d:.4f} ({interpretation})")
    LOGGER.info(f"    Significant: {significant}")
    
    # Extract baseline method from comparison name
    baseline_method = comparison_name.replace('CascadePlanner vs. ', '')
    
    return {
        'comparison': comparison_name,
        'baseline_method': baseline_method,
        'metric': metric_name,
        'mean_diff': mean_diff,
        't_statistic': t_stat,
        'p_value': p_ttest,  # Primary p-value for visualization
        'p_value_ttest': p_ttest,
        'p_value_wilcoxon': p_wilcoxon,
        'cohens_d': d,
        'effect_size': interpretation,
        'significant': significant,
        'n_pairs': len(planner_values)
    }


def align_results(planner_df, baseline_df, baseline_method):
    """Align planner and baseline results by task and deadline."""
    # Merge on task and deadline_ms
    merged = pd.merge(
        planner_df[['task', 'deadline_ms', 'deadline_hit_rate', 'accuracy']],
        baseline_df[baseline_df['method'] == baseline_method][['task', 'deadline_ms', 'deadline_hit_rate', 'accuracy']],
        on=['task', 'deadline_ms'],
        suffixes=('_planner', '_baseline')
    )
    
    return merged


def main():
    LOGGER.info("Starting statistical significance testing...")
    LOGGER.info(f"Significance level: α = {ALPHA}")
    
    # Load results
    baseline_df, planner_df = load_results()
    
    results = []
    
    # Compare CascadePlanner vs. each baseline
    baseline_methods = ['StaticSmall', 'StaticLarge', 'ThroughputAutotuner']
    metrics = ['deadline_hit_rate', 'accuracy']
    
    for baseline_method in baseline_methods:
        LOGGER.info(f"\nComparing CascadePlanner vs. {baseline_method}...")
        
        # Align results
        aligned = align_results(planner_df, baseline_df, baseline_method)
        
        if len(aligned) == 0:
            LOGGER.warning(f"No aligned results for {baseline_method}")
            continue
        
        LOGGER.info(f"  Aligned {len(aligned)} paired measurements")
        
        # Test each metric
        for metric in metrics:
            planner_values = aligned[f'{metric}_planner'].values
            baseline_values = aligned[f'{metric}_baseline'].values
            
            result = paired_comparison(
                planner_values,
                baseline_values,
                f"CascadePlanner vs. {baseline_method}",
                metric
            )
            
            if result is not None:
                results.append(result)
    
    # Save results
    if len(results) > 0:
        df = pd.DataFrame(results)
        output_path = Path("results/statistical_tests.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_csv(df, output_path)
        
        LOGGER.info(f"\nStatistical testing complete. Results saved to {output_path}")
        LOGGER.info(f"Total tests: {len(results)}")
        
        # Summary
        significant_count = df['significant'].sum()
        LOGGER.info(f"\nSummary:")
        LOGGER.info(f"  Significant results: {significant_count}/{len(results)}")
        LOGGER.info(f"  Mean Cohen's d: {df['cohens_d'].mean():.3f}")
        
        # Highlight key findings
        LOGGER.info("\nKey findings:")
        for _, row in df.iterrows():
            if row['significant']:
                direction = "higher" if row['mean_diff'] > 0 else "lower"
                LOGGER.info(f"  {row['comparison']} - {row['metric']}: {direction} by {abs(row['mean_diff']):.3f} (p={row['p_value_ttest']:.4f}, d={row['cohens_d']:.3f})")
    else:
        LOGGER.warning("No statistical tests performed")


if __name__ == "__main__":
    main()

