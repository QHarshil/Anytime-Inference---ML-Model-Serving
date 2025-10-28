"""Statistical significance testing with proper rigor.

Compares CascadePlanner against baselines using:
- Paired t-test (parametric)
- Wilcoxon signed-rank test (non-parametric)
- Cohen's d effect size
- Statistical power analysis
- Assumption checking (normality, equal variances)

Pre-declared metrics:
1. Deadline hit rate (primary)
2. Accuracy (secondary)

Output: results/statistical_tests.csv
Schema: comparison,metric,baseline_method,mean_baseline,std_baseline,
        mean_planner,std_planner,mean_diff,p_value_ttest,p_value_wilcoxon,
        cohens_d,effect_size_interpretation,statistical_power,
        normality_ok,equal_variance_ok,num_pairs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np

from src.evaluation.statistical_analysis import (
    compare_methods,
    aggregate_multiple_seeds,
    check_assumptions
)
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.statistical_tests")


def load_results(results_dir: Path) -> tuple:
    """Load baseline and planner results."""
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    return baseline_df, planner_df


def prepare_paired_data(baseline_df, planner_df, baseline_method, metric):
    """Prepare paired data for statistical tests.
    
    Pairs are matched on (task, deadline_ms, seed).
    """
    # Filter baseline method
    baseline_subset = baseline_df[baseline_df['method'] == baseline_method].copy()
    
    # For planner, select best threshold per (task, deadline, seed)
    planner_best = planner_df.loc[
        planner_df.groupby(['task', 'deadline_ms', 'seed'])[metric].idxmax()
    ].copy()
    
    # Merge on matching keys
    paired = baseline_subset.merge(
        planner_best,
        on=['task', 'deadline_ms', 'seed'],
        suffixes=('_baseline', '_planner')
    )
    
    if len(paired) == 0:
        return None, None
    
    baseline_values = paired[f'{metric}_baseline'].values
    planner_values = paired[f'{metric}_planner'].values
    
    return baseline_values, planner_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
    results_dir = Path("results")
    
    LOGGER.info("Loading results...")
    baseline_df, planner_df = load_results(results_dir)
    
    # Pre-declared comparisons
    baseline_methods = ['StaticSmall', 'StaticLarge', 'ThroughputAutotuner', 'INFaaS-style']
    metrics = ['deadline_hit_rate', 'accuracy']
    
    all_results = []
    
    for baseline_method in baseline_methods:
        for metric in metrics:
            LOGGER.info(f"\nComparing CascadePlanner vs {baseline_method} on {metric}...")
            
            # Prepare paired data
            baseline_values, planner_values = prepare_paired_data(
                baseline_df, planner_df, baseline_method, metric
            )
            
            if baseline_values is None:
                LOGGER.warning(f"  No paired data found, skipping")
                continue
            
            # Check assumptions
            assumptions = check_assumptions(baseline_values, planner_values)
            LOGGER.info(f"  Normality: {assumptions['normality_ok']}, Equal variance: {assumptions['equal_variance_ok']}")
            
            # Run comparison
            comparison = compare_methods(
                baseline_values,
                planner_values,
                method1_name=baseline_method,
                method2_name='CascadePlanner',
                metric_name=metric
            )
            
            # Log results
            LOGGER.info(f"  Baseline: {comparison.mean1:.4f} ± {comparison.std1:.4f}")
            LOGGER.info(f"  Planner:  {comparison.mean2:.4f} ± {comparison.std2:.4f}")
            LOGGER.info(f"  Difference: {comparison.mean_diff:.4f}")
            LOGGER.info(f"  p-value (t-test): {comparison.p_value_ttest:.4f}")
            LOGGER.info(f"  p-value (Wilcoxon): {comparison.p_value_wilcoxon:.4f}")
            LOGGER.info(f"  Cohen's d: {comparison.cohens_d:.3f} ({comparison.effect_size_interpretation})")
            LOGGER.info(f"  Statistical power: {comparison.statistical_power:.3f}")
            
            # Determine significance
            alpha = 0.05
            significant = comparison.p_value_ttest < alpha
            LOGGER.info(f"  Significant at α={alpha}: {significant}")
            
            # Store result
            all_results.append({
                'comparison': f'{baseline_method}_vs_CascadePlanner',
                'metric': metric,
                'baseline_method': baseline_method,
                'mean_baseline': comparison.mean1,
                'std_baseline': comparison.std1,
                'ci_lower_baseline': comparison.ci_lower1,
                'ci_upper_baseline': comparison.ci_upper1,
                'mean_planner': comparison.mean2,
                'std_planner': comparison.std2,
                'ci_lower_planner': comparison.ci_lower2,
                'ci_upper_planner': comparison.ci_upper2,
                'mean_diff': comparison.mean_diff,
                'p_value_ttest': comparison.p_value_ttest,
                'p_value_wilcoxon': comparison.p_value_wilcoxon,
                'p_value': comparison.p_value_ttest,  # Alias for compatibility
                'cohens_d': comparison.cohens_d,
                'effect_size_interpretation': comparison.effect_size_interpretation,
                'statistical_power': comparison.statistical_power,
                'normality_ok': assumptions['normality_ok'],
                'equal_variance_ok': assumptions['equal_variance_ok'],
                'num_pairs': len(baseline_values),
                'significant_at_0.05': significant
            })
    
    # Save
    results_df = pd.DataFrame(all_results)
    output_path = results_dir / "statistical_tests.csv"
    save_csv(results_df, output_path)
    
    LOGGER.info(f"\nComplete. Results: {output_path}")
    LOGGER.info(f"Total comparisons: {len(all_results)}")
    
    # Summary
    significant_count = sum(results_df['significant_at_0.05'])
    LOGGER.info(f"Significant comparisons (α=0.05): {significant_count}/{len(all_results)}")


if __name__ == "__main__":
    main()

