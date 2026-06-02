"""Statistical significance tests comparing CascadePlanner against baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd

from src.evaluation.statistical_analysis import compare_methods, check_assumptions
from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.statistical_tests")


def load_results(results_dir: Path):
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    return baseline_df, planner_df


def prepare_paired_data(baseline_df, planner_df, baseline_method, metric):
    baseline_subset = baseline_df[baseline_df["method"] == baseline_method].copy()
    planner_best = planner_df.loc[
        planner_df.groupby(["task", "deadline_ms", "seed"])[metric].idxmax()
    ].copy()
    paired = baseline_subset.merge(
        planner_best, on=["task", "deadline_ms", "seed"], suffixes=("_baseline", "_planner")
    )
    if paired.empty:
        return None, None
    return paired[f"{metric}_baseline"].to_numpy(), paired[f"{metric}_planner"].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    _ = args.quick

    results_dir = Path("results")
    baseline_df, planner_df = load_results(results_dir)

    baseline_methods = ["StaticSmall", "StaticLarge", "ThroughputAutotuner", "INFaaS-style"]
    metrics = ["deadline_hit_rate", "accuracy"]

    rows = []
    alpha = 0.05
    for baseline_method in baseline_methods:
        for metric in metrics:
            LOGGER.info("Comparing CascadePlanner vs %s on %s", baseline_method, metric)
            baseline_values, planner_values = prepare_paired_data(
                baseline_df, planner_df, baseline_method, metric
            )
            if baseline_values is None:
                LOGGER.warning("  No paired data, skipping")
                continue

            assumptions = check_assumptions(baseline_values, planner_values)
            comparison = compare_methods(
                baseline_values,
                planner_values,
                method_a_name=baseline_method,
                method_b_name="CascadePlanner",
                metric_name=metric,
                alpha=alpha,
            )

            LOGGER.info("  baseline=%.4f ± %.4f", comparison.mean_a, comparison.std_a)
            LOGGER.info("  planner =%.4f ± %.4f", comparison.mean_b, comparison.std_b)
            LOGGER.info("  diff=%.4f  p_ttest=%.4f  d=%.3f (%s)  power=%.3f",
                        comparison.mean_diff,
                        comparison.p_value_ttest,
                        comparison.cohens_d,
                        comparison.effect_size_interpretation,
                        comparison.statistical_power)

            rows.append({
                "comparison": f"{baseline_method}_vs_CascadePlanner",
                "metric": metric,
                "baseline_method": baseline_method,
                "mean_baseline": comparison.mean_a,
                "std_baseline": comparison.std_a,
                "mean_planner": comparison.mean_b,
                "std_planner": comparison.std_b,
                "mean_diff": comparison.mean_diff,
                "ci_lower_diff": comparison.ci_lower,
                "ci_upper_diff": comparison.ci_upper,
                "p_value_ttest": comparison.p_value_ttest,
                "p_value_wilcoxon": comparison.p_value_wilcoxon,
                "cohens_d": comparison.cohens_d,
                "effect_size_interpretation": comparison.effect_size_interpretation,
                "statistical_power": comparison.statistical_power,
                "normality_ok": assumptions["normality_ok"],
                "equal_variance_ok": assumptions["equal_variance_ok"],
                "num_pairs": len(baseline_values),
                "significant_at_0.05": comparison.significant,
            })

    results_df = pd.DataFrame(rows)
    output_path = results_dir / "statistical_tests.csv"
    save_csv(results_df, output_path)
    LOGGER.info("Wrote %d comparisons to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
