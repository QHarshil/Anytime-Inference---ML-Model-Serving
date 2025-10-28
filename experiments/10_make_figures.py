"""Generate plots from evaluation results."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import read_dataframe
from src.utils.logger import configure_logger
from src.utils.visualization import plot_metric_vs_deadline, plot_pareto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", default="results/planner_results.csv")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    configure_logger()
    planner_df = read_dataframe(Path(args.planner))
    output_dir = Path(args.output_dir)

    plot_pareto(
        planner_df,
        title="CascadePlanner Pareto Frontier",
        path=output_dir / "pareto_frontiers.png",
    )
    plot_metric_vs_deadline(
        planner_df,
        metric="deadline_hit_rate",
        title="Deadline Hit-Rate vs Deadline",
        ylabel="Hit-Rate",
        path=output_dir / "deadline_hit_rate.png",
    )


if __name__ == "__main__":
    main()
