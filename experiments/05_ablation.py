"""Generate simple ablation summaries from profiling data."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import read_dataframe, write_dataframe
from src.utils.logger import configure_logger


ABLATION_COLUMNS = {
    "batch_size": "batch_size",
    "model_size": "model",
    "quantization": "variant",
    "device": "device",
}


def summarise(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = df.groupby(column).agg({"lat_p95_ms": "mean", "accuracy": "mean"}).reset_index()
    grouped.rename(columns={"lat_p95_ms": "latency_p95_ms"}, inplace=True)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="results/planner_results.csv")
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()

    configure_logger()
    df = read_dataframe(Path(args.profiles))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, column in ABLATION_COLUMNS.items():
        if column not in df.columns:
            continue
        summary = summarise(df, column)
        write_dataframe(summary, output_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
