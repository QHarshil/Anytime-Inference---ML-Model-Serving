"""Plotting utilities."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from .io import ensure_parent_dir
from .logger import get_logger

LOGGER = get_logger("utils.visualization")


def plot_pareto(df: pd.DataFrame, *, title: str, path: Path) -> None:
    """Plot a latency vs. accuracy Pareto scatter plot."""

    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["lat_p95_ms"], df["accuracy"], c="tab:blue", label="Configs")
    ax.set_xlabel("Latency P95 (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved Pareto plot to %s", path)


def plot_metric_vs_deadline(
    df: pd.DataFrame,
    metric: str,
    *,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    """Plot a metric against deadlines."""

    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped = df.groupby("deadline")[metric].mean().sort_index()
    ax.plot(grouped.index, grouped.values, marker="o", label=metric)
    ax.set_xlabel("Deadline (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved metric plot to %s", path)
