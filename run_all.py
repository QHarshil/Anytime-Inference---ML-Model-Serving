#!/usr/bin/env python3
"""Run the offline profiling and evaluation pipeline.

Stages are declared in ``STAGES`` rather than spelled out one call at a time, so
``--quick`` reaches every stage that accepts it and the pass/fail policy for each
stage is visible in one place.

Profiling and evaluation stages are required: a failure there leaves downstream
stages without input, so the run stops. Analysis stages only post-process
results already on disk, so a failure there is reported and the run continues.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BAR = "=" * 80


@dataclass(frozen=True)
class Stage:
    script: str
    description: str
    group: str
    required: bool

    def command(self, python: str, quick: bool) -> list[str]:
        cmd = [python, self.script]
        if quick:
            cmd.append("--quick")
        return cmd


# ``download`` takes no --quick flag; handled separately below.
DOWNLOAD = Stage(
    script="data/download_datasets.py",
    description="Download datasets (SST-2, CIFAR-10)",
    group="download",
    required=True,
)

STAGES: tuple[Stage, ...] = (
    Stage(
        "experiments/01_profile_latency.py",
        "Latency profiling (text + image models)",
        "profiling",
        True,
    ),
    Stage(
        "experiments/02_profile_accuracy.py",
        "Accuracy profiling (text + image models)",
        "profiling",
        True,
    ),
    Stage(
        "experiments/03_run_baselines.py",
        "Baseline evaluation (Static, Heuristic, INFaaS-style)",
        "evaluation",
        True,
    ),
    Stage(
        "experiments/04_run_planner.py", "Planner evaluation (CascadePlanner)", "evaluation", True
    ),
    Stage(
        "experiments/06_statistical_tests.py",
        "Statistical significance tests (paired t-test, Wilcoxon, Cohen's d)",
        "analysis",
        False,
    ),
    Stage(
        "experiments/07_pareto_analysis.py",
        "Pareto frontier analysis (hypervolume, dominance ratio)",
        "analysis",
        False,
    ),
    Stage(
        "experiments/05_ablation.py",
        "Ablation studies (model size, quantisation, batch size)",
        "analysis",
        False,
    ),
    Stage(
        "experiments/08_workload.py", "Workload sensitivity (steady vs bursty)", "analysis", False
    ),
    Stage(
        "experiments/09_failure_analysis.py",
        "Failure analysis (deadline miss, model crash, workload spike)",
        "analysis",
        False,
    ),
    Stage(
        "experiments/10_make_figures.py",
        "Generate figures (Pareto, hit-rate, ablations)",
        "analysis",
        False,
    ),
)

OUTPUTS = (
    "results/latency_profiles.csv",
    "results/accuracy_profiles.csv",
    "results/baseline_results.csv",
    "results/planner_results.csv",
    "results/statistical_tests.csv",
    "results/pareto_analysis.csv",
    "results/ablation_results.csv",
    "results/workload_sensitivity.csv",
    "results/failure_miss_analysis.csv",
    "results/failure_degradation_strategies.csv",
    "results/figures/*.png",
)


def run_command(cmd: list[str], description: str) -> bool:
    """Run one stage, echoing its command and wall time."""
    print(f"\n{BAR}\nSTEP: {description}\n{BAR}")
    print(f"Command: {' '.join(cmd)}\n")

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nFAILED: {description} after {(time.time() - start) / 60:.1f} min "
            f"(exit {exc.returncode})"
        )
        return False
    print(f"\nOK: {description} in {(time.time() - start) / 60:.1f} min")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Anytime Inference Planner pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument(
        "--skip-profiling", action="store_true", help="Skip profiling (latency + accuracy)"
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Skip evaluation (baselines + planner)"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Forward --quick to every stage that accepts it"
    )
    args = parser.parse_args()

    if not Path("pyproject.toml").exists():
        print("Error: must run from the repository root", file=sys.stderr)
        return 1

    skipped_groups = set()
    if args.skip_profiling:
        skipped_groups.add("profiling")
    if args.skip_evaluation:
        skipped_groups.add("evaluation")

    print(BAR)
    print("ANYTIME INFERENCE PLANNER - OFFLINE PIPELINE")
    print(BAR)
    print(f"Skip download:   {args.skip_download}")
    print(f"Skip profiling:  {args.skip_profiling}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print(f"Quick test mode: {args.quick_test}")
    print(BAR)

    started = time.time()
    failures: list[str] = []

    if args.skip_download:
        print(f"\nSKIP: {DOWNLOAD.description}")
    elif not run_command([sys.executable, DOWNLOAD.script], DOWNLOAD.description):
        print(f"\nPipeline stopped: {DOWNLOAD.description} is required")
        return 1

    for stage in STAGES:
        if stage.group in skipped_groups:
            print(f"\nSKIP: {stage.description}")
            continue
        if run_command(stage.command(sys.executable, args.quick_test), stage.description):
            continue
        if stage.required:
            print(f"\nPipeline stopped: {stage.description} is required by later stages")
            return 1
        failures.append(stage.description)

    elapsed = time.time() - started
    print(f"\n{BAR}")
    print("PIPELINE COMPLETED" if not failures else "PIPELINE COMPLETED WITH FAILURES")
    print(BAR)
    print(f"Total time: {elapsed / 60:.1f} minutes")
    if failures:
        print(f"\n{len(failures)} optional stage(s) failed:")
        for description in failures:
            print(f"  - {description}")
    print("\nResults written to:")
    for output in OUTPUTS:
        print(f"  - {output}")
    print(BAR)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
