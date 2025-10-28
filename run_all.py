#!/usr/bin/env python3
"""
Master script to run the entire Anytime Inference Planner pipeline.

Usage:
    python run_all.py [--skip-download] [--skip-profiling] [--skip-evaluation]

This script will:
1. Download datasets (if not skipped)
2. Run latency profiling (if not skipped)
3. Run accuracy profiling (if not skipped)
4. Run baseline evaluation (if not skipped)
5. Run planner evaluation (if not skipped)
6. Run statistical tests
7. Run Pareto analysis
8. Run ablation studies
9. Run workload experiments
10. Run failure analysis
11. Generate all figures

Estimated time: 40-46 hours on GPU
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Anytime Inference Planner pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-profiling", action="store_true", help="Skip profiling (latency + accuracy)")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation (baselines + planner)")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with minimal data")
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("Error: Must run from repository root directory")
        sys.exit(1)
    
    print("="*80)
    print("ANYTIME INFERENCE PLANNER - FULL PIPELINE")
    print("="*80)
    print(f"Skip download: {args.skip_download}")
    print(f"Skip profiling: {args.skip_profiling}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print(f"Quick test mode: {args.quick_test}")
    print("="*80)
    
    start_time_total = time.time()
    
    # Step 1: Download datasets
    if not args.skip_download:
        success = run_command(
            [sys.executable, "data/download_datasets.py"],
            "Download datasets (SST-2, CIFAR-10)"
        )
        if not success:
            print("\n✗ Pipeline failed at dataset download")
            sys.exit(1)
    else:
        print("\n⊘ Skipping dataset download")
    
    # Step 2: Latency profiling
    if not args.skip_profiling:
        cmd = [sys.executable, "experiments/01_profile_latency.py"]
        if args.quick_test:
            cmd.append("--quick")
        
        success = run_command(cmd, "Latency profiling (text + image models)")
        if not success:
            print("\n✗ Pipeline failed at latency profiling")
            sys.exit(1)
    else:
        print("\n⊘ Skipping latency profiling")
    
    # Step 3: Accuracy profiling
    if not args.skip_profiling:
        cmd = [sys.executable, "experiments/02_profile_accuracy.py"]
        if args.quick_test:
            cmd.append("--quick")
        
        success = run_command(cmd, "Accuracy profiling (text + image models)")
        if not success:
            print("\n✗ Pipeline failed at accuracy profiling")
            sys.exit(1)
    else:
        print("\n⊘ Skipping accuracy profiling")
    
    # Step 4: Baseline evaluation
    if not args.skip_evaluation:
        success = run_command(
            [sys.executable, "experiments/03_run_baselines.py"],
            "Baseline evaluation (Static, Heuristic, INFaaS-style)"
        )
        if not success:
            print("\n✗ Pipeline failed at baseline evaluation")
            sys.exit(1)
    else:
        print("\n⊘ Skipping baseline evaluation")
    
    # Step 5: Planner evaluation
    if not args.skip_evaluation:
        success = run_command(
            [sys.executable, "experiments/04_run_planner.py"],
            "Planner evaluation (CascadePlanner)"
        )
        if not success:
            print("\n✗ Pipeline failed at planner evaluation")
            sys.exit(1)
    else:
        print("\n⊘ Skipping planner evaluation")
    
    # Step 6: Statistical tests
    success = run_command(
        [sys.executable, "experiments/06_statistical_tests.py"],
        "Statistical significance tests (paired t-test, Wilcoxon, Cohen's d)"
    )
    if not success:
        print("\n⚠ Warning: Statistical tests failed, continuing...")
    
    # Step 7: Pareto analysis
    success = run_command(
        [sys.executable, "experiments/07_pareto_analysis.py"],
        "Pareto frontier analysis (hypervolume, dominance ratio)"
    )
    if not success:
        print("\n⚠ Warning: Pareto analysis failed, continuing...")
    
    # Step 8: Ablation studies
    success = run_command(
        [sys.executable, "experiments/05_ablation.py"],
        "Ablation studies (model size, quantization, batch size)"
    )
    if not success:
        print("\n⚠ Warning: Ablation studies failed, continuing...")
    
    # Step 9: Workload experiments
    success = run_command(
        [sys.executable, "experiments/08_workload.py"],
        "Workload sensitivity (steady vs. bursty)"
    )
    if not success:
        print("\n⚠ Warning: Workload experiments failed, continuing...")
    
    # Step 10: Failure analysis
    success = run_command(
        [sys.executable, "experiments/09_failure_analysis.py"],
        "Failure analysis (deadline miss, model crash, workload spike)"
    )
    if not success:
        print("\n⚠ Warning: Failure analysis failed, continuing...")
    
    # Step 11: Generate figures
    success = run_command(
        [sys.executable, "experiments/10_make_figures.py"],
        "Generate all figures (Pareto, hit-rate, ablations, etc.)"
    )
    if not success:
        print("\n⚠ Warning: Figure generation failed, continuing...")
    
    # Summary
    elapsed_total = time.time() - start_time_total
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"Total time: {elapsed_total/3600:.1f} hours ({elapsed_total/60:.1f} minutes)")
    print("\nResults saved to:")
    print("  - results/latency_profiles.csv")
    print("  - results/accuracy_profiles.csv")
    print("  - results/baseline_results.csv")
    print("  - results/planner_results.csv")
    print("  - results/statistical_tests.csv")
    print("  - results/pareto_analysis.csv")
    print("  - results/ablation_results.csv")
    print("  - results/workload_results.csv")
    print("  - results/failure_miss_analysis.csv")
    print("  - results/failure_degradation_strategies.csv")
    print("  - results/plots/*.png")
    print("\nNext steps:")
    print("  1. Review results in results/ directory")
    print("  2. Check plots in results/plots/")
    print("  3. Read paper/draft.md for detailed analysis")
    print("  4. Push to GitHub and start emailing supervisors!")
    print("="*80)

if __name__ == "__main__":
    main()
