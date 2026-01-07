# Anytime Inference Planner

Deadline-aware ML model serving system that dynamically selects model variant, quantization level, batch size, and device to meet latency targets while maximizing accuracy.

## Overview

Cascade-based planner that profiles configurations offline and selects optimal settings at runtime based on deadline constraints. Supports graceful degradation under load.

**Features:**
- Dynamic configuration selection (model, quantization, batch, device)
- Two-stage cascade inference for accuracy/latency tradeoff
- Real latency profiling (per-request traces, not synthetic)
- Workload sensitivity analysis (steady vs. bursty traffic)
- Graceful degradation strategies

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download datasets (SST-2, CIFAR-10)
python data/download_datasets.py

# Quick test (2-3 hours GPU, 4-6 hours CPU)
python run_all.py --quick-test

# Full evaluation (8-12 hours GPU)
python run_all.py
```

## How It Works

1. **Profile** latency and accuracy for all configurations offline
2. **Plan** at runtime: given deadline, select config that maximizes accuracy while meeting latency target
3. **Cascade** (optional): run small model first, escalate to large model only if confidence is low

```python
from src.planner import CascadePlanner
from src.profiler import LatencyProfiler, AccuracyProfiler

# Load profiles
latency_profiles = LatencyProfiler.load("results/latency_profiles.csv")
accuracy_profiles = AccuracyProfiler.load("results/accuracy_profiles.csv")

# Create planner
planner = CascadePlanner(
    latency_profiles=latency_profiles,
    accuracy_profiles=accuracy_profiles,
    cascade_threshold=0.9
)

# Select config for 100ms deadline
config = planner.select(deadline_ms=100)
# Returns: {"model": "distilbert", "quantization": "int8", "batch_size": 4, "device": "cpu"}
```

## Results

| Method | Hit Rate | Accuracy | Notes |
|--------|----------|----------|-------|
| StaticSmall | 0.95-0.98 | 0.80-0.82 | Always use small model |
| StaticLarge | 0.60-0.65 | 0.88-0.90 | Always use large model |
| ThroughputAutotuner | 0.82-0.87 | 0.84-0.86 | Optimize for throughput |
| INFaaS-style | 0.80-0.85 | 0.85-0.87 | Variant selection baseline |
| **CascadePlanner** | **0.88-0.93** | **0.86-0.88** | This implementation |

Improvements: +30-50% hit rate vs. StaticLarge, +5-8% accuracy vs. StaticSmall

## Models

| Model | Params | Task | Quantization |
|-------|--------|------|--------------|
| DistilBERT | 66M | Text (SST-2) | FP32, FP16, INT8 |
| MiniLM | 22M | Text (SST-2) | FP32, FP16, INT8 |
| MobileNetV2 | 3.5M | Vision (CIFAR-10) | FP32, FP16, INT8 |
| ResNet18 | 11M | Vision (CIFAR-10) | FP32, FP16, INT8 |

## Configuration

| Parameter | Options | Default |
|-----------|---------|---------|
| Devices | CPU, CUDA | CPU |
| Precision | FP32, FP16, INT8 | FP32 |
| Batch sizes | 1, 4, 8 | 1 |
| Cascade threshold | 0.7 - 0.95 | 0.9 |
| Deadlines | 50ms, 100ms, 150ms | 100ms |

## Architecture

```
anytime-inference-planner/
├── src/
│   ├── models/          # Model zoo and cascade logic
│   ├── profiler/        # Latency and accuracy profiling
│   ├── planner/         # CascadePlanner and baselines
│   ├── evaluation/      # Inference and statistical analysis
│   └── workloads/       # Traffic generation (Poisson, bursty)
├── experiments/         # Experimental scripts (01-10)
├── data/                # Dataset download scripts
├── results/             # Outputs (CSV, figures)
└── configs/             # Configuration files
```

## Experiments

Run in order:

```bash
python experiments/01_profile_latency.py
python experiments/02_profile_accuracy.py
python experiments/03_run_baselines.py
python experiments/04_run_planner.py
python experiments/05_ablation.py
python experiments/06_statistical_tests.py
python experiments/07_pareto_analysis.py
python experiments/08_workload.py
python experiments/09_failure_analysis.py
python experiments/10_make_figures.py
```

Or run all: `python run_all.py`

## Outputs

**CSV files** in `results/`:
- `latency_profiles.csv` - Per-config latency measurements
- `accuracy_profiles.csv` - Per-config accuracy measurements
- `planner_results.csv` - Planner evaluation results
- `ablation_results.csv` - Ablation studies
- `workload_sensitivity.csv` - Steady vs. bursty traffic

**Figures** in `results/figures/`:
- `01_pareto_frontier.png` - Latency vs accuracy tradeoff
- `02_baseline_comparison.png` - Hit rate and accuracy comparison
- `03_cascade_threshold.png` - Threshold sensitivity
- `05_workload_sensitivity.png` - Traffic pattern impact

## Testing

```bash
python run_tests.py
```

## Statistical Methodology

| Aspect | Approach |
|--------|----------|
| Primary metric | Deadline hit rate |
| Secondary metric | Accuracy |
| Significance level | α = 0.05 |
| Effect size | Cohen's d (small: 0.2, medium: 0.5, large: 0.8) |

**Tests:**
- Paired t-test (parametric)
- Wilcoxon signed-rank (non-parametric)
- Shapiro-Wilk (normality check)
- Levene (equal variance check)

**Reporting:** Mean ± 95% CI, effect sizes, p-values for both parametric and non-parametric tests.

Results in `results/statistical_tests.csv`.

## Limitations

- Offline profiling only (no online adaptation)
- Limited model zoo (4 models)
- Synthetic workloads (Poisson, bursty)
- Single-task evaluation

## License

MIT
