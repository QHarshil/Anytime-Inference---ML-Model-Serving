# Anytime Inference Planner

Deadline-aware configuration selection for ML inference systems.

## Overview

This project implements a cascade-based planner that jointly optimizes model variant, quantization, batch size, and device selection to meet latency deadlines while maximizing task accuracy. The system profiles configurations offline and selects optimal settings at runtime based on deadline constraints.

**Key contributions:**
- Real inference measurements (per-request latency traces, not synthetic sampling)
- Cascade evaluation with two-stage inference
- Statistical rigor (paired tests, effect sizes, confidence intervals)
- Ablation studies across model size, quantization, and batch size
- Trace-driven workload sensitivity analysis (steady vs. bursty traffic)
- Graceful degradation strategies grounded in recorded latencies

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

```bash
python data/download_datasets.py
```

This downloads:
- SST-2 (sentiment analysis, 872 validation examples)
- CIFAR-10 (image classification, 10,000 test examples)

### Run Experiments

**Quick test (2-3 hours on GPU, 4-6 hours on CPU):**
```bash
python run_all.py --quick-test
```

**Full evaluation (8-12 hours on GPU, 16-24 hours on CPU):**
```bash
python run_all.py
```

### Run Tests

```bash
python run_tests.py
```

## Repository Structure

```
anytime-inference-planner/
├── src/
│   ├── models/              # Model zoo and cascade logic
│   ├── profiler/            # Latency and accuracy profiling
│   ├── planner/             # Planner and baselines
│   ├── evaluation/          # Real inference and statistical analysis
│   ├── theory/              # Pareto analysis
│   ├── workloads/           # Workload generation
│   └── utils/               # IO, metrics, logging
├── experiments/             # 10 experimental scripts (01-10)
├── tests/                   # Unit and integration tests
├── training/                # Fine-tuning scripts (optional)
├── data/                    # Dataset download scripts
├── results/                 # Experimental outputs (generated)
├── configs/                 # Configuration files
└── paper/                   # Draft paper outline
```

## Experimental Pipeline

The pipeline consists of 10 experiments that must be run in order:

1. **01_profile_latency.py** - Profile latency for all configurations
2. **02_profile_accuracy.py** - Profile accuracy for all configurations
3. **03_run_baselines.py** - Evaluate baseline methods (StaticSmall, StaticLarge, ThroughputAutotuner, INFaaS-style)
4. **04_run_planner.py** - Evaluate CascadePlanner with different thresholds
5. **05_ablation.py** - Ablation studies (model size, quantization, batch size)
6. **06_statistical_tests.py** - Statistical significance testing
7. **07_pareto_analysis.py** - Pareto frontier analysis and dominance comparisons
8. **08_workload.py** - Workload sensitivity analysis
9. **09_failure_analysis.py** - Failure analysis and graceful degradation
10. **10_make_figures.py** - Generate all visualizations

## Results

Results are saved to `results/` directory:

**CSV files:**
- `latency_profiles.csv` - Latency measurements for all configurations
- `accuracy_profiles.csv` - Accuracy measurements for all configurations
- `baseline_results.csv` - Baseline method evaluations
- `planner_results.csv` - Planner evaluations
- `ablation_results.csv` - Ablation study results
- `statistical_tests.csv` - Statistical test results
- `pareto_analysis.csv` - Pareto analysis results
- `workload_sensitivity.csv` - Workload sensitivity results (trace-driven)
- `failure_miss_analysis.csv` - Deadline miss analysis
- `failure_degradation_strategies.csv` - Graceful degradation strategies

**Figures** (in `results/figures/`):
- `01_pareto_frontier.png` - Pareto frontier (latency vs accuracy)
- `02_baseline_comparison.png` - Baseline comparison (hit rate and accuracy)
- `03_cascade_threshold.png` - Cascade threshold sensitivity
- `04_ablation_studies.png` - Ablation study results
- `05_workload_sensitivity.png` - Workload sensitivity (steady vs. bursty)
- `06_statistical_significance.png` - Statistical significance (effect sizes)

## Expected Results

Results vary by hardware, random seeds, and dataset splits. Expected ranges:

| Method | Hit Rate | Accuracy | Hypervolume |
|--------|----------|----------|-------------|
| StaticSmall | 0.95-0.98 | 0.80-0.82 | 140-150 |
| StaticLarge | 0.60-0.65 | 0.88-0.90 | 95-105 |
| ThroughputAutotuner | 0.82-0.87 | 0.84-0.86 | 160-175 |
| INFaaS-style | 0.80-0.85 | 0.85-0.87 | 165-180 |
| **CascadePlanner** | **0.88-0.93** | **0.86-0.88** | **185-200** |

**Improvements:**
- Hit rate: +30-50% vs. StaticLarge, +5-10% vs. INFaaS-style
- Accuracy: +5-8% vs. StaticSmall, +1-3% vs. INFaaS-style
- Statistical significance: p < 0.05, Cohen's d > 0.5 (medium to large effect)

## Models and Datasets

### Text Models
- **DistilBERT** (66M params, fine-tuned on SST-2)
- **MiniLM** (22M params, fine-tuned on SST-2)

### Vision Models
- **MobileNetV2** (3.5M params, ImageNet pre-trained)
- **ResNet18** (11M params, ImageNet pre-trained)

### Datasets
- **SST-2**: Stanford Sentiment Treebank (binary sentiment classification)
- **CIFAR-10**: 10-class image classification

## Configuration Options

### Profiling
- **Devices**: CPU, GPU (CUDA)
- **Precision**: FP32, FP16 (GPU), INT8 (CPU)
- **Batch sizes**: 1, 4, 8
- **Cascade thresholds**: 0.7, 0.8, 0.9, 0.95

### Evaluation
- **Deadlines**: 50ms, 100ms, 150ms (configurable)
- **Seeds**: Multiple independent runs for statistical rigor
- **Workloads**: Steady (Poisson), Bursty (alternating rates)

## Statistical Methodology

### Pre-declared Protocol
- **Primary metric**: Deadline hit rate
- **Secondary metric**: Accuracy
- **Significance level**: α = 0.05
- **Effect size**: Cohen's d with interpretation (small/medium/large)

### Tests
- **Paired t-test**: Parametric comparison
- **Wilcoxon signed-rank test**: Non-parametric comparison
- **Assumption checking**: Normality (Shapiro-Wilk), Equal variance (Levene)
- **Power analysis**: Statistical power for each comparison

### Reporting
- Mean ± 95% confidence interval
- Effect sizes with interpretation
- p-values for both parametric and non-parametric tests

## Limitations

1. **Offline profiling**: Configurations are profiled offline and cached. The planner does not adapt online to changing workloads or model drift.

2. **Limited model zoo**: Only 2 text models and 2 vision models. Real systems would have larger model zoos.

3. **Synthetic workloads**: Workload sensitivity uses synthetic traces (Poisson, bursty). Real production traces would be more complex.

4. **Single-task evaluation**: Each task (text, vision) is evaluated independently. Multi-task scenarios are not considered.

5. **No online learning**: The cascade threshold is fixed per deadline. Adaptive threshold adjustment based on runtime feedback is future work.

## Future Work

1. **Online adaptation**: Implement online learning to adapt cascade thresholds based on runtime feedback.

2. **Multi-task scenarios**: Extend to scenarios where multiple tasks share resources.

3. **Larger model zoo**: Expand to include more models (BERT-base, ResNet-50, etc.).

4. **Real production traces**: Evaluate on real serving logs from production systems.

5. **RL-based planner**: Frame as Markov Decision Process (MDP) and use reinforcement learning for policy optimization.

6. **GPU scheduling**: Add GPU-specific optimizations (kernel fusion, memory management).

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{anytime-inference-planner,
  title={Anytime Inference Planner: Deadline-Aware Configuration Selection for ML Inference},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-username]/anytime-inference-planner}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue or contact [your-email].
