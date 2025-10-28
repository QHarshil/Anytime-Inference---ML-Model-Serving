# Anytime Inference Planner

**Deadline-aware configuration selection for efficient ML inference.**

This repository follows the structure defined in the "Anytime Inference Planner – Ultimate Complete Guide". It provides a research-grade implementation of an anytime planner that profiles multiple model configurations, selects the best option for a given latency deadline, and reports rigorous evaluation metrics.

## Repository Layout

The project is organised according to the guide:

```
anytime-inference-planner/
├── configs/                 # YAML configs (protocols, models, deadlines)
├── data/                    # Dataset download script and cache directory
├── experiments/             # Reproducible experiment entry points
├── notebooks/               # Analysis notebooks (placeholders)
├── paper/                   # Draft paper scaffold
├── results/                 # Generated CSVs and plots (placeholders)
├── src/                     # Core Python package
└── tests/                   # Unit tests
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download datasets:
   ```bash
   python data/download_datasets.py
   ```
3. Run profiling and evaluation (see `experiments/` for details).

## Experiments

The `experiments/` directory contains numbered scripts covering the full workflow: profiling, baseline evaluation, planner evaluation, ablations, statistical analysis, and figure generation. Each script is self-documented and can be run independently.

## Results

Generated CSV files and visualisations are stored under `results/`. The repository only ships with placeholders; run the experiments to populate the directory.

## License

This project is released under the MIT License. See `LICENSE` for details.
