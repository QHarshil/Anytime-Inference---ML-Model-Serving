"""Run latency profiling for all configurations."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure project root on import path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import List

import pandas as pd
import yaml

from src.models.model_zoo import ModelZoo
from src.profiler.latency_profiler import LatencyProfiler
from src.utils.io import write_dataframe
from src.utils.logger import configure_logger


def load_text_samples(n: int = 32) -> List[str]:
    """Return dummy text samples. In practice load SST-2 validation data."""

    return ["sample sentence" for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml", help="Model configuration YAML")
    parser.add_argument("--output", default="results/latency_profiles.csv", help="Output CSV path")
    parser.add_argument("--device", default="cpu", help="Device to profile on (cpu or cuda)")
    args = parser.parse_args()

    configure_logger()
    model_zoo = ModelZoo()
    profiler = LatencyProfiler(model_zoo)

    config = yaml.safe_load(Path(args.config).read_text())
    profiles = []

    texts = load_text_samples()

    for model_cfg in config.get("text", []):
        for variant in model_cfg.get("variants", []):
            try:
                profile = profiler.profile_text(model_cfg["name"], variant, args.device, texts)
                profiles.append(profile)
            except Exception as exc:
                print(f"[WARN] Skipping {model_cfg['name']} {variant}: {exc}")

    if profiles:
        df = profiler.to_dataframe(profiles)
        write_dataframe(df, Path(args.output))
    else:
        print("[WARN] No latency profiles generated.")


if __name__ == "__main__":
    main()
