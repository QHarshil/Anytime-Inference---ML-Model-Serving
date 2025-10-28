"""Run accuracy profiling for all configurations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import yaml

from src.models.model_zoo import ModelZoo
from src.profiler.accuracy_profiler import AccuracyProfiler
from src.utils.io import write_dataframe
from src.utils.logger import configure_logger


def load_text_dataset(n: int = 32) -> tuple[List[str], np.ndarray]:
    texts = ["sample sentence" for _ in range(n)]
    labels = np.random.randint(0, 2, size=n)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--output", default="results/accuracy_profiles.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    configure_logger()
    model_zoo = ModelZoo()
    profiler = AccuracyProfiler(model_zoo)

    config = yaml.safe_load(Path(args.config).read_text())
    profiles = []
    texts, labels = load_text_dataset()

    for model_cfg in config.get("text", []):
        for variant in model_cfg.get("variants", []):
            try:
                profile = profiler.profile_text(model_cfg["name"], variant, args.device, texts, labels)
                profiles.append(profile)
            except Exception as exc:
                print(f"[WARN] Skipping {model_cfg['name']} {variant}: {exc}")

    if profiles:
        df = profiler.to_dataframe(profiles)
        write_dataframe(df, Path(args.output))
    else:
        print("[WARN] No accuracy profiles generated.")


if __name__ == "__main__":
    main()
